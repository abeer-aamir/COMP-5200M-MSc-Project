from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import time
import urllib.request
import urllib.error
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .commands import CommandError, CommandResult, CommandRunner
from .config import PROJECT_ROOT, EnvironmentConfig, EnvironmentLock


class EnvironmentError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class CachePaths:
    kind: Path
    kubectl: Path
    calico_manifest: Path
    preparation_receipt: Path


class EnvironmentPreparer:
    def __init__(
        self,
        config: EnvironmentConfig,
        *,
        runner: CommandRunner | None = None,
        timeout_seconds: int = 300,
    ):
        self.config = config
        self.lock = config.lock
        self.runner = runner or CommandRunner()
        self.timeout_seconds = timeout_seconds

    def cache_paths(self) -> CachePaths:
        return CachePaths(
            kind=(
                self.config.cache_root
                / "kind"
                / self.lock.kind_version
                / self.lock.kind_filename
            ),
            kubectl=(
                self.config.cache_root
                / "kubectl"
                / self.lock.kubectl_version
                / self.lock.kubectl_filename
            ),
            calico_manifest=(
                self.config.cache_root
                / "calico"
                / self.lock.calico_version
                / self.lock.calico_filename
            ),
            preparation_receipt=self.config.cache_root / "environment-preparation.json",
        )

    @staticmethod
    def _require_windows_amd64() -> None:
        machine = platform.machine().lower()
        if platform.system() != "Windows" or machine not in {"amd64", "x86_64"}:
            raise EnvironmentError(
                "The frozen tool lock currently supports Windows amd64 only"
            )

    def _download(self, url: str, destination: Path, sha256: str) -> None:
        if destination.is_file() and _sha256_file(destination) == sha256:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "AIPyCraft-Kubernetes-Baseline/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with partial.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            actual = _sha256_file(partial)
            if actual != sha256:
                raise EnvironmentError(
                    f"Checksum mismatch for {url}: expected {sha256}, got {actual}"
                )
            partial.replace(destination)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EnvironmentError(f"Could not download locked artifact {url}: {exc}") from exc
        finally:
            partial.unlink(missing_ok=True)

    def prepare(self) -> dict[str, Any]:
        """Download locked tools and pull images. This never calls an LLM API."""

        self._require_windows_amd64()
        paths = self.cache_paths()
        self._download(self.lock.kind_url, paths.kind, self.lock.kind_sha256)
        self._download(
            self.lock.kubectl_url, paths.kubectl, self.lock.kubectl_sha256
        )
        self._download(
            self.lock.calico_manifest_url,
            paths.calico_manifest,
            self.lock.calico_manifest_sha256,
        )
        docker_identity = self._docker_identity()
        pulled: list[dict[str, str]] = []
        self.runner.run(
            ["docker", "pull", self.lock.node_image_source],
            timeout=self.timeout_seconds,
        )
        source_image_id = self._inspect_image_id(self.lock.node_image_source)
        expected_source_id = self.lock.node_image_source.rsplit("@", 1)[1]
        if source_image_id != expected_source_id:
            raise EnvironmentError(
                "Locked base node image resolved to "
                f"{source_image_id}, expected {expected_source_id}"
            )
        self.runner.run(
            [
                "docker",
                "build",
                "--network",
                "none",
                "--provenance=false",
                "--tag",
                self.lock.node_image,
                "--file",
                self.lock.node_image_dockerfile,
                PROJECT_ROOT,
            ],
            timeout=self.timeout_seconds,
        )
        node_image_id = self._inspect_image_id(self.lock.node_image)
        for image in self.lock.preload_images:
            self.runner.run(
                ["docker", "pull", image.source], timeout=self.timeout_seconds
            )
            self.runner.run(
                ["docker", "tag", image.source, image.tag], timeout=30
            )
            pulled.append(
                {"tag": image.tag, "source": image.source, "digest": image.digest}
            )
        receipt = self._preparation_receipt(
            source_image_id=source_image_id,
            node_image_id=node_image_id,
        )
        _write_json(paths.preparation_receipt, receipt)
        return {
            "status": "prepared",
            "kind": str(paths.kind),
            "kubectl": str(paths.kubectl),
            "calico_manifest": str(paths.calico_manifest),
            "docker": docker_identity,
            "node_image": {
                "tag": self.lock.node_image,
                "source": self.lock.node_image_source,
                "id": node_image_id,
            },
            "preparation_receipt": str(paths.preparation_receipt),
            "preloaded_images": pulled,
        }

    def _preparation_receipt(
        self, *, source_image_id: str, node_image_id: str
    ) -> dict[str, Any]:
        """Bind a machine-local build output to the repository's locked inputs."""

        return {
            "schema_version": 1,
            "environment_lock_sha256": _sha256_file(self.lock.path),
            "node_image": {
                "tag": self.lock.node_image,
                "source": self.lock.node_image_source,
                "source_id": source_image_id,
                "id": node_image_id,
                "dockerfile_sha256": _sha256_file(
                    self.lock.node_image_dockerfile
                ),
                "entrypoint_sha256": _sha256_file(
                    self.lock.node_image_entrypoint
                ),
            },
        }

    def _load_preparation_receipt(self, path: Path) -> dict[str, Any]:
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EnvironmentError(
                f"Missing machine preparation receipt {path}; run prepare first"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvironmentError(
                f"Could not read machine preparation receipt {path}: {exc}"
            ) from exc
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema_version",
            "environment_lock_sha256",
            "node_image",
        }:
            raise EnvironmentError("Machine preparation receipt has invalid fields")
        image = receipt.get("node_image")
        expected_image_fields = {
            "tag",
            "source",
            "source_id",
            "id",
            "dockerfile_sha256",
            "entrypoint_sha256",
        }
        if (
            receipt.get("schema_version") != 1
            or not isinstance(image, dict)
            or set(image) != expected_image_fields
        ):
            raise EnvironmentError("Machine preparation receipt has an invalid schema")
        return receipt

    def _inspect_image_id(self, reference: str) -> str:
        result = self.runner.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
            timeout=30,
        )
        image_id = result.stdout.strip()
        if not image_id:
            raise EnvironmentError(f"Docker returned no image id for {reference}")
        return image_id

    def _docker_identity(self) -> dict[str, Any]:
        context_result = self.runner.run(["docker", "context", "show"], timeout=30)
        context = context_result.stdout.strip()
        endpoint_result = self.runner.run(
            [
                "docker",
                "context",
                "inspect",
                context,
                "--format",
                "{{json .Endpoints.docker.Host}}",
            ],
            timeout=30,
        )
        endpoint = endpoint_result.stdout.strip().strip('"')
        server_result = self.runner.run(
            ["docker", "version", "--format", "{{json .Server}}"], timeout=30
        )
        try:
            server = json.loads(server_result.stdout)
        except json.JSONDecodeError as exc:
            raise EnvironmentError("Docker did not return parseable server identity") from exc
        if not isinstance(server, dict):
            raise EnvironmentError("Docker server identity was not an object")
        server_os = str(server.get("Os", "")).lower()
        server_arch = str(server.get("Arch", "")).lower()
        if server_os != "linux" or server_arch not in {"amd64", "x86_64"}:
            raise EnvironmentError(
                "The frozen harness requires a Linux/amd64 Docker engine"
            )
        if not endpoint.lower().startswith("npipe://"):
            raise EnvironmentError(
                "The frozen Windows harness requires a local Docker named-pipe "
                "context; remote Docker contexts are refused"
            )
        return {
            "context": context,
            "endpoint": endpoint,
            "server_version": server.get("Version"),
            "server_os": server_os,
            "server_arch": server_arch,
        }

    def doctor(self) -> dict[str, Any]:
        """Verify local prerequisites without downloading or making model calls."""

        self._require_windows_amd64()
        paths = self.cache_paths()
        checks: dict[str, Any] = {}
        for label, path, expected in (
            ("kind", paths.kind, self.lock.kind_sha256),
            ("kubectl", paths.kubectl, self.lock.kubectl_sha256),
            ("calico_manifest", paths.calico_manifest, self.lock.calico_manifest_sha256),
            (
                "node_image_dockerfile",
                self.lock.node_image_dockerfile,
                self.lock.node_image_dockerfile_sha256,
            ),
            (
                "node_image_entrypoint",
                self.lock.node_image_entrypoint,
                self.lock.node_image_entrypoint_sha256,
            ),
            ("kind_config", self.config.kind_config, self.lock.kind_config_sha256),
        ):
            if not path.is_file():
                raise EnvironmentError(
                    f"Missing {label} cache file {path}; run the prepare command first"
                )
            actual = _sha256_file(path)
            if actual != expected:
                raise EnvironmentError(
                    f"Checksum mismatch for cached {label}: expected {expected}, got {actual}"
                )
            checks[label] = {"path": str(path), "sha256": actual}

        kubectl = self.runner.run(
            [paths.kubectl, "version", "--client", "-o", "json"], timeout=30
        )
        checks["docker"] = self._docker_identity()
        try:
            checks["kubectl_client"] = json.loads(kubectl.stdout)
        except json.JSONDecodeError:
            raise EnvironmentError("The locked kubectl did not return JSON")
        version = checks["kubectl_client"].get("clientVersion", {}).get("gitVersion")
        if version != self.lock.kubectl_version:
            raise EnvironmentError(
                f"Expected kubectl {self.lock.kubectl_version}, got {version!r}"
            )
        source_id = self._inspect_image_id(self.lock.node_image_source)
        source_digest = self.lock.node_image_source.rsplit("@", 1)[1]
        if source_id != source_digest:
            raise EnvironmentError(
                f"Locked base node image resolved to {source_id}, expected {source_digest}"
            )
        node_image_id = self._inspect_image_id(self.lock.node_image)
        receipt = self._load_preparation_receipt(paths.preparation_receipt)
        expected_receipt = self._preparation_receipt(
            source_image_id=source_id,
            node_image_id=node_image_id,
        )
        if receipt != expected_receipt:
            raise EnvironmentError(
                "The local kind image or its locked build inputs changed after prepare; "
                "run prepare again"
            )
        checks["node_image"] = {
            "reference": self.lock.node_image,
            "source": self.lock.node_image_source,
            "source_id": source_id,
            "id": node_image_id,
            "preparation_receipt": str(paths.preparation_receipt),
            "preparation_receipt_sha256": _sha256_file(
                paths.preparation_receipt
            ),
        }
        image_checks: list[dict[str, str]] = []
        for image in self.lock.preload_images:
            tag_id = self._inspect_image_id(image.tag)
            source_id = self._inspect_image_id(image.source)
            if tag_id != source_id:
                raise EnvironmentError(
                    f"Local tag {image.tag} does not resolve to locked digest {image.digest}"
                )
            image_checks.append(
                {
                    "tag": image.tag,
                    "source": image.source,
                    "digest": image.digest,
                    "id": tag_id,
                }
            )
        checks["preload_images"] = image_checks
        manifest_images = set(
            re.findall(
                r"(?m)^\s*image:\s*[\"']?([^\s\"']+)",
                paths.calico_manifest.read_text(encoding="utf-8"),
            )
        )
        expected_calico = {
            image.tag for image in self.lock.preload_images if "calico" in image.tag
        }
        if manifest_images != expected_calico:
            raise EnvironmentError(
                "Locked Calico manifest image references do not match preloaded tags: "
                f"manifest={sorted(manifest_images)}, expected={sorted(expected_calico)}"
            )
        checks["calico_manifest_images"] = sorted(manifest_images)
        return {"status": "ready", "checks": checks}


@dataclass
class AttemptEnvironment:
    cluster_name: str
    network_name: str
    context: str
    kubeconfig: Path
    attempt_dir: Path
    kubectl_path: str
    runner: CommandRunner
    command_timeout_seconds: int
    isolation_inspection: dict[str, Any] | None = None

    @property
    def node_name(self) -> str:
        return f"{self.cluster_name}-control-plane"

    @property
    def container_kubeconfig(self) -> str:
        return "/etc/kubernetes/admin.conf"

    @property
    def kubectl_context(self) -> str:
        return self.context

    def kubectl_prefix(self) -> list[str]:
        return [
            "docker",
            "exec",
            "-i",
            self.node_name,
            "kubectl",
            "--kubeconfig",
            self.container_kubeconfig,
            "--context",
            self.kubectl_context,
        ]

    def _prepare_input(self, path: Path) -> tuple[str, str, int]:
        source = path.resolve()
        if not source.is_file():
            raise EnvironmentError(f"kubectl input file does not exist: {source}")
        content = source.read_text(encoding="utf-8")
        return content, _sha256_file(source), len(content.encode("utf-8"))

    def kubectl(
        self,
        args: list[str | Path],
        *,
        check: bool = True,
        timeout: int | None = None,
    ) -> CommandResult:
        rendered_args: list[str | Path] = []
        input_text: str | None = None
        input_audit: dict[str, Any] | None = None
        for item in args:
            if isinstance(item, Path):
                if input_text is not None:
                    raise EnvironmentError("Only one file input is permitted per kubectl command")
                input_text, digest, byte_count = self._prepare_input(item)
                rendered_args.append("-")
                input_audit = {
                    "source": str(item.resolve()),
                    "source_sha256": digest,
                    "bytes": byte_count,
                    "method": "stdin",
                }
            else:
                rendered_args.append(item)
        result = self.runner.run(
            [*self.kubectl_prefix(), *rendered_args],
            timeout=timeout or self.command_timeout_seconds,
            check=check,
            input_text=input_text,
        )
        if input_audit is not None:
            input_audit["command"] = result.audit_dict()
            transfer_path = self.attempt_dir / "environment_file_transfers.jsonl"
            with transfer_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(input_audit, sort_keys=True) + "\n")
        return result

    def kubectl_json(self, args: list[str]) -> dict[str, Any]:
        result = self.kubectl([*args, "-o", "json"])
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise EnvironmentError("kubectl did not return JSON") from exc
        if not isinstance(value, dict):
            raise EnvironmentError("kubectl JSON response was not an object")
        return value

    def deploy(self, candidate_path: Path) -> CommandResult:
        try:
            return self.kubectl(
                ["apply", "--validate=false", "-f", candidate_path],
                check=False,
                timeout=self.command_timeout_seconds,
            )
        except CommandError as exc:
            return exc.result

    def readyz(self) -> CommandResult:
        return self.kubectl(["get", "--raw=/readyz"], check=False, timeout=30)

    def metadata(self, lock: EnvironmentLock) -> dict[str, Any]:
        return {
            "cluster_name": self.cluster_name,
            "network_name": self.network_name,
            "context": self.context,
            "api_access": "docker_exec_only",
            "container_kubeconfig": self.container_kubeconfig,
            "container_kubectl_context": self.kubectl_context,
            "host_kubeconfig": None,
            "isolation_inspection": self.isolation_inspection,
            "kind_version": lock.kind_version,
            "kubectl_version": lock.kubectl_version,
            "node_image": lock.node_image,
            "kubernetes_minor": lock.kubernetes_minor,
            "kind_config_sha256": lock.kind_config_sha256,
            "environment_lock_sha256": _sha256_file(lock.path),
            "docker_network_subnet": lock.docker_network_subnet,
            "docker_network_gateway": lock.docker_network_gateway,
            "calico_version": lock.calico_version,
            "calico_manifest_sha256": lock.calico_manifest_sha256,
            "preload_images": [image.tag for image in lock.preload_images],
        }


_SAFE_CLUSTER = re.compile(r"\Aaipc-[a-z0-9-]{1,42}-a[1-9][0-9]*\Z")
_SAFE_NETWORK = re.compile(r"\Aaipc-[a-z0-9-]{1,42}-a[1-9][0-9]*-net\Z")
_NETWORK_DISPOSABLE_KEY = "org.aipycraft.disposable"
_NETWORK_OWNER_KEY = "org.aipycraft.owner"


def _kind_create_acceptance_basis(result: CommandResult) -> str | None:
    """Allow the intentionally unpublished API only with later live verification."""

    if result.returncode == 124:
        return "timeout_pending_mandatory_in_container_api_verification"
    detail = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 and "failed to get api server port" in detail:
        return "expected_unpublished_api_port_error"
    return None


@dataclass
class _AttemptOwnership:
    owner_token: str
    network_acquired: bool = False
    cluster_create_started: bool = False


class IsolatedKindHarness:
    def __init__(
        self,
        config: EnvironmentConfig,
        *,
        command_timeout_seconds: int,
        max_attempts: int,
        runner: CommandRunner | None = None,
        kubectl_path: str | Path | None = None,
    ):
        if isinstance(max_attempts, bool) or max_attempts < 1:
            raise EnvironmentError("max_attempts must be a positive integer")
        self.config = config
        self.lock = config.lock
        self.command_timeout_seconds = command_timeout_seconds
        self.kind_create_timeout_seconds = max(
            300, command_timeout_seconds * 3
        )
        self.max_attempts = max_attempts
        self.runner = runner or CommandRunner()
        self.preparer = EnvironmentPreparer(config, runner=self.runner)
        self.kubectl_path = str(
            kubectl_path or self.preparer.cache_paths().kubectl
        )

    def preflight(self) -> dict[str, Any]:
        return self.preparer.doctor()

    def smoke(self) -> dict[str, Any]:
        """Create, test, and delete a real cluster without making a model call."""

        preflight = self.preflight()
        run_id = f"smoke-{uuid.uuid4().hex[:16]}"
        output_dir = self.config.cache_root / "smoke-runs" / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        with self.attempt(run_id, 1, output_dir) as env:
            ready = env.readyz()
            if ready.returncode != 0 or "ok" not in ready.stdout.lower():
                raise EnvironmentError(
                    ready.stderr.strip()
                    or ready.stdout.strip()
                    or "The disposable API server did not pass readyz"
                )
            metadata = env.metadata(self.lock)
        return {
            "status": "passed",
            "paid_calls_made": False,
            "preflight": preflight,
            "environment": metadata,
            "artifacts": str(output_dir),
            "cleanup": json.loads(
                (output_dir / "cleanup.json").read_text(encoding="utf-8")
            ),
        }

    @staticmethod
    def _names(run_id: str, attempt_number: int) -> tuple[str, str]:
        token = re.sub(r"[^a-z0-9]", "", run_id.lower())[-24:]
        cluster = f"aipc-{token}-a{attempt_number}"
        network = f"{cluster}-net"
        if not _SAFE_CLUSTER.fullmatch(cluster) or not _SAFE_NETWORK.fullmatch(network):
            raise EnvironmentError("Could not create safe disposable resource names")
        return cluster, network

    def _wait_pod_phase(
        self,
        env: AttemptEnvironment,
        namespace: str,
        pod: str,
        expected: str,
        timeout_seconds: int,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_phase = "<absent>"
        while time.monotonic() < deadline:
            result = env.kubectl(
                ["get", "pod", pod, "-n", namespace, "-o", "json"],
                check=False,
                timeout=20,
            )
            if result.returncode == 0:
                try:
                    last_phase = json.loads(result.stdout).get("status", {}).get(
                        "phase", "<unknown>"
                    )
                except json.JSONDecodeError:
                    last_phase = "<invalid-json>"
                if last_phase == expected:
                    return
                if last_phase == "Failed" and expected != "Failed":
                    logs = env.kubectl(
                        ["logs", pod, "-n", namespace], check=False, timeout=20
                    )
                    raise EnvironmentError(
                        f"Harness pod {pod} failed: {logs.stdout.strip()} "
                        f"{logs.stderr.strip()}"
                    )
            time.sleep(1)
        raise EnvironmentError(
            f"Timed out waiting for harness pod {pod} phase {expected}; "
            f"last phase={last_phase}"
        )

    def _smoke_test(self, env: AttemptEnvironment) -> dict[str, Any]:
        namespace = "aipc-harness-smoke"
        manifest = env.attempt_dir / "harness-smoke.yaml"
        manifest.write_text(
            """apiVersion: v1
kind: Namespace
metadata:
  name: aipc-harness-smoke
---
apiVersion: v1
kind: Pod
metadata:
  name: server
  namespace: aipc-harness-smoke
  labels:
    app: server
spec:
  containers:
    - name: server
      image: busybox:1.36.1
      imagePullPolicy: Never
      command: [\"sh\", \"-ec\", \"mkdir -p /www; echo ok > /www/index.html; httpd -f -p 8080 -h /www\"]
  restartPolicy: Never
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: smoke-data
  namespace: aipc-harness-smoke
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 16Mi
---
apiVersion: v1
kind: Pod
metadata:
  name: storage-probe
  namespace: aipc-harness-smoke
spec:
  containers:
    - name: probe
      image: busybox:1.36.1
      imagePullPolicy: Never
      command:
        - sh
        - -ec
        - |
          echo storage-ok > /data/probe
          test "$(cat /data/probe)" = storage-ok
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: smoke-data
  restartPolicy: Never
---
apiVersion: v1
kind: Service
metadata:
  name: server
  namespace: aipc-harness-smoke
spec:
  selector:
    app: server
  ports:
    - port: 8080
      targetPort: 8080
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-labelled-server-ingress
  namespace: aipc-harness-smoke
spec:
  podSelector:
    matchLabels:
      app: server
  policyTypes: [Ingress]
  ingress:
    - from:
        - podSelector:
            matchLabels:
              access: smoke
      ports:
        - protocol: TCP
          port: 8080
""",
            encoding="utf-8",
        )
        applied = env.kubectl(["apply", "-f", manifest], timeout=60)
        try:
            env.kubectl(
                [
                    "wait",
                    "--for=condition=Ready",
                    "pod/server",
                    "-n",
                    namespace,
                    "--timeout=60s",
                ],
                timeout=70,
            )
            env.kubectl(
                [
                    "wait",
                    "--for=jsonpath={.status.phase}=Bound",
                    "pvc/smoke-data",
                    "-n",
                    namespace,
                    "--timeout=90s",
                ],
                timeout=100,
            )
            self._wait_pod_phase(env, namespace, "storage-probe", "Succeeded", 90)
            allowed_probe = env.kubectl(
                [
                    "run",
                    "allowed-network-probe",
                    "-n",
                    namespace,
                    "--image=busybox:1.36.1",
                    "--image-pull-policy=Never",
                    "--restart=Never",
                    "--labels=access=smoke",
                    "--command",
                    "--",
                    "sh",
                    "-ec",
                    "test \"$(wget -T 8 -qO- http://server:8080)\" = ok",
                ],
                timeout=30,
            )
            self._wait_pod_phase(
                env, namespace, "allowed-network-probe", "Succeeded", 45
            )
            denied_probe = env.kubectl(
                [
                    "run",
                    "denied-network-probe",
                    "-n",
                    namespace,
                    "--image=busybox:1.36.1",
                    "--image-pull-policy=Never",
                    "--restart=Never",
                    "--command",
                    "--",
                    "sh",
                    "-ec",
                    "if wget -T 4 -qO- http://server:8080 >/dev/null 2>&1; then exit 42; else exit 0; fi",
                ],
                timeout=30,
            )
            self._wait_pod_phase(
                env, namespace, "denied-network-probe", "Succeeded", 45
            )
            egress_probe = env.kubectl(
                [
                    "run",
                    "external-egress-probe",
                    "-n",
                    namespace,
                    "--image=busybox:1.36.1",
                    "--image-pull-policy=Never",
                    "--restart=Never",
                    "--command",
                    "--",
                    "sh",
                    "-ec",
                    "if wget -T 4 -qO- http://1.1.1.1 >/dev/null 2>&1; then exit 42; else exit 0; fi",
                ],
                timeout=30,
            )
            self._wait_pod_phase(
                env, namespace, "external-egress-probe", "Succeeded", 35
            )
            return {
                "status": "passed",
                "allowed_service_probe": allowed_probe.audit_dict(),
                "denied_network_policy_probe": denied_probe.audit_dict(),
                "storage_probe": {"pvc_phase": "Bound", "pod_phase": "Succeeded"},
                "external_egress_probe": egress_probe.audit_dict(),
                "apply": applied.audit_dict(),
            }
        finally:
            env.kubectl(
                ["delete", "namespace", namespace, "--wait=true", "--timeout=60s"],
                check=False,
                timeout=70,
            )

    def _precheck_absent(self, cluster: str, network: str, kind_path: Path) -> None:
        def definitely_missing(result: CommandResult) -> bool:
            detail = f"{result.stdout}\n{result.stderr}".lower()
            return result.returncode == 1 and (
                "no such" in detail or "not found" in detail
            )

        network_check = self.runner.run(
            ["docker", "network", "inspect", network], timeout=20, check=False
        )
        if network_check.returncode == 0:
            raise EnvironmentError(f"Refusing existing Docker network {network}")
        if not definitely_missing(network_check):
            raise EnvironmentError(
                f"Could not safely establish that Docker network {network} is absent"
            )
        node_check = self.runner.run(
            ["docker", "container", "inspect", f"{cluster}-control-plane"],
            timeout=20,
            check=False,
        )
        if node_check.returncode == 0:
            raise EnvironmentError(f"Refusing existing kind node for {cluster}")
        if not definitely_missing(node_check):
            raise EnvironmentError(
                f"Could not safely establish that kind node {cluster}-control-plane is absent"
            )
        clusters = self.runner.run(
            [kind_path, "get", "clusters"], timeout=30, check=False
        )
        if clusters.returncode != 0:
            raise EnvironmentError("Could not enumerate existing kind clusters safely")
        if cluster in {line.strip() for line in clusters.stdout.splitlines()}:
            raise EnvironmentError(f"Refusing existing kind cluster {cluster}")

    def _inspect_isolation(
        self, cluster: str, network: str, owner_token: str
    ) -> dict[str, Any]:
        network_result = self.runner.run(
            ["docker", "network", "inspect", network], timeout=30
        )
        node_name = f"{cluster}-control-plane"
        node_result = self.runner.run(
            ["docker", "container", "inspect", node_name], timeout=30
        )
        try:
            network_data = json.loads(network_result.stdout)[0]
            node_data = json.loads(node_result.stdout)[0]
        except (json.JSONDecodeError, IndexError, TypeError) as exc:
            raise EnvironmentError("Docker isolation inspection was malformed") from exc

        labels = network_data.get("Labels") or {}
        ipam_configs = (network_data.get("IPAM") or {}).get("Config") or []
        subnet_match = any(
            item.get("Subnet") == self.lock.docker_network_subnet
            and item.get("Gateway") == self.lock.docker_network_gateway
            for item in ipam_configs
        )
        if (
            network_data.get("Internal") is not True
            or labels.get(_NETWORK_DISPOSABLE_KEY) != "true"
            or labels.get(_NETWORK_OWNER_KEY) != owner_token
            or not subnet_match
        ):
            raise EnvironmentError("Docker network failed locked isolation checks")

        attached_networks = (node_data.get("NetworkSettings") or {}).get(
            "Networks"
        ) or {}
        if set(attached_networks) != {network}:
            raise EnvironmentError(
                f"kind node has unexpected network attachments: {sorted(attached_networks)}"
            )
        node_labels = (node_data.get("Config") or {}).get("Labels") or {}
        if node_labels.get("io.x-k8s.kind.cluster") != cluster:
            raise EnvironmentError("Docker node does not belong to the expected kind cluster")
        if (node_data.get("State") or {}).get("Running") is not True:
            raise EnvironmentError("The isolated kind control-plane container is not running")

        requested_bindings = (node_data.get("HostConfig") or {}).get("PortBindings") or {}
        if set(requested_bindings) - {"6443/tcp"}:
            raise EnvironmentError(
                "The isolated kind node requested unexpected host ports: "
                f"{sorted(requested_bindings)}"
            )
        port_state = (node_data.get("NetworkSettings") or {}).get("Ports") or {}
        published_ports = {
            port: bindings for port, bindings in port_state.items() if bindings
        }
        if published_ports:
            raise EnvironmentError(
                "The isolated kind node unexpectedly publishes host ports: "
                f"{sorted(published_ports)}"
            )

        mounts = node_data.get("Mounts") or []
        unexpected_binds = [
            item
            for item in mounts
            if item.get("Type") == "bind" and item.get("Destination") != "/lib/modules"
        ]
        if unexpected_binds:
            raise EnvironmentError(
                "kind node has unexpected host bind mounts: "
                f"{[item.get('Destination') for item in unexpected_binds]}"
            )
        return {
            "network_internal": True,
            "network_subnet": self.lock.docker_network_subnet,
            "network_gateway": self.lock.docker_network_gateway,
            "node_networks": sorted(attached_networks),
            "requested_host_port_bindings": requested_bindings,
            "published_ports": published_ports,
            "mounts": [
                {
                    "type": item.get("Type"),
                    "source": item.get("Source"),
                    "destination": item.get("Destination"),
                    "rw": item.get("RW"),
                }
                for item in mounts
            ],
        }

    def _setup(
        self,
        run_id: str,
        attempt_number: int,
        attempt_dir: Path,
        ownership: _AttemptOwnership,
    ) -> AttemptEnvironment:
        cluster, network = self._names(run_id, attempt_number)
        cache = self.preparer.cache_paths()
        kubeconfig = attempt_dir / "kubeconfig"
        context = f"kind-{cluster}"
        self._precheck_absent(cluster, network, cache.kind)
        self.runner.run(
            [
                "docker",
                "network",
                "create",
                "--driver",
                "bridge",
                "--internal",
                "--subnet",
                self.lock.docker_network_subnet,
                "--gateway",
                self.lock.docker_network_gateway,
                "--label",
                f"{_NETWORK_DISPOSABLE_KEY}=true",
                "--label",
                f"{_NETWORK_OWNER_KEY}={ownership.owner_token}",
                network,
            ],
            timeout=30,
        )
        ownership.network_acquired = True
        ownership.cluster_create_started = True
        create_result = self.runner.run(
            [
                cache.kind,
                "create",
                "cluster",
                "--retain",
                "--name",
                cluster,
                "--image",
                self.lock.node_image,
                "--config",
                self.config.kind_config,
                "--kubeconfig",
                kubeconfig,
            ],
            timeout=self.kind_create_timeout_seconds,
            check=False,
            env={"KIND_EXPERIMENTAL_DOCKER_NETWORK": network},
        )
        acceptance_basis = _kind_create_acceptance_basis(create_result)
        accepted_unpublished_api = acceptance_basis is not None
        _write_json(
            attempt_dir / "kind_create.json",
            {
                "command": create_result.audit_dict(),
                "accepted_unpublished_api_result": accepted_unpublished_api,
                "acceptance_basis": acceptance_basis,
            },
        )
        if create_result.returncode != 0 and not accepted_unpublished_api:
            raise CommandError(create_result)
        context_result = self.runner.run(
            [
                "docker",
                "exec",
                f"{cluster}-control-plane",
                "kubectl",
                "--kubeconfig",
                "/etc/kubernetes/admin.conf",
                "config",
                "current-context",
            ],
            timeout=30,
        )
        context = context_result.stdout.strip()
        if not re.fullmatch(r"kubernetes-admin@[A-Za-z0-9._-]+", context):
            raise EnvironmentError(
                f"Refusing unexpected in-node kubectl context {context!r}"
            )
        _write_json(
            attempt_dir / "container_api_access.json",
            {
                "method": "docker_exec",
                "node": f"{cluster}-control-plane",
                "kubeconfig": "/etc/kubernetes/admin.conf",
                "context": context,
                "context_discovery": context_result.audit_dict(),
            },
        )
        env = AttemptEnvironment(
            cluster_name=cluster,
            network_name=network,
            context=context,
            kubeconfig=kubeconfig,
            attempt_dir=attempt_dir,
            kubectl_path=self.kubectl_path,
            runner=self.runner,
            command_timeout_seconds=self.command_timeout_seconds,
        )
        env.isolation_inspection = self._inspect_isolation(
            cluster, network, ownership.owner_token
        )
        ready = env.readyz()
        if ready.returncode != 0 or "ok" not in ready.stdout.lower():
            raise EnvironmentError(
                "The unexposed Kubernetes API was not ready through docker exec: "
                + (ready.stderr.strip() or ready.stdout.strip() or "no output")
            )
        self.runner.run(
            [
                cache.kind,
                "load",
                "docker-image",
                *[image.tag for image in self.lock.preload_images],
                "--name",
                cluster,
            ],
            timeout=180,
            env={"KIND_EXPERIMENTAL_DOCKER_NETWORK": network},
        )
        env.kubectl(["create", "-f", cache.calico_manifest], timeout=120)
        env.kubectl(
            [
                "wait",
                "--for=condition=Available",
                "deployment/calico-kube-controllers",
                "-n",
                "kube-system",
                "--timeout=180s",
            ],
            timeout=195,
        )
        env.kubectl(
            [
                "wait",
                "--for=condition=Ready",
                "pod",
                "-l",
                "k8s-app=calico-node",
                "-n",
                "kube-system",
                "--timeout=180s",
            ],
            timeout=195,
        )
        env.kubectl(
            ["wait", "--for=condition=Ready", "nodes", "--all", "--timeout=120s"],
            timeout=135,
        )
        env.kubectl(
            [
                "wait",
                "--for=condition=Available",
                "deployment/coredns",
                "-n",
                "kube-system",
                "--timeout=120s",
            ],
            timeout=135,
        )
        env.kubectl(
            [
                "wait",
                "--for=condition=Available",
                "deployment/local-path-provisioner",
                "-n",
                "local-path-storage",
                "--timeout=120s",
            ],
            timeout=135,
        )
        storage_class = env.kubectl_json(["get", "storageclass", "standard"])
        annotations = storage_class.get("metadata", {}).get("annotations", {})
        if (
            storage_class.get("provisioner") != "rancher.io/local-path"
            or annotations.get("storageclass.kubernetes.io/is-default-class") != "true"
        ):
            raise EnvironmentError("The locked default dynamic StorageClass is unavailable")
        smoke = self._smoke_test(env)
        _write_json(attempt_dir / "environment_smoke.json", smoke)
        _write_json(attempt_dir / "environment.json", env.metadata(self.lock))
        return env

    def _cleanup(
        self,
        cluster: str,
        network: str,
        attempt_dir: Path,
        *,
        ownership: _AttemptOwnership,
        export_logs: bool,
    ) -> dict[str, Any]:
        if not _SAFE_CLUSTER.fullmatch(cluster) or not _SAFE_NETWORK.fullmatch(network):
            raise EnvironmentError("Refusing cleanup outside disposable name prefixes")
        cache = self.preparer.cache_paths()
        report: dict[str, Any] = {
            "cluster": cluster,
            "network": network,
            "network_acquired": ownership.network_acquired,
            "cluster_create_started": ownership.cluster_create_started,
            "warnings": [],
            "cleanup_errors": [],
        }

        def run_cleanup(
            label: str, argv: list[str | Path], timeout: int
        ) -> CommandResult | None:
            try:
                result = self.runner.run(
                    argv, timeout=timeout, check=False
                )
                report[label] = result.audit_dict()
                return result
            except Exception as exc:
                if isinstance(exc, CommandError):
                    report[label] = exc.result.audit_dict()
                else:
                    report[label] = {"error": f"{type(exc).__name__}: {exc}"}
                return None

        if export_logs and ownership.cluster_create_started and cache.kind.is_file():
            logs = run_cleanup(
                "export_logs",
                [
                    cache.kind,
                    "export",
                    "logs",
                    attempt_dir / "kind-logs",
                    "--name",
                    cluster,
                ],
                120,
            )
            if logs is None or logs.returncode != 0:
                report["warnings"].append("kind log export did not complete")

        if ownership.cluster_create_started:
            if cache.kind.is_file():
                deleted = run_cleanup(
                    "delete_cluster",
                    [cache.kind, "delete", "cluster", "--name", cluster],
                    90,
                )
                if deleted is None or deleted.returncode != 0:
                    node_name = f"{cluster}-control-plane"
                    node_inspect = run_cleanup(
                        "fallback_node_inspect",
                        [
                            "docker",
                            "container",
                            "inspect",
                            "--format",
                            "{{json .Config.Labels}}",
                            node_name,
                        ],
                        20,
                    )
                    node_removed = False
                    if node_inspect is not None and node_inspect.returncode == 0:
                        try:
                            node_labels = json.loads(node_inspect.stdout)
                        except json.JSONDecodeError:
                            node_labels = {}
                        if node_labels.get("io.x-k8s.kind.cluster") == cluster:
                            direct = run_cleanup(
                                "fallback_remove_node",
                                ["docker", "container", "rm", "--force", "--volumes", node_name],
                                60,
                            )
                            node_removed = direct is not None and direct.returncode == 0
                            if not node_removed:
                                time.sleep(2)
                                retry = run_cleanup(
                                    "fallback_remove_node_retry",
                                    ["docker", "container", "rm", "--force", "--volumes", node_name],
                                    60,
                                )
                                node_removed = retry is not None and retry.returncode == 0
                    elif node_inspect is not None and node_inspect.returncode == 1:
                        detail = f"{node_inspect.stdout}\n{node_inspect.stderr}".lower()
                        node_removed = "no such" in detail or "not found" in detail
                    if not node_removed:
                        report["cleanup_errors"].append(
                            "kind and the ownership-checked fallback did not delete the disposable cluster"
                        )
            else:
                report["cleanup_errors"].append(
                    "locked kind executable was unavailable during cleanup"
                )

        if ownership.network_acquired:
            inspection = run_cleanup(
                "network_inspect",
                [
                    "docker",
                    "network",
                    "inspect",
                    "--format",
                    "{{json .Labels}}",
                    network,
                ],
                20,
            )
            if inspection is not None and inspection.returncode == 0:
                try:
                    labels = json.loads(inspection.stdout)
                except json.JSONDecodeError:
                    labels = {}
                if (
                    labels.get(_NETWORK_DISPOSABLE_KEY) != "true"
                    or labels.get(_NETWORK_OWNER_KEY) != ownership.owner_token
                ):
                    report["cleanup_errors"].append(
                        f"refused to remove {network}: invocation ownership was not verified"
                    )
                else:
                    removed = run_cleanup(
                        "remove_network", ["docker", "network", "rm", network], 30
                    )
                    if removed is None or removed.returncode != 0:
                        retry = run_cleanup(
                            "remove_network_retry",
                            ["docker", "network", "rm", network],
                            30,
                        )
                        if retry is None or retry.returncode != 0:
                            report["cleanup_errors"].append(
                                "Docker did not remove the acquired disposable network"
                            )
            elif inspection is None:
                report["cleanup_errors"].append(
                    "Docker network ownership could not be inspected"
                )
            elif inspection.returncode == 1 and (
                "no such" in f"{inspection.stdout}\n{inspection.stderr}".lower()
                or "not found" in f"{inspection.stdout}\n{inspection.stderr}".lower()
            ):
                report["network_already_absent"] = True
            else:
                report["cleanup_errors"].append(
                    "Docker network absence or ownership could not be established"
                )

        _write_json(attempt_dir / "cleanup.json", report)
        return report

    @contextmanager
    def attempt(
        self, run_id: str, attempt_number: int, attempt_dir: Path
    ) -> Iterator[AttemptEnvironment]:
        if (
            isinstance(attempt_number, bool)
            or attempt_number < 1
            or attempt_number > self.max_attempts
        ):
            raise EnvironmentError(
                f"Attempt number must be between 1 and {self.max_attempts}"
            )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        cluster, network = self._names(run_id, attempt_number)
        ownership = _AttemptOwnership(owner_token=uuid.uuid4().hex)
        primary_error: BaseException | None = None
        try:
            env = self._setup(run_id, attempt_number, attempt_dir, ownership)
            yield env
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                cleanup = self._cleanup(
                    cluster,
                    network,
                    attempt_dir,
                    ownership=ownership,
                    export_logs=ownership.cluster_create_started,
                )
                if cleanup.get("cleanup_errors"):
                    message = "; ".join(cleanup["cleanup_errors"])
                    if primary_error is not None:
                        primary_error.add_note(f"Cleanup also failed: {message}")
                    else:
                        raise EnvironmentError(f"Disposable cleanup failed: {message}")
            except Exception as cleanup_exc:
                if primary_error is not None:
                    primary_error.add_note(
                        f"Cleanup raised {type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                else:
                    raise
