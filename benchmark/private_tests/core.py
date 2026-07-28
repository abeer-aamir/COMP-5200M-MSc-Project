from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


class EvaluationError(RuntimeError):
    pass


class KubectlError(EvaluationError):
    def __init__(self, command: list[str], stdout: str, stderr: str, code: int):
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.code = code
        message = stderr.strip() or stdout.strip() or f"kubectl exited {code}"
        super().__init__(message)


@dataclass(frozen=True)
class Outcome:
    requirement_id: str
    name: str
    passed: bool
    detail: str
    duration_ms: int


class ResultCollector:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.outcomes: list[Outcome] = []

    def check(
        self, requirement_id: str, name: str, operation: Callable[[], str | None]
    ) -> None:
        started = time.monotonic()
        try:
            detail = operation() or "passed"
            passed = True
        except Exception as exc:  # each requirement must still produce a result
            detail = f"{type(exc).__name__}: {exc}"
            passed = False
        self.outcomes.append(
            Outcome(
                requirement_id=requirement_id,
                name=name,
                passed=passed,
                detail=detail,
                duration_ms=round((time.monotonic() - started) * 1000),
            )
        )

    def report(self) -> dict[str, Any]:
        passed = sum(item.passed for item in self.outcomes)
        return {
            "task_id": self.task_id,
            "status": "passed" if passed == len(self.outcomes) else "failed",
            "passed": passed,
            "failed": len(self.outcomes) - passed,
            "total": len(self.outcomes),
            "outcomes": [asdict(item) for item in self.outcomes],
        }


class Kubectl:
    def __init__(
        self,
        context: str,
        namespace: str,
        kubeconfig: str | Path | None = None,
        command_timeout: int = 45,
        executable: str | Path = "kubectl",
        command_prefix: Sequence[str] | None = None,
    ):
        if not context.strip():
            raise EvaluationError("an explicit kubectl context is required")
        if namespace not in {"order-system", "pipeline-ns"}:
            raise EvaluationError(f"refusing unexpected evaluation namespace {namespace!r}")
        self.context = context
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.command_timeout = command_timeout
        self.executable = str(executable)
        self.command_prefix = list(command_prefix) if command_prefix else None

    def _base(self) -> list[str]:
        command = list(self.command_prefix) if self.command_prefix else [self.executable]
        command.extend(["--context", self.context])
        if self.kubeconfig:
            command.extend(["--kubeconfig", str(self.kubeconfig)])
        return command

    def run(
        self,
        args: Iterable[str],
        *,
        check: bool = True,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [*self._base(), *args]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout or self.command_timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise EvaluationError("kubectl was not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise EvaluationError(f"command timed out: {' '.join(command)}") from exc
        if check and result.returncode:
            raise KubectlError(command, result.stdout, result.stderr, result.returncode)
        return result

    def json(self, args: Iterable[str]) -> dict[str, Any]:
        result = self.run([*args, "-o", "json"])
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise EvaluationError("kubectl did not return JSON") from exc

    def get(self, kind: str, name: str, *, namespace: bool = True) -> dict[str, Any]:
        args = ["get", kind, name]
        if namespace:
            args.extend(["-n", self.namespace])
        return self.json(args)

    def list(self, kind: str, label: str | None = None) -> list[dict[str, Any]]:
        args = ["get", kind, "-n", self.namespace]
        if label:
            args.extend(["-l", label])
        return self.json(args).get("items", [])

    def rollout(self, kind: str, name: str, timeout_seconds: int = 180) -> None:
        self.run(
            [
                "rollout",
                "status",
                f"{kind}/{name}",
                "-n",
                self.namespace,
                f"--timeout={timeout_seconds}s",
            ],
            timeout=timeout_seconds + 10,
        )

    def scale(self, kind: str, name: str, replicas: int) -> None:
        self.run(
            [
                "scale",
                f"{kind}/{name}",
                "-n",
                self.namespace,
                f"--replicas={replicas}",
            ]
        )

    def wait_until(
        self,
        predicate: Callable[[], bool],
        description: str,
        timeout_seconds: int = 120,
        interval_seconds: float = 2,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(interval_seconds)
        raise EvaluationError(f"timed out waiting for {description}")

    def exec(self, pod: str, shell_command: str, container: str | None = None) -> str:
        args = ["exec", "-n", self.namespace, pod]
        if container:
            args.extend(["-c", container])
        args.extend(["--", "sh", "-ec", shell_command])
        return self.run(args).stdout

    def auth_can_i(self, service_account: str, verb: str, resource: str) -> bool:
        result = self.run(
            [
                "auth",
                "can-i",
                verb,
                resource,
                "-n",
                self.namespace,
                f"--as=system:serviceaccount:{self.namespace}:{service_account}",
            ],
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "yes"

    def delete(self, kind: str, name: str) -> None:
        if not name.startswith("aipc-eval-"):
            raise EvaluationError(f"refusing to delete non-evaluator object {name!r}")
        self.run(
            ["delete", kind, name, "-n", self.namespace, "--ignore-not-found=true"],
            check=False,
        )

    def run_probe_pod(
        self,
        name: str,
        shell_command: str,
        *,
        labels: dict[str, str] | None = None,
        expect_success: bool = True,
        timeout_seconds: int = 35,
    ) -> str:
        if not name.startswith("aipc-eval-"):
            raise EvaluationError("probe pod names must use the aipc-eval- prefix")
        self.delete("pod", name)
        args = [
            "run",
            name,
            "-n",
            self.namespace,
            "--image=busybox:1.36.1",
            "--restart=Never",
        ]
        if labels:
            rendered = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
            args.append(f"--labels={rendered}")
        args.extend(["--command", "--", "sh", "-ec", shell_command])
        self.run(args)
        try:
            terminal: dict[str, Any] = {}

            def completed() -> bool:
                nonlocal terminal
                terminal = self.get("pod", name)
                return terminal.get("status", {}).get("phase") in {"Succeeded", "Failed"}

            self.wait_until(completed, f"probe pod {name} to finish", timeout_seconds, 1)
            phase = terminal.get("status", {}).get("phase")
            logs = self.run(
                ["logs", name, "-n", self.namespace], check=False
            ).stdout.strip()
            succeeded = phase == "Succeeded"
            if succeeded != expect_success:
                expectation = "succeed" if expect_success else "fail"
                raise EvaluationError(
                    f"probe was expected to {expectation}, phase={phase}, logs={logs!r}"
                )
            return f"probe phase={phase}" + (f", logs={logs}" if logs else "")
        finally:
            self.delete("pod", name)

    def create_job_from_cronjob(
        self,
        cronjob: str,
        suffix: str,
        *,
        expect_success: bool,
        timeout_seconds: int = 70,
    ) -> str:
        name = f"aipc-eval-{suffix}"
        self.delete("job", name)
        self.run(
            [
                "create",
                "job",
                name,
                f"--from=cronjob/{cronjob}",
                "-n",
                self.namespace,
            ]
        )
        try:
            terminal: dict[str, Any] = {}

            def completed() -> bool:
                nonlocal terminal
                terminal = self.get("job", name)
                status = terminal.get("status", {})
                return bool(status.get("succeeded") or status.get("failed"))

            self.wait_until(completed, f"Job {name} to finish", timeout_seconds, 2)
            status = terminal.get("status", {})
            succeeded = bool(status.get("succeeded"))
            if succeeded != expect_success:
                expectation = "succeed" if expect_success else "fail"
                pods = self.list("pods", f"job-name={name}")
                pod_name = pods[0].get("metadata", {}).get("name") if pods else None
                logs = ""
                if pod_name:
                    logs = self.run(
                        ["logs", pod_name, "-n", self.namespace], check=False
                    ).stdout.strip()
                raise EvaluationError(
                    f"Job was expected to {expectation}; status={status}, logs={logs!r}"
                )
            return f"Job {'succeeded' if succeeded else 'failed'} as expected"
        finally:
            self.delete("job", name)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def pod_spec(workload: dict[str, Any]) -> dict[str, Any]:
    return workload["spec"]["template"]["spec"]


def pod_labels(workload: dict[str, Any]) -> dict[str, str]:
    return workload["spec"]["template"]["metadata"].get("labels", {})


def all_containers(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [*spec.get("initContainers", []), *spec.get("containers", [])]


def command_text(container: dict[str, Any]) -> str:
    return " ".join([*container.get("command", []), *container.get("args", [])])


def find_mount(container: dict[str, Any], path: str) -> dict[str, Any] | None:
    return next(
        (item for item in container.get("volumeMounts", []) if item.get("mountPath") == path),
        None,
    )


def find_volume(spec: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((item for item in spec.get("volumes", []) if item.get("name") == name), None)


def assert_hardened(workload: dict[str, Any], label: str) -> None:
    spec = pod_spec(workload)
    pod_security = spec.get("securityContext", {})
    for container in all_containers(spec):
        security = container.get("securityContext", {})
        run_as_non_root = security.get(
            "runAsNonRoot", pod_security.get("runAsNonRoot")
        )
        run_as_user = security.get("runAsUser", pod_security.get("runAsUser"))
        require(run_as_non_root is True, f"{label}: every container must runAsNonRoot")
        require(run_as_user == 1000, f"{label}: every container must runAsUser 1000")
        require(
            security.get("readOnlyRootFilesystem") is True,
            f"{label}: every container needs readOnlyRootFilesystem",
        )
        dropped = set(security.get("capabilities", {}).get("drop", []))
        require("ALL" in dropped, f"{label}: every container must drop ALL capabilities")


def role_bound_to_service_account(
    kube: Kubectl, service_account: str
) -> dict[str, Any]:
    for binding in kube.list("rolebindings"):
        subjects = binding.get("subjects", [])
        if any(
            item.get("kind") == "ServiceAccount"
            and item.get("name") == service_account
            and item.get("namespace", kube.namespace) == kube.namespace
            for item in subjects
        ):
            role_ref = binding.get("roleRef", {})
            require(role_ref.get("kind") == "Role", "RoleBinding must reference a Role")
            return kube.get("role", role_ref.get("name", ""))
    raise EvaluationError(f"no RoleBinding found for {service_account}")


def assert_exact_role(
    role: dict[str, Any], verbs: set[str], resources: set[str]
) -> None:
    found: set[tuple[str, str]] = set()
    for rule in role.get("rules", []):
        require(set(rule.get("apiGroups", [])) == {""}, "Role has a non-core API group")
        require(not rule.get("resourceNames"), "Role must not use additional resource names")
        for resource in rule.get("resources", []):
            for verb in rule.get("verbs", []):
                found.add((resource, verb))
    expected = {(resource, verb) for resource in resources for verb in verbs}
    require(found == expected, f"Role permissions differ: expected={expected}, actual={found}")


def volume_claim_template(
    statefulset: dict[str, Any], name: str
) -> dict[str, Any]:
    for claim in statefulset.get("spec", {}).get("volumeClaimTemplates", []):
        if claim.get("metadata", {}).get("name") == name:
            return claim
    raise EvaluationError(f"volumeClaimTemplate {name!r} not found")


def assert_claim(claim: dict[str, Any], size: str = "1Gi") -> None:
    spec = claim.get("spec", {})
    require(set(spec.get("accessModes", [])) == {"ReadWriteOnce"}, "claim must be RWO")
    actual = spec.get("resources", {}).get("requests", {}).get("storage")
    require(actual == size, f"claim storage must be {size}, got {actual!r}")


def int_or_string(value: Any) -> str:
    return str(value).strip()
