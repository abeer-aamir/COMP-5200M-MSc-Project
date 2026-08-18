from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import quote

from aipycraft_k8s.commands import CommandResult, CommandRunner
from aipycraft_k8s.tasks import CANONICAL_TASK_NAMESPACES


class EvaluationError(RuntimeError):
    pass


class RequirementFailure(EvaluationError):
    """A deterministic mismatch attributable to the deployed candidate."""


class EvaluationInfrastructureError(EvaluationError):
    """The evaluator could not reliably determine candidate correctness."""


class SuiteExecutionError(EvaluationError):
    def __init__(
        self,
        task_id: str,
        requirement_id: str,
        name: str,
        cause: BaseException,
        outcomes: list["Outcome"],
    ):
        self.task_id = task_id
        self.requirement_id = requirement_id
        self.name = name
        self.cause = cause
        self.outcomes = list(outcomes)
        self.status = (
            "infrastructure_error"
            if isinstance(cause, EvaluationInfrastructureError)
            else "evaluator_error"
        )
        super().__init__(
            f"{requirement_id} ({name}) could not be evaluated: "
            f"{type(cause).__name__}: {cause}"
        )

    def partial_report(self) -> dict[str, Any]:
        passed = sum(item.passed for item in self.outcomes)
        return {
            "task_id": self.task_id,
            "status": self.status,
            "passed": passed,
            "failed": len(self.outcomes) - passed,
            "total": len(self.outcomes),
            "outcomes": [asdict(item) for item in self.outcomes],
            "interrupted_requirement_id": self.requirement_id,
            "interrupted_requirement_name": self.name,
            "error": str(self),
        }


class KubectlError(EvaluationInfrastructureError):
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
        except RequirementFailure as exc:
            detail = f"{type(exc).__name__}: {exc}"
            passed = False
        except Exception as exc:
            raise SuiteExecutionError(
                self.task_id,
                requirement_id,
                name,
                exc,
                self.outcomes,
            ) from exc
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
        if namespace not in set(CANONICAL_TASK_NAMESPACES.values()):
            raise EvaluationError(f"refusing unexpected evaluation namespace {namespace!r}")
        self.context = context
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.command_timeout = command_timeout
        self.executable = str(executable)
        self.command_prefix = list(command_prefix) if command_prefix else None
        self.runner = CommandRunner()
        self._api_resource_scope_cache: dict[str, dict[str, bool]] = {}

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
    ) -> CommandResult:
        command = [*self._base(), *args]
        result = self.runner.run(
            command,
            timeout=timeout or self.command_timeout,
            check=False,
        )
        missing_executable = (
            result.returncode == 127
            and (
                "command was not found:" in result.stderr.lower()
                or "executable file not found" in result.stderr.lower()
            )
        )
        if missing_executable:
            raise EvaluationInfrastructureError("kubectl was not found on PATH")
        evaluator_timeout = (
            result.returncode == 124
            and "timed out after" in result.stderr.lower()
            and "spawned process tree terminated" in result.stderr.lower()
        )
        if evaluator_timeout:
            raise EvaluationInfrastructureError(
                f"command timed out: {' '.join(command)}"
            )
        if check and result.returncode:
            raise KubectlError(command, result.stdout, result.stderr, result.returncode)
        return result

    def json(self, args: Iterable[str]) -> dict[str, Any]:
        result = self.run([*args, "-o", "json"])
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise EvaluationInfrastructureError("kubectl did not return JSON") from exc

    def resource_is_namespaced(self, api_version: str, kind: str) -> bool:
        """Resolve a Kind's authoritative scope from live API discovery."""

        if (
            not isinstance(api_version, str)
            or not api_version
            or api_version != api_version.strip()
        ):
            raise EvaluationInfrastructureError(
                "candidate resource has a malformed apiVersion"
            )
        if not isinstance(kind, str) or not kind or kind != kind.strip():
            raise EvaluationInfrastructureError("candidate resource has a malformed kind")

        scopes = self._api_resource_scope_cache.get(api_version)
        if scopes is None:
            parts = api_version.split("/")
            if len(parts) == 1 and parts[0]:
                discovery_path = f"/api/{quote(parts[0], safe='')}"
            elif len(parts) == 2 and all(parts):
                discovery_path = (
                    f"/apis/{quote(parts[0], safe='')}/{quote(parts[1], safe='')}"
                )
            else:
                raise EvaluationInfrastructureError(
                    f"candidate apiVersion {api_version!r} cannot be discovered"
                )

            result = self.run(["get", f"--raw={discovery_path}"])
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise EvaluationInfrastructureError(
                    f"API discovery for {api_version!r} returned malformed JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise EvaluationInfrastructureError(
                    f"API discovery for {api_version!r} returned a non-object"
                )
            if payload.get("groupVersion") != api_version:
                raise EvaluationInfrastructureError(
                    f"API discovery groupVersion did not match {api_version!r}"
                )
            resources = payload.get("resources")
            if not isinstance(resources, list):
                raise EvaluationInfrastructureError(
                    f"API discovery for {api_version!r} has no resources list"
                )

            discovered: dict[str, bool] = {}
            for resource in resources:
                if not isinstance(resource, dict):
                    raise EvaluationInfrastructureError(
                        f"API discovery for {api_version!r} contains a malformed resource"
                    )
                name = resource.get("name")
                discovered_kind = resource.get("kind")
                namespaced = resource.get("namespaced")
                if (
                    not isinstance(name, str)
                    or not isinstance(discovered_kind, str)
                    or not isinstance(namespaced, bool)
                ):
                    raise EvaluationInfrastructureError(
                        f"API discovery for {api_version!r} contains malformed scope data"
                    )
                if "/" in name:
                    continue
                previous = discovered.get(discovered_kind)
                if previous is not None and previous != namespaced:
                    raise EvaluationInfrastructureError(
                        f"API discovery for {api_version!r} is ambiguous for "
                        f"kind {discovered_kind!r}"
                    )
                discovered[discovered_kind] = namespaced
            self._api_resource_scope_cache[api_version] = discovered
            scopes = discovered

        if kind not in scopes:
            raise EvaluationInfrastructureError(
                f"live API discovery for {api_version!r} did not contain "
                f"candidate kind {kind!r}"
            )
        return scopes[kind]

    def get(self, kind: str, name: str, *, namespace: bool = True) -> dict[str, Any]:
        args = ["get", kind, name]
        if namespace:
            args.extend(["-n", self.namespace])
        try:
            return self.json(args)
        except KubectlError as exc:
            detail = f"{exc.stdout}\n{exc.stderr}".lower()
            if "notfound" in detail or "not found" in detail:
                raise RequirementFailure(
                    f"required {kind}/{name} was not found"
                ) from exc
            raise

    def list(
        self,
        kind: str,
        label: str | None = None,
        *,
        namespace: bool | str = True,
    ) -> list[dict[str, Any]]:
        args = ["get", kind]
        if namespace is True:
            args.extend(["-n", self.namespace])
        elif isinstance(namespace, str):
            args.extend(["-n", namespace])
        if label:
            args.extend(["-l", label])
        return self.json(args).get("items", [])

    def rollout(self, kind: str, name: str, timeout_seconds: int = 180) -> None:
        try:
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
        except KubectlError as exc:
            detail = f"{exc.stdout}\n{exc.stderr}".lower()
            candidate_markers = (
                "timed out waiting for the condition",
                "exceeded its progress deadline",
                "not found",
                "notfound",
            )
            if any(marker in detail for marker in candidate_markers):
                raise RequirementFailure(
                    f"{kind}/{name} did not complete rollout: {exc}"
                ) from exc
            raise

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
        raise RequirementFailure(f"timed out waiting for {description}")

    def exec(self, pod: str, shell_command: str, container: str | None = None) -> str:
        args = ["exec", "-n", self.namespace, pod]
        if container:
            args.extend(["-c", container])
        args.extend(["--", "sh", "-ec", shell_command])
        try:
            return self.run(args).stdout
        except KubectlError as exc:
            detail = f"{exc.stdout}\n{exc.stderr}".lower()
            if "command terminated with exit code" in detail:
                raise RequirementFailure(
                    f"command failed in pod {pod}: {exc}"
                ) from exc
            raise

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
        answer = result.stdout.strip().lower()
        if answer == "yes" and result.returncode == 0:
            return True
        if answer == "no" and result.returncode in {0, 1}:
            return False
        raise EvaluationInfrastructureError(
            "kubectl auth can-i did not return a reliable yes/no result: "
            f"returncode={result.returncode}, stdout={result.stdout!r}, "
            f"stderr={result.stderr!r}"
        )

    def delete(self, kind: str, name: str) -> None:
        if not name.startswith("aipc-eval-"):
            raise EvaluationError(f"refusing to delete non-evaluator object {name!r}")
        self.run(
            ["delete", kind, name, "-n", self.namespace, "--ignore-not-found=true"],
            check=False,
        )

    def _raise_evaluator_creation_failure(
        self, exc: KubectlError, object_description: str
    ) -> None:
        """Attribute admission rejection to the candidate only with a healthy API."""

        detail = f"{exc.stdout}\n{exc.stderr}".lower()
        candidate_rejection_markers = (
            "admission",
            "denied",
            "exceeded quota",
            "failed quota",
            "forbidden",
            "not found",
            "notfound",
            "podsecurity",
            "webhook",
        )
        if not any(marker in detail for marker in candidate_rejection_markers):
            raise exc

        ready = self.run(
            ["get", "--raw=/readyz"],
            check=False,
            timeout=min(self.command_timeout, 10),
        )
        if ready.returncode == 0 and "ok" in ready.stdout.lower():
            raise RequirementFailure(
                f"the evaluator could not create {object_description} because the "
                f"deployed candidate rejected it while the Kubernetes API remained "
                f"healthy: {exc}"
            ) from exc
        raise exc

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
        try:
            self.run(args)
        except KubectlError as exc:
            self._raise_evaluator_creation_failure(exc, f"probe Pod {name}")
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
                raise RequirementFailure(
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
        try:
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
        except KubectlError as exc:
            self._raise_evaluator_creation_failure(exc, f"probe Job {name}")
        try:
            terminal: dict[str, Any] = {}
            terminal_condition: str | None = None

            def completed() -> bool:
                nonlocal terminal, terminal_condition
                terminal = self.get("job", name)
                terminal_condition = job_terminal_condition(terminal)
                return terminal_condition is not None

            self.wait_until(completed, f"Job {name} to finish", timeout_seconds, 2)
            status = terminal.get("status", {})
            succeeded = terminal_condition == "Complete"
            if succeeded != expect_success:
                expectation = "succeed" if expect_success else "fail"
                pods = self.list("pods", f"job-name={name}")
                pod_name = pods[0].get("metadata", {}).get("name") if pods else None
                logs = ""
                if pod_name:
                    logs = self.run(
                        ["logs", pod_name, "-n", self.namespace], check=False
                    ).stdout.strip()
                raise RequirementFailure(
                    f"Job was expected to {expectation}; status={status}, logs={logs!r}"
                )
            return f"Job {'succeeded' if succeeded else 'failed'} as expected"
        finally:
            self.delete("job", name)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RequirementFailure(message)


def job_terminal_condition(job: dict[str, Any]) -> str | None:
    """Return only a terminal Job condition, never a retry counter/target."""

    true_conditions = {
        item.get("type")
        for item in job.get("status", {}).get("conditions", []) or []
        if isinstance(item, dict) and item.get("status") == "True"
    }
    if "Failed" in true_conditions:
        return "Failed"
    if "Complete" in true_conditions:
        return "Complete"
    return None


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
        require(
            run_as_non_root is not False,
            f"{label}: a container explicitly permits root execution",
        )
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
    matching_bindings: list[dict[str, Any]] = []
    for binding in kube.list("rolebindings"):
        subjects = binding.get("subjects", [])
        if any(
            item.get("kind") == "ServiceAccount"
            and item.get("name") == service_account
            and item.get("namespace", kube.namespace) == kube.namespace
            for item in subjects
        ):
            matching_bindings.append(binding)

    cluster_bindings = []
    for binding in kube.list("clusterrolebindings", namespace=False):
        subjects = binding.get("subjects", [])
        if any(
            item.get("kind") == "ServiceAccount"
            and item.get("name") == service_account
            and item.get("namespace") == kube.namespace
            for item in subjects
        ):
            cluster_bindings.append(binding)

    require(
        not cluster_bindings,
        f"{service_account} has an additional ClusterRoleBinding",
    )
    require(
        len(matching_bindings) == 1,
        f"expected exactly one RoleBinding for {service_account}, "
        f"found {len(matching_bindings)}",
    )
    role_ref = matching_bindings[0].get("roleRef", {})
    require(role_ref.get("kind") == "Role", "RoleBinding must reference a Role")
    return kube.get("role", role_ref.get("name", ""))


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
    raise RequirementFailure(f"volumeClaimTemplate {name!r} not found")


def assert_claim(claim: dict[str, Any], size: str = "1Gi") -> None:
    spec = claim.get("spec", {})
    require(set(spec.get("accessModes", [])) == {"ReadWriteOnce"}, "claim must be RWO")
    actual = spec.get("resources", {}).get("requests", {}).get("storage")
    require(actual == size, f"claim storage must be {size}, got {actual!r}")


def int_or_string(value: Any) -> str:
    return str(value).strip()
