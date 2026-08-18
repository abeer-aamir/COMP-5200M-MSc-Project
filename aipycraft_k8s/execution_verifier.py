from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any
from urllib.parse import urlparse

from .environment import AttemptEnvironment, EnvironmentError
from .tasks import BenchmarkTask


_WORKLOADS = ("deployment", "statefulset", "daemonset")


def _resource_items(
    env: AttemptEnvironment,
    resource: str,
    namespace: str,
    *,
    timeout: int | None = None,
) -> list[dict[str, Any]]:
    value = env.kubectl_json(
        ["get", resource, "-n", namespace], timeout=timeout
    )
    items = value.get("items", [])
    if not isinstance(items, list):
        raise EnvironmentError(f"kubectl returned malformed {resource} items")
    return [item for item in items if isinstance(item, dict)]


def _name(item: dict[str, Any]) -> str:
    return str(item.get("metadata", {}).get("name", "<unnamed>"))


def _command_detail(result: Any) -> dict[str, Any]:
    audit = getattr(result, "audit_dict", None)
    if callable(audit):
        return dict(audit())
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_ms": result.duration_ms,
        "started_at": getattr(result, "started_at", None),
        "finished_at": getattr(result, "finished_at", None),
        "timeout_seconds": getattr(result, "timeout_seconds", None),
        "deadline_overrun_ms": getattr(result, "deadline_overrun_ms", 0),
        "deadline_overrun": getattr(result, "deadline_overrun", False),
        "host_pause_suspected": getattr(result, "host_pause_suspected", False),
    }


def _workload_selector(item: dict[str, Any]) -> str | None:
    selector = item.get("spec", {}).get("selector", {})
    if not isinstance(selector, dict):
        return None
    requirements: list[str] = []
    match_labels = selector.get("matchLabels", {})
    if isinstance(match_labels, dict):
        requirements.extend(
            f"{key}={value}" for key, value in sorted(match_labels.items())
        )
    match_expressions = selector.get("matchExpressions", [])
    if isinstance(match_expressions, list):
        for expression in match_expressions:
            if not isinstance(expression, dict):
                continue
            key = expression.get("key")
            operator = expression.get("operator")
            values = expression.get("values", [])
            if not isinstance(key, str) or not key:
                continue
            if operator in {"In", "NotIn"} and isinstance(values, list) and values:
                rendered_values = ",".join(str(value) for value in values)
                keyword = "in" if operator == "In" else "notin"
                requirements.append(f"{key} {keyword} ({rendered_values})")
            elif operator == "Exists":
                requirements.append(key)
            elif operator == "DoesNotExist":
                requirements.append(f"!{key}")
    return ",".join(requirements) if requirements else None


def _probe_name(cronjob_name: str) -> str:
    safe = re.sub(r"[^a-z0-9-]", "-", cronjob_name.lower()).strip("-")
    safe = safe[:34].rstrip("-") or "cronjob"
    return f"aipc-exec-{safe}-{uuid.uuid4().hex[:8]}"


def _ready_pod(pods: list[dict[str, Any]]) -> dict[str, Any] | None:
    for pod in pods:
        if any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in pod.get("status", {}).get("conditions", []) or []
        ):
            return pod
    return None


def _target_container_and_port(
    pod: dict[str, Any], target_port: Any
) -> tuple[str | None, int | None]:
    containers = pod.get("spec", {}).get("containers", []) or []
    if isinstance(target_port, bool):
        return None, None
    if isinstance(target_port, int):
        container_name = next(
            (
                str(container.get("name"))
                for container in containers
                if any(
                    port.get("containerPort") == target_port
                    for port in container.get("ports", []) or []
                )
            ),
            str(containers[0].get("name")) if containers else None,
        )
        return container_name, target_port
    if isinstance(target_port, str):
        for container in containers:
            for port in container.get("ports", []) or []:
                if port.get("name") == target_port:
                    value = port.get("containerPort")
                    if isinstance(value, int) and not isinstance(value, bool):
                        return str(container.get("name")), value
    return None, None


def _condition_true(value: dict[str, Any], *types: str) -> dict[str, Any] | None:
    wanted = set(types)
    for condition in value.get("status", {}).get("conditions", []) or []:
        if (
            isinstance(condition, dict)
            and condition.get("type") in wanted
            and condition.get("status") == "True"
        ):
            return condition
    return None


def _workload_state(resource: str, item: dict[str, Any]) -> dict[str, Any]:
    """Return a stable, task-agnostic rollout state from Kubernetes status."""

    spec = item.get("spec", {}) or {}
    status = item.get("status", {}) or {}
    desired = int(spec.get("replicas", 1) or 0)
    generation = int(item.get("metadata", {}).get("generation", 0) or 0)
    observed = int(status.get("observedGeneration", 0) or 0)
    conditions = status.get("conditions", []) or []

    if resource == "deployment":
        for condition in conditions:
            if (
                isinstance(condition, dict)
                and condition.get("type") == "Progressing"
                and condition.get("status") == "False"
                and condition.get("reason") == "ProgressDeadlineExceeded"
            ):
                return {
                    "state": "failed",
                    "reason": "ProgressDeadlineExceeded",
                    "status": status,
                }
        ready = (
            observed >= generation
            and int(status.get("updatedReplicas", 0) or 0) >= desired
            and int(status.get("availableReplicas", 0) or 0) >= desired
            and int(status.get("unavailableReplicas", 0) or 0) == 0
        )
    elif resource == "statefulset":
        ready = (
            observed >= generation
            and int(status.get("readyReplicas", 0) or 0) >= desired
            and int(status.get("currentReplicas", desired) or 0) >= desired
        )
    elif resource == "daemonset":
        desired = int(status.get("desiredNumberScheduled", 0) or 0)
        ready = (
            observed >= generation
            and int(status.get("numberReady", 0) or 0) >= desired
            and int(status.get("numberUnavailable", 0) or 0) == 0
        )
    else:
        raise ValueError(f"Unsupported workload resource {resource}")
    return {
        "state": "ready" if ready else "pending",
        "reason": None if ready else "waiting_for_operational_replicas",
        "desired": desired,
        "generation": generation,
        "observed_generation": observed,
        "status": status,
    }


def _job_state(item: dict[str, Any]) -> dict[str, Any]:
    status = item.get("status", {}) or {}
    complete = _condition_true(item, "Complete")
    failed = _condition_true(item, "Failed", "FailureTarget")
    if complete is not None:
        state = "succeeded"
        condition = complete
    elif failed is not None:
        state = "failed"
        condition = failed
    else:
        state = "pending"
        condition = None
    return {
        "state": state,
        "condition": condition,
        "status": status,
    }


def _command_suppresses_output(spec: dict[str, Any]) -> bool:
    text = " ".join(
        str(value)
        for field in ("command", "args")
        for value in (spec.get(field, []) or [])
    )
    return any(token in text for token in (">/dev/null", "2>/dev/null", "2>&1"))


def _pod_events(
    env: AttemptEnvironment,
    namespace: str,
    pod_name: str,
    *,
    command_timeout: int,
) -> dict[str, Any]:
    result = env.kubectl(
        [
            "get",
            "events",
            "-n",
            namespace,
            "--field-selector",
            f"involvedObject.name={pod_name}",
            "-o",
            "json",
        ],
        check=False,
        timeout=command_timeout,
    )
    detail = _command_detail(result)
    if result.returncode != 0:
        return {"status": "capture_error", "command": detail, "items": []}
    try:
        value = json.loads(result.stdout)
        raw_items = value.get("items", []) if isinstance(value, dict) else []
    except Exception as exc:
        return {
            "status": "capture_error",
            "command": detail,
            "error": f"{type(exc).__name__}: {exc}",
            "items": [],
        }
    items = []
    for event in raw_items[-30:]:
        if not isinstance(event, dict):
            continue
        items.append(
            {
                "type": event.get("type"),
                "reason": event.get("reason"),
                "message": event.get("message"),
                "count": event.get("count"),
                "first_timestamp": event.get("firstTimestamp"),
                "last_timestamp": event.get("lastTimestamp"),
            }
        )
    return {"status": "captured", "items": items}


_DNS_TARGET = re.compile(
    r"\b(?:nslookup|getent\s+hosts)\s+([a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?)\b",
    re.IGNORECASE,
)
_HTTP_TARGET = re.compile(r"https?://[^\s'\";|&<>]+", re.IGNORECASE)


def _dependency_targets(container_spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    text = " ".join(
        str(value)
        for field in ("command", "args")
        for value in (container_spec.get(field, []) or [])
    )
    hosts = list(dict.fromkeys(match.group(1) for match in _DNS_TARGET.finditer(text)))
    urls: list[str] = []
    for raw in _HTTP_TARGET.findall(text):
        candidate = raw.rstrip(").,]")
        parsed = urlparse(candidate)
        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and re.fullmatch(r"[a-z0-9.-]+", parsed.hostname, re.IGNORECASE)
        ):
            urls.append(candidate)
    return hosts[:4], list(dict.fromkeys(urls))[:4]


def _active_dependency_diagnostics(
    env: AttemptEnvironment,
    namespace: str,
    pod_name: str,
    container_name: str,
    container_spec: dict[str, Any],
    *,
    command_timeout: int,
) -> dict[str, Any] | None:
    """Split opaque init loops into safe, bounded DNS and HTTP observations."""

    hosts, urls = _dependency_targets(container_spec)
    if not hosts and not urls:
        return None
    timeout = min(command_timeout, 15)

    def execute(command: list[str]) -> dict[str, Any]:
        result = env.kubectl(
            [
                "exec",
                pod_name,
                "-n",
                namespace,
                "-c",
                container_name,
                "--",
                *command,
            ],
            check=False,
            timeout=timeout,
        )
        return _command_detail(result)

    return {
        "schema_version": 1,
        "safety": (
            "Read-only probes derived from strictly validated targets in the "
            "candidate's currently running init-container command."
        ),
        "resolv_conf": execute(["cat", "/etc/resolv.conf"]),
        "dns": [
            {"target": host, "command": execute(["nslookup", host])}
            for host in hosts
        ],
        "http": [
            {
                "target": url,
                "command": execute(["wget", "-S", "-O", "/dev/null", url]),
            }
            for url in urls
        ],
    }


def _pod_execution_diagnostics(
    env: AttemptEnvironment,
    namespace: str,
    pods: list[dict[str, Any]],
    *,
    command_timeout: int,
    pod_limit: int = 8,
    container_limit: int = 12,
) -> dict[str, Any]:
    """Capture bounded Pod state, specs, logs, and events before cleanup."""

    result: dict[str, Any] = {
        "status": "captured",
        "matched_pods": len(pods),
        "pod_limit": pod_limit,
        "truncated_pods": max(len(pods) - pod_limit, 0),
        "pods": [],
    }
    for pod in pods[:pod_limit]:
        pod_name = _name(pod)
        status = pod.get("status", {}) or {}
        spec = pod.get("spec", {}) or {}
        status_by_name: dict[str, tuple[str, dict[str, Any]]] = {}
        for field, container_type in (
            ("initContainerStatuses", "init"),
            ("containerStatuses", "main"),
        ):
            for item in status.get(field, []) or []:
                if isinstance(item, dict) and item.get("name"):
                    status_by_name[str(item["name"])] = (container_type, item)
        specs: list[tuple[str, dict[str, Any]]] = []
        for field, container_type in (
            ("initContainers", "init"),
            ("containers", "main"),
        ):
            for item in spec.get(field, []) or []:
                if isinstance(item, dict) and item.get("name"):
                    specs.append((container_type, item))
        seen_names = {str(item.get("name")) for _, item in specs}
        for name, (container_type, _) in status_by_name.items():
            if name not in seen_names:
                specs.append((container_type, {"name": name}))

        pod_record: dict[str, Any] = {
            "pod": pod_name,
            "phase": status.get("phase"),
            "reason": status.get("reason"),
            "message": status.get("message"),
            "pod_ip": status.get("podIP"),
            "host_ip": status.get("hostIP"),
            "conditions": status.get("conditions", []),
            "containers": [],
            "truncated_containers": max(len(specs) - container_limit, 0),
        }
        for container_type, container_spec in specs[:container_limit]:
            name = str(container_spec.get("name", "<unnamed>"))
            _, container_status = status_by_name.get(name, (container_type, {}))
            current = env.kubectl(
                [
                    "logs",
                    pod_name,
                    "-n",
                    namespace,
                    "-c",
                    name,
                    "--tail=200",
                ],
                check=False,
                timeout=command_timeout,
            )
            record: dict[str, Any] = {
                "container": name,
                "container_type": container_type,
                "image": container_spec.get("image") or container_status.get("image"),
                "command": container_spec.get("command"),
                "args": container_spec.get("args"),
                "output_suppressed_by_candidate_command": (
                    _command_suppresses_output(container_spec)
                ),
                "ready": container_status.get("ready"),
                "restart_count": container_status.get("restartCount"),
                "state": container_status.get("state"),
                "last_state": container_status.get("lastState"),
                "logs": _command_detail(current),
            }
            if container_type == "init" and isinstance(
                container_status.get("state", {}).get("running"), dict
            ):
                active = _active_dependency_diagnostics(
                    env,
                    namespace,
                    pod_name,
                    name,
                    container_spec,
                    command_timeout=command_timeout,
                )
                if active is not None:
                    record["active_dependency_diagnostics"] = active
            restart_count = container_status.get("restartCount")
            if isinstance(restart_count, int) and restart_count > 0:
                previous = env.kubectl(
                    [
                        "logs",
                        pod_name,
                        "-n",
                        namespace,
                        "-c",
                        name,
                        "--previous",
                        "--tail=200",
                    ],
                    check=False,
                    timeout=command_timeout,
                )
                record["previous_logs"] = _command_detail(previous)
            pod_record["containers"].append(record)
        pod_record["events"] = _pod_events(
            env,
            namespace,
            pod_name,
            command_timeout=command_timeout,
        )
        result["pods"].append(pod_record)
    return result


def _pods_for_selector(
    env: AttemptEnvironment,
    namespace: str,
    selector: str | None,
    *,
    command_timeout: int,
) -> list[dict[str, Any]]:
    if not selector:
        return []
    value = env.kubectl_json(
        ["get", "pods", "-n", namespace, "-l", selector],
        timeout=command_timeout,
    )
    raw = value.get("items", [])
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _namespace_failure_context(
    env: AttemptEnvironment,
    namespace: str,
    *,
    command_timeout: int,
) -> dict[str, Any]:
    """Capture bounded live dependency context without task-specific knowledge."""

    context: dict[str, Any] = {"services": [], "network_policies": []}
    try:
        for service in _resource_items(
            env, "services", namespace, timeout=command_timeout
        )[:20]:
            name = _name(service)
            service_record: dict[str, Any] = {
                "name": name,
                "selector": service.get("spec", {}).get("selector"),
                "cluster_ip": service.get("spec", {}).get("clusterIP"),
                "ports": service.get("spec", {}).get("ports", []),
            }
            try:
                endpoints = env.kubectl_json(
                    ["get", "endpoints", name, "-n", namespace],
                    timeout=command_timeout,
                )
                service_record["endpoint_subsets"] = endpoints.get("subsets", [])
            except Exception as exc:
                service_record["endpoint_capture_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            context["services"].append(service_record)
        context["network_policies"] = [
            {
                "name": _name(policy),
                "spec": policy.get("spec", {}),
            }
            for policy in _resource_items(
                env, "networkpolicies", namespace, timeout=command_timeout
            )[:20]
        ]
    except Exception as exc:
        context["capture_error"] = f"{type(exc).__name__}: {exc}"
    return context




def _api_command_is_not_found(detail: str) -> bool:
    return bool(
        re.search(
            r"error from server\s*\(notfound\):[^\n]*namespaces?[^\n]*not found"
            r"|^namespaces?[^\n]*not found$",
            detail,
            re.IGNORECASE | re.MULTILINE,
        )
    )


_CANDIDATE_API_REJECTION = re.compile(
    r"error from server\s*\((?:notfound|forbidden|invalid|badrequest)\)"
    r"|admission webhook.+denied the request"
    r"|(?:exceeded|failed) quota",
    re.IGNORECASE | re.DOTALL,
)
_REMOTE_COMMAND_EXIT = re.compile(
    r"command terminated with exit code\s+\d+"
    r"|command exited with (?:a )?non-zero exit code",
    re.IGNORECASE,
)
_DETERMINISTIC_EXEC_STATE_FAILURE = re.compile(
    r"container not found"
    r"|cannot exec into a container in a completed pod"
    r"|pod does not have a host assigned",
    re.IGNORECASE,
)


def _failed_command_attribution(
    env: AttemptEnvironment,
    namespace: str,
    result: Any,
    *,
    command_timeout: int,
    allow_remote_exit: bool,
) -> dict[str, Any]:
    """Classify a failed evaluator command without blaming transport on YAML.

    A nonzero kubectl/docker result is not by itself evidence of a candidate
    defect.  Repair is permitted only after the API and required namespace are
    still healthy and the result carries a deterministic Kubernetes rejection,
    in-container exit status, or workload-state error.
    """

    record: dict[str, Any] = {
        "status": "infrastructure",
        "basis": "unclassified_nonzero_evaluator_command",
        "failed_command": _command_detail(result),
        "api_readiness": None,
        "candidate_namespace": None,
    }
    try:
        ready = env.readyz()
        record["api_readiness"] = _command_detail(ready)
    except Exception as exc:
        record["basis"] = "api_readiness_confirmation_raised"
        record["confirmation_error"] = f"{type(exc).__name__}: {exc}"
        return record
    if ready.returncode != 0 or "ok" not in ready.stdout.lower():
        record["basis"] = "api_not_ready_after_failed_command"
        return record

    try:
        namespace_result = env.kubectl(
            ["get", "namespace", namespace, "-o", "json"],
            check=False,
            timeout=command_timeout,
        )
        record["candidate_namespace"] = _command_detail(namespace_result)
    except Exception as exc:
        record["basis"] = "namespace_confirmation_raised"
        record["confirmation_error"] = f"{type(exc).__name__}: {exc}"
        return record

    namespace_detail = (
        f"{namespace_result.stdout}\n{namespace_result.stderr}"
    ).lower()
    if namespace_result.returncode != 0:
        if _api_command_is_not_found(namespace_detail):
            record["status"] = "candidate"
            record["basis"] = "required_namespace_missing_with_healthy_api"
        else:
            record["basis"] = "namespace_inspection_failed_with_healthy_api"
        return record
    try:
        namespace_phase = (
            json.loads(namespace_result.stdout).get("status", {}).get("phase")
        )
    except Exception as exc:
        record["basis"] = "namespace_confirmation_malformed"
        record["confirmation_error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["namespace_phase"] = namespace_phase
    if namespace_phase != "Active":
        record["status"] = "candidate"
        record["basis"] = "required_namespace_non_active_with_healthy_api"
        return record

    detail = f"{result.stdout}\n{result.stderr}"
    if _CANDIDATE_API_REJECTION.search(detail):
        record["status"] = "candidate"
        record["basis"] = "deterministic_kubernetes_rejection_with_healthy_api"
    elif allow_remote_exit and _REMOTE_COMMAND_EXIT.search(detail):
        record["status"] = "candidate"
        record["basis"] = "deterministic_remote_command_exit_with_healthy_api"
    elif allow_remote_exit and _DETERMINISTIC_EXEC_STATE_FAILURE.search(detail):
        record["status"] = "candidate"
        record["basis"] = "deterministic_exec_state_failure_with_healthy_api"
    return record


def _repair_feedback(report: dict[str, Any]) -> dict[str, Any]:
    def compact_finding(finding: dict[str, Any]) -> dict[str, Any]:
        compact = {
            key: finding.get(key)
            for key in ("type", "resource", "reason", "state", "message")
            if finding.get(key) is not None
        }
        diagnostics = finding.get("pod_diagnostics", {}) or {}
        representative_pods = []
        for pod in (diagnostics.get("pods", []) or [])[:2]:
            containers = []
            for container in (pod.get("containers", []) or [])[:4]:
                containers.append(
                    {
                        key: container.get(key)
                        for key in (
                            "container",
                            "container_type",
                            "command",
                            "args",
                            "state",
                            "last_state",
                            "output_suppressed_by_candidate_command",
                            "logs",
                            "previous_logs",
                            "active_dependency_diagnostics",
                        )
                        if container.get(key) is not None
                    }
                )
            representative_pods.append(
                {
                    "pod": pod.get("pod"),
                    "phase": pod.get("phase"),
                    "reason": pod.get("reason"),
                    "conditions": pod.get("conditions"),
                    "containers": containers,
                    "events": pod.get("events"),
                }
            )
        if representative_pods:
            compact["representative_pod_diagnostics"] = representative_pods
        return compact

    suppressed = []
    for collection_name in ("failures", "correlated_observations"):
        for finding in report.get(collection_name, []) or []:
            diagnostics = finding.get("pod_diagnostics", {}) or {}
            for pod in diagnostics.get("pods", []) or []:
                for container in pod.get("containers", []) or []:
                    if container.get("output_suppressed_by_candidate_command"):
                        suppressed.append(
                            {
                                "pod": pod.get("pod"),
                                "container": container.get("container"),
                            }
                        )
    notes = [
        "These are task-agnostic operational observations; no hidden requirement output is included.",
        "Failures that select the same workload may be correlated symptoms rather than independent root causes.",
    ]
    if suppressed:
        notes.append(
            "One or more candidate commands suppress stdout/stderr; blank logs do not prove success. "
            "Use the captured command, container state, exit reason, events, and dependency context."
        )
    return {
        "schema_version": 2,
        "message": "The deployed candidate failed generic operational checks.",
        "primary_failures": [
            compact_finding(item) for item in (report.get("failures", []) or [])[:6]
        ],
        "downstream_correlated_observations": [
            compact_finding(item)
            for item in (report.get("correlated_observations", []) or [])[:4]
        ],
        "namespace_failure_context": report.get("namespace_failure_context"),
        "candidate_output_suppression_detected": bool(suppressed),
        "containers_suppressing_output": suppressed,
        "diagnostic_notes": notes,
    }


def run_execution_gate(
    task: BenchmarkTask,
    env: AttemptEnvironment,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run bounded, task-agnostic operational checks after deployment.

    Jobs and workloads share one deadline.  A terminal Job failure ends the
    readiness wait immediately, after which the gate snapshots every unresolved
    resource and captures logs/events before probe cleanup.  Hidden requirement
    data is never available to this function.
    """

    gate_started = time.monotonic()
    command_timeout = max(int(timeout_seconds), 10)
    diagnostic_timeout = min(command_timeout, 30)
    report: dict[str, Any] = {
        "schema_version": 2,
        "task_id": task.task_id,
        "namespace": task.namespace,
        "status": "running",
        "repair_on_failure": True,
        "task_specific": False,
        "wait_strategy": "failed_aware_polling_shared_deadline",
        "wait_budget_seconds": command_timeout,
        "checks": [],
        "failures": [],
        "correlated_observations": [],
        "probe_cleanup": [],
        "resource_counts": {},
    }

    ready = env.readyz()
    report["api_readiness_before_gate"] = _command_detail(ready)
    if ready.returncode != 0 or "ok" not in ready.stdout.lower():
        report["status"] = "infrastructure_error"
        report["error"] = (
            ready.stderr.strip() or ready.stdout.strip() or "Kubernetes API not ready"
        )
        report["duration_ms"] = round((time.monotonic() - gate_started) * 1000)
        return report

    # Namespace absence is a candidate defect, not an infrastructure outage.
    namespace_result = env.kubectl(
        ["get", "namespace", task.namespace, "-o", "json"],
        check=False,
        timeout=diagnostic_timeout,
    )
    namespace_check: dict[str, Any] = {
        "type": "candidate_namespace",
        "resource": f"namespace/{task.namespace}",
        "command": _command_detail(namespace_result),
    }
    report["checks"].append(namespace_check)
    if namespace_result.returncode != 0:
        confirmation = env.readyz()
        report["api_readiness_after_namespace_error"] = _command_detail(confirmation)
        detail = f"{namespace_result.stdout}\n{namespace_result.stderr}".lower()
        if _api_command_is_not_found(detail) and (
            confirmation.returncode == 0 and "ok" in confirmation.stdout.lower()
        ):
            report["failures"].append(
                {
                    "type": "required_namespace_missing",
                    "resource": f"namespace/{task.namespace}",
                    "diagnostic": _command_detail(namespace_result),
                }
            )
            report["status"] = "failed"
            report["repair_feedback"] = _repair_feedback(report)
        else:
            report["status"] = "infrastructure_error"
            report["error"] = (
                namespace_result.stderr.strip()
                or namespace_result.stdout.strip()
                or "Could not inspect candidate namespace"
            )
        report["duration_ms"] = round((time.monotonic() - gate_started) * 1000)
        return report
    try:
        namespace = json.loads(namespace_result.stdout)
    except json.JSONDecodeError as exc:
        report["status"] = "infrastructure_error"
        report["error"] = f"Namespace inspection returned malformed JSON: {exc}"
        report["duration_ms"] = round((time.monotonic() - gate_started) * 1000)
        return report
    namespace_phase = (namespace.get("status") or {}).get("phase")
    namespace_check["phase"] = namespace_phase
    if namespace_phase != "Active":
        report["failures"].append(
            {
                "type": "required_namespace_not_active",
                "resource": f"namespace/{task.namespace}",
                "phase": namespace_phase,
            }
        )
        report["status"] = "failed"
        report["repair_feedback"] = _repair_feedback(report)
        report["duration_ms"] = round((time.monotonic() - gate_started) * 1000)
        return report

    probe_jobs: dict[str, dict[str, Any]] = {}
    try:
        direct_jobs = [
            item
            for item in _resource_items(
                env, "jobs", task.namespace, timeout=diagnostic_timeout
            )
            if not item.get("metadata", {}).get("ownerReferences")
        ]
        cronjobs = _resource_items(
            env, "cronjobs", task.namespace, timeout=diagnostic_timeout
        )
        workloads = {
            resource: _resource_items(
                env, resource, task.namespace, timeout=diagnostic_timeout
            )
            for resource in _WORKLOADS
        }
        services = _resource_items(
            env, "services", task.namespace, timeout=diagnostic_timeout
        )
        report["resource_counts"] = {
            "direct_jobs": len(direct_jobs),
            "cronjobs": len(cronjobs),
            **{
                f"{resource}s": len(items)
                for resource, items in workloads.items()
            },
            "services": len(services),
        }
        monitored_count = (
            len(direct_jobs)
            + len(cronjobs)
            + sum(len(items) for items in workloads.values())
            + len(services)
        )
        # Prevent a generated fan-out from multiplying evaluator time or
        # diagnostic volume.  This cap is intentionally far above every task.
        if monitored_count > 100:
            report["failures"].append(
                {
                    "type": "operational_resource_limit_exceeded",
                    "observed": monitored_count,
                    "limit": 100,
                    "counts": report["resource_counts"],
                }
            )
            report["namespace_failure_context"] = _namespace_failure_context(
                env,
                task.namespace,
                command_timeout=diagnostic_timeout,
            )
            report["repair_feedback"] = _repair_feedback(report)
            report["status"] = "failed"
            report["duration_ms"] = round(
                (time.monotonic() - gate_started) * 1000
            )
            return report

        for cronjob in cronjobs:
            cronjob_name = _name(cronjob)
            job_name = _probe_name(cronjob_name)
            created = env.kubectl(
                [
                    "create",
                    "job",
                    job_name,
                    f"--from=cronjob/{cronjob_name}",
                    "-n",
                    task.namespace,
                ],
                check=False,
                timeout=diagnostic_timeout,
            )
            check: dict[str, Any] = {
                "type": "cronjob_execution",
                "resource": f"cronjob/{cronjob_name}",
                "probe_job": job_name,
                "create": _command_detail(created),
                "outcome": "pending" if created.returncode == 0 else "create_failed",
            }
            report["checks"].append(check)
            if created.returncode != 0:
                attribution = _failed_command_attribution(
                    env,
                    task.namespace,
                    created,
                    command_timeout=diagnostic_timeout,
                    allow_remote_exit=False,
                )
                check["failure_attribution"] = attribution
                if attribution["status"] == "candidate":
                    report["failures"].append(
                        {
                            "type": "cronjob_create_failed",
                            "resource": f"cronjob/{cronjob_name}",
                            "probe_job": job_name,
                            "diagnostic": _command_detail(created),
                            "failure_attribution": attribution,
                        }
                    )
                    continue
                report["status"] = "infrastructure_error"
                report["error"] = (
                    "Could not reliably exercise "
                    f"cronjob/{cronjob_name}: "
                    f"{created.stderr.strip() or created.stdout.strip()}"
                )
                break
            probe_jobs[job_name] = {
                "name": job_name,
                "origin": "cronjob",
                "resource": f"cronjob/{cronjob_name}",
                "check": check,
                "seen": False,
                "last_object": None,
                "state": {"state": "pending"},
                "history": [],
            }

        if report["status"] == "infrastructure_error":
            # A prior CronJob may already have produced a probe Job. Request
            # bounded, non-blocking cleanup before returning inconclusive.
            for job_name in probe_jobs:
                deleted = env.kubectl(
                    [
                        "delete",
                        "job",
                        job_name,
                        "-n",
                        task.namespace,
                        "--ignore-not-found=true",
                        "--wait=false",
                    ],
                    check=False,
                    timeout=diagnostic_timeout,
                )
                report["probe_cleanup"].append(
                    {"resource": f"job/{job_name}", **_command_detail(deleted)}
                )
            report["duration_ms"] = round(
                (time.monotonic() - gate_started) * 1000
            )
            return report

        monitored_jobs: dict[str, dict[str, Any]] = dict(probe_jobs)
        for job in direct_jobs:
            name = _name(job)
            check = {
                "type": "job_execution",
                "resource": f"job/{name}",
                "outcome": "pending",
            }
            report["checks"].append(check)
            monitored_jobs[name] = {
                "name": name,
                "origin": "job",
                "resource": f"job/{name}",
                "check": check,
                "seen": True,
                "last_object": job,
                "state": _job_state(job),
                "history": [],
            }

        monitored_workloads: dict[tuple[str, str], dict[str, Any]] = {}
        for resource, items in workloads.items():
            for item in items:
                name = _name(item)
                check = {
                    "type": "workload_rollout",
                    "resource": f"{resource}/{name}",
                    "outcome": "pending",
                }
                report["checks"].append(check)
                monitored_workloads[(resource, name)] = {
                    "resource_kind": resource,
                    "name": name,
                    "resource": f"{resource}/{name}",
                    "initial": item,
                    "last_object": item,
                    "state": _workload_state(resource, item),
                    "history": [],
                    "check": check,
                }

        poll_started = time.monotonic()
        deadline = poll_started + command_timeout
        definitive_failure = bool(report["failures"])
        timed_out = False

        while monitored_jobs or monitored_workloads:
            elapsed_ms = round((time.monotonic() - poll_started) * 1000)
            remaining = deadline - time.monotonic()
            poll_timeout = max(min(round(remaining) + 2, diagnostic_timeout), 5)

            current_workloads: dict[str, dict[str, dict[str, Any]]] = {}
            for resource in {
                key[0] for key in monitored_workloads
            }:
                current_workloads[resource] = {
                    _name(item): item
                    for item in _resource_items(
                        env,
                        resource,
                        task.namespace,
                        timeout=poll_timeout,
                    )
                }
            current_jobs = (
                {
                    _name(item): item
                    for item in _resource_items(
                        env, "jobs", task.namespace, timeout=poll_timeout
                    )
                }
                if monitored_jobs
                else {}
            )

            for (resource, name), record in monitored_workloads.items():
                item = current_workloads.get(resource, {}).get(name)
                state = (
                    {
                        "state": "failed",
                        "reason": "workload_disappeared_after_deployment",
                    }
                    if item is None
                    else _workload_state(resource, item)
                )
                if item is not None:
                    record["last_object"] = item
                if state.get("state") != record["state"].get("state"):
                    record["history"].append(
                        {
                            "elapsed_ms": elapsed_ms,
                            "state": state.get("state"),
                            "reason": state.get("reason"),
                        }
                    )
                record["state"] = state
                if state.get("state") == "failed":
                    definitive_failure = True

            for name, record in monitored_jobs.items():
                item = current_jobs.get(name)
                if item is None:
                    if record["seen"]:
                        state = {
                            "state": "failed",
                            "reason": "job_disappeared_before_result_capture",
                        }
                    elif elapsed_ms >= 15_000:
                        state = {
                            "state": "failed",
                            "reason": "created_job_never_became_observable",
                        }
                    else:
                        state = {"state": "pending", "reason": "awaiting_job_visibility"}
                else:
                    record["seen"] = True
                    record["last_object"] = item
                    state = _job_state(item)
                if state.get("state") != record["state"].get("state"):
                    record["history"].append(
                        {
                            "elapsed_ms": elapsed_ms,
                            "state": state.get("state"),
                            "reason": state.get("reason"),
                            "condition": state.get("condition"),
                        }
                    )
                record["state"] = state
                if state.get("state") == "failed":
                    definitive_failure = True

            all_resolved = all(
                record["state"].get("state") in {"ready", "failed"}
                for record in monitored_workloads.values()
            ) and all(
                record["state"].get("state") in {"succeeded", "failed"}
                for record in monitored_jobs.values()
            )
            if definitive_failure or all_resolved:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(min(2.0, max(deadline - time.monotonic(), 0)))

        report["operational_wait"] = {
            "strategy": "failed_aware_polling_shared_deadline",
            "budget_seconds": command_timeout,
            "elapsed_ms": round((time.monotonic() - poll_started) * 1000),
            "stopped_on_definitive_failure": definitive_failure,
            "deadline_reached": timed_out,
        }

        for record in monitored_workloads.values():
            state_name = record["state"].get("state")
            record["check"]["outcome"] = state_name
            record["check"]["state"] = record["state"]
            record["check"]["history"] = record["history"]
            if state_name == "ready":
                continue
            selector = _workload_selector(record["last_object"])
            pods = _pods_for_selector(
                env,
                task.namespace,
                selector,
                command_timeout=diagnostic_timeout,
            )
            diagnostics = _pod_execution_diagnostics(
                env,
                task.namespace,
                pods,
                command_timeout=diagnostic_timeout,
            )
            finding = {
                "resource": record["resource"],
                "state": record["state"],
                "history": record["history"],
                "pod_diagnostics": diagnostics,
            }
            if state_name == "failed" or timed_out:
                report["failures"].append(
                    {
                        "type": "workload_not_operational",
                        **finding,
                        "reason": (
                            record["state"].get("reason")
                            if state_name == "failed"
                            else "shared_operational_deadline_reached"
                        ),
                    }
                )
            else:
                report["correlated_observations"].append(
                    {
                        "type": "workload_pending_when_other_failure_became_definitive",
                        **finding,
                    }
                )

        for record in monitored_jobs.values():
            state_name = record["state"].get("state")
            record["check"]["outcome"] = state_name
            record["check"]["state"] = record["state"]
            record["check"]["history"] = record["history"]
            if state_name == "succeeded":
                continue
            pods = _pods_for_selector(
                env,
                task.namespace,
                f"job-name={record['name']}",
                command_timeout=diagnostic_timeout,
            )
            diagnostics = _pod_execution_diagnostics(
                env,
                task.namespace,
                pods,
                command_timeout=diagnostic_timeout,
            )
            finding = {
                "resource": record["resource"],
                "job": f"job/{record['name']}",
                "state": record["state"],
                "history": record["history"],
                "job_status": (
                    (record["last_object"] or {}).get("status", {})
                    if isinstance(record["last_object"], dict)
                    else {}
                ),
                "pod_diagnostics": diagnostics,
            }
            if state_name == "failed" or timed_out:
                report["failures"].append(
                    {
                        "type": (
                            "cronjob_execution_failed"
                            if record["origin"] == "cronjob"
                            else "job_execution_failed"
                        ),
                        **finding,
                        "reason": (
                            record["state"].get("reason")
                            if state_name == "failed"
                            else "shared_operational_deadline_reached"
                        ),
                    }
                )
            else:
                report["correlated_observations"].append(
                    {
                        "type": "job_pending_when_other_failure_became_definitive",
                        **finding,
                    }
                )

        # Capture evidence above before deleting probe Jobs.  Non-blocking
        # deletion avoids adding another per-probe timeout to a disposable cluster.
        for job_name in probe_jobs:
            deleted = env.kubectl(
                [
                    "delete",
                    "job",
                    job_name,
                    "-n",
                    task.namespace,
                    "--ignore-not-found=true",
                    "--wait=false",
                ],
                check=False,
                timeout=diagnostic_timeout,
            )
            cleanup = {
                "resource": f"job/{job_name}",
                **_command_detail(deleted),
            }
            report["probe_cleanup"].append(cleanup)
            if deleted.returncode != 0:
                report.setdefault("warnings", []).append(
                    f"Could not request deletion of execution probe job/{job_name}"
                )
                if not report["failures"]:
                    report["status"] = "infrastructure_error"
                    report["error"] = (
                        f"Could not remove execution-gate probe job {job_name}: "
                        f"{deleted.stderr.strip() or deleted.stdout.strip()}"
                    )

        if report["status"] == "infrastructure_error":
            report["duration_ms"] = round((time.monotonic() - gate_started) * 1000)
            return report

        for service in services:
            service_name = _name(service)
            selector = service.get("spec", {}).get("selector")
            if not isinstance(selector, dict) or not selector:
                continue
            endpoints = env.kubectl_json(
                ["get", "endpoints", service_name, "-n", task.namespace],
                timeout=diagnostic_timeout,
            )
            subsets = endpoints.get("subsets", []) or []
            ready_addresses = sum(
                len(subset.get("addresses", []) or [])
                for subset in subsets
                if isinstance(subset, dict)
            )
            not_ready_addresses = sum(
                len(subset.get("notReadyAddresses", []) or [])
                for subset in subsets
                if isinstance(subset, dict)
            )
            related_workloads = []
            for record in monitored_workloads.values():
                workload_selector = (
                    record["last_object"].get("spec", {})
                    .get("selector", {})
                    .get("matchLabels", {})
                )
                if (
                    isinstance(workload_selector, dict)
                    and workload_selector
                    and all(selector.get(k) == v for k, v in workload_selector.items())
                ):
                    related_workloads.append(record["resource"])
            check = {
                "type": "service_ready_endpoints",
                "resource": f"service/{service_name}",
                "ready_addresses": ready_addresses,
                "not_ready_addresses": not_ready_addresses,
                "selected_workloads": related_workloads,
            }
            report["checks"].append(check)
            if ready_addresses == 0:
                report["failures"].append(
                    {
                        "type": "service_has_no_ready_endpoints",
                        "resource": f"service/{service_name}",
                        "ready_addresses": ready_addresses,
                        "not_ready_addresses": not_ready_addresses,
                        "selected_workloads": related_workloads,
                        "likely_downstream_of": [
                            item["resource"]
                            for item in report["failures"]
                            if item.get("type") == "workload_not_operational"
                            and item.get("resource") in related_workloads
                        ],
                    }
                )
                continue

            selector_text = ",".join(
                f"{key}={value}" for key, value in sorted(selector.items())
            )
            selected = env.kubectl_json(
                [
                    "get",
                    "pods",
                    "-n",
                    task.namespace,
                    "-l",
                    selector_text,
                ],
                timeout=diagnostic_timeout,
            ).get("items", [])
            pod = _ready_pod(
                [item for item in selected if isinstance(item, dict)]
                if isinstance(selected, list)
                else []
            )
            if pod is None:
                report["failures"].append(
                    {
                        "type": "service_has_no_ready_selected_pod",
                        "resource": f"service/{service_name}",
                    }
                )
                continue
            pod_name = _name(pod)
            for port_spec in service.get("spec", {}).get("ports", []) or []:
                if port_spec.get("protocol", "TCP") != "TCP":
                    report["checks"].append(
                        {
                            "type": "service_target_port",
                            "resource": f"service/{service_name}",
                            "protocol": port_spec.get("protocol"),
                            "status": "not_checked_non_tcp",
                        }
                    )
                    continue
                target = port_spec.get("targetPort", port_spec.get("port"))
                container_name, target_number = _target_container_and_port(pod, target)
                if container_name is None or target_number is None:
                    report["failures"].append(
                        {
                            "type": "service_target_port_unresolved",
                            "resource": f"service/{service_name}",
                            "target_port": target,
                            "pod": pod_name,
                        }
                    )
                    continue
                connected = env.kubectl(
                    [
                        "exec",
                        "-n",
                        task.namespace,
                        pod_name,
                        "-c",
                        container_name,
                        "--",
                        "sh",
                        "-ec",
                        (
                            f"port=$(printf '%04X' {target_number}); "
                            "awk -v p=\":$port\" "
                            "'$2 ~ (p \"$\") && $4 == \"0A\" "
                            "{found=1} END {exit !found}' "
                            "/proc/net/tcp /proc/net/tcp6"
                        ),
                    ],
                    check=False,
                    timeout=diagnostic_timeout,
                )
                report["checks"].append(
                    {
                        "type": "service_target_port",
                        "resource": f"service/{service_name}",
                        "pod": pod_name,
                        "container": container_name,
                        "target_port": target_number,
                        "command": _command_detail(connected),
                    }
                )
                if connected.returncode != 0:
                    attribution = _failed_command_attribution(
                        env,
                        task.namespace,
                        connected,
                        command_timeout=diagnostic_timeout,
                        allow_remote_exit=True,
                    )
                    report["checks"][-1]["failure_attribution"] = attribution
                    if attribution["status"] == "candidate":
                        report["failures"].append(
                            {
                                "type": "service_target_not_listening",
                                "resource": f"service/{service_name}",
                                "pod": pod_name,
                                "container": container_name,
                                "target_port": target_number,
                                "diagnostic": _command_detail(connected),
                                "failure_attribution": attribution,
                            }
                        )
                        continue
                    report["status"] = "infrastructure_error"
                    report["error"] = (
                        "Could not reliably inspect the listening target for "
                        f"service/{service_name}: "
                        f"{connected.stderr.strip() or connected.stdout.strip()}"
                    )
                    report["duration_ms"] = round(
                        (time.monotonic() - gate_started) * 1000
                    )
                    return report

        for pod in _resource_items(
            env, "pods", task.namespace, timeout=diagnostic_timeout
        ):
            metadata = pod.get("metadata", {})
            if metadata.get("ownerReferences"):
                continue
            pod_name = _name(pod)
            phase = pod.get("status", {}).get("phase")
            ready_condition = next(
                (
                    condition.get("status")
                    for condition in pod.get("status", {}).get("conditions", []) or []
                    if condition.get("type") == "Ready"
                ),
                None,
            )
            passed = phase == "Succeeded" or (
                phase == "Running" and ready_condition == "True"
            )
            report["checks"].append(
                {
                    "type": "standalone_pod_execution",
                    "resource": f"pod/{pod_name}",
                    "phase": phase,
                    "ready": ready_condition,
                }
            )
            if not passed:
                report["failures"].append(
                    {
                        "type": "standalone_pod_not_operational",
                        "resource": f"pod/{pod_name}",
                        "phase": phase,
                        "ready": ready_condition,
                        "pod_diagnostics": _pod_execution_diagnostics(
                            env,
                            task.namespace,
                            [pod],
                            command_timeout=diagnostic_timeout,
                        ),
                    }
                )
    except Exception as exc:
        report["gate_exception"] = f"{type(exc).__name__}: {exc}"
        confirmation = env.readyz()
        report["api_readiness_after_gate_exception"] = _command_detail(
            confirmation
        )
        namespace_after = env.kubectl(
            ["get", "namespace", task.namespace, "-o", "json"],
            check=False,
            timeout=diagnostic_timeout,
        )
        report["namespace_after_gate_exception"] = _command_detail(namespace_after)
        namespace_detail = (
            f"{namespace_after.stdout}\n{namespace_after.stderr}"
        ).lower()
        candidate_namespace_failure: dict[str, Any] | None = None
        if (
            confirmation.returncode == 0
            and "ok" in confirmation.stdout.lower()
            and namespace_after.returncode != 0
            and _api_command_is_not_found(namespace_detail)
        ):
            candidate_namespace_failure = {
                "type": "required_namespace_disappeared_during_execution_gate",
                "resource": f"namespace/{task.namespace}",
                "gate_exception": report["gate_exception"],
                "diagnostic": _command_detail(namespace_after),
            }
        elif (
            confirmation.returncode == 0
            and "ok" in confirmation.stdout.lower()
            and namespace_after.returncode == 0
        ):
            try:
                phase = (
                    json.loads(namespace_after.stdout).get("status", {}).get("phase")
                )
            except Exception:
                phase = None
            if phase not in {None, "Active"}:
                candidate_namespace_failure = {
                    "type": "required_namespace_became_non_active_during_execution_gate",
                    "resource": f"namespace/{task.namespace}",
                    "phase": phase,
                    "gate_exception": report["gate_exception"],
                }
        if (
            report["failures"]
            and confirmation.returncode == 0
            and "ok" in confirmation.stdout.lower()
        ):
            report.setdefault("diagnostic_capture_errors", []).append(
                report["gate_exception"]
            )
            if candidate_namespace_failure is not None:
                report["failures"].append(candidate_namespace_failure)
            report["status"] = "failed"
            report["repair_feedback"] = _repair_feedback(report)
        elif candidate_namespace_failure is not None:
            report["failures"].append(candidate_namespace_failure)
            report["status"] = "failed"
            report["repair_feedback"] = _repair_feedback(report)
        else:
            report["status"] = "infrastructure_error"
            report["error"] = report["gate_exception"]
        report["duration_ms"] = round((time.monotonic() - gate_started) * 1000)
        return report

    if report["failures"]:
        report["namespace_failure_context"] = _namespace_failure_context(
            env,
            task.namespace,
            command_timeout=diagnostic_timeout,
        )
        report["repair_feedback"] = _repair_feedback(report)
        report["status"] = "failed"
    else:
        report["status"] = "passed"
    report["duration_ms"] = round((time.monotonic() - gate_started) * 1000)
    return report
