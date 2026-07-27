from __future__ import annotations

import time
from typing import Any

from .environment import AttemptEnvironment, EnvironmentError


_INFRA_NAMESPACES = {
    "kube-system",
    "kube-public",
    "kube-node-lease",
    "local-path-storage",
    "aipc-harness-smoke",
}
_FATAL_WAITING_REASONS = {
    "CrashLoopBackOff",
    "CreateContainerConfigError",
    "CreateContainerError",
    "ErrImagePull",
    "ImagePullBackOff",
    "InvalidImageName",
    "RunContainerError",
    "StartError",
}
_FATAL_EVENT_REASONS = {
    "BackOff",
    "ErrImagePull",
    "Failed",
    "FailedAttachVolume",
    "FailedCreate",
    "FailedCreatePodSandBox",
    "FailedMount",
}


def _pod_identity(pod: dict[str, Any]) -> tuple[str, str]:
    metadata = pod.get("metadata", {})
    return str(metadata.get("namespace", "default")), str(
        metadata.get("name", "<unnamed>")
    )


def _container_failures(
    pod: dict[str, Any], field: str, container_type: str
) -> list[dict[str, str]]:
    namespace, pod_name = _pod_identity(pod)
    failures: list[dict[str, str]] = []
    for status in pod.get("status", {}).get(field, []) or []:
        waiting = (status.get("state") or {}).get("waiting") or {}
        terminated = (status.get("state") or {}).get("terminated") or {}
        reason = waiting.get("reason")
        if reason in _FATAL_WAITING_REASONS:
            failures.append(
                {
                    "type": "container_waiting",
                    "namespace": namespace,
                    "pod": pod_name,
                    "container": str(status.get("name", "<unnamed>")),
                    "container_type": container_type,
                    "reason": str(reason),
                    "message": str(waiting.get("message", "")),
                }
            )
        exit_code = terminated.get("exitCode")
        if exit_code not in {None, 0} and not status.get("restartCount"):
            failures.append(
                {
                    "type": "container_terminated",
                    "namespace": namespace,
                    "pod": pod_name,
                    "container": str(status.get("name", "<unnamed>")),
                    "container_type": container_type,
                    "reason": str(terminated.get("reason", "Error")),
                    "message": f"exitCode={exit_code}: {terminated.get('message', '')}",
                }
            )
    return failures


def extract_pod_failures(pods: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return concrete execution failures without checking task requirements."""

    failures: list[dict[str, str]] = []
    for pod in pods:
        namespace, name = _pod_identity(pod)
        if namespace in _INFRA_NAMESPACES:
            continue
        status = pod.get("status", {})
        if status.get("phase") == "Failed":
            failures.append(
                {
                    "type": "pod_failed",
                    "namespace": namespace,
                    "pod": name,
                    "reason": str(status.get("reason", "Failed")),
                    "message": str(status.get("message", "")),
                }
            )
        for condition in status.get("conditions", []) or []:
            if (
                condition.get("type") == "PodScheduled"
                and condition.get("status") == "False"
                and condition.get("reason") == "Unschedulable"
            ):
                failures.append(
                    {
                        "type": "unschedulable",
                        "namespace": namespace,
                        "pod": name,
                        "reason": "Unschedulable",
                        "message": str(condition.get("message", "")),
                    }
                )
        failures.extend(_container_failures(pod, "initContainerStatuses", "init"))
        failures.extend(_container_failures(pod, "containerStatuses", "main"))
    return failures


def extract_event_failures(
    events: list[dict[str, Any]], pods: list[dict[str, Any]] | None = None
) -> list[dict[str, str]]:
    def recovered(pod: dict[str, Any]) -> bool:
        status = pod.get("status", {})
        if status.get("phase") in {"Running", "Succeeded"}:
            return True
        for field in ("initContainerStatuses", "containerStatuses"):
            for container in status.get(field, []) or []:
                state = container.get("state") or {}
                if state.get("running"):
                    return True
                terminated = state.get("terminated") or {}
                if terminated.get("exitCode") == 0:
                    return True
        return False

    recovered_pods = {
        _pod_identity(pod) for pod in pods or [] if recovered(pod)
    }
    failures: list[dict[str, str]] = []
    for event in events:
        metadata = event.get("metadata", {})
        namespace = str(metadata.get("namespace", "default"))
        if namespace in _INFRA_NAMESPACES or event.get("type") != "Warning":
            continue
        reason = str(event.get("reason", ""))
        if reason not in _FATAL_EVENT_REASONS:
            continue
        involved = event.get("involvedObject", {})
        involved_name = str(involved.get("name", "<unknown-object>"))
        if (
            involved.get("kind") == "Pod"
            and (namespace, involved_name) in recovered_pods
        ):
            # Events are historical. Do not repair a candidate for a transient
            # warning after the affected Pod has demonstrably recovered.
            continue
        failures.append(
            {
                "type": "warning_event",
                "namespace": namespace,
                "pod": involved_name,
                "reason": reason,
                "message": str(event.get("message", "")),
            }
        )
    return failures


def _pod_summary(pod: dict[str, Any]) -> dict[str, Any]:
    namespace, name = _pod_identity(pod)
    status = pod.get("status", {})
    return {
        "namespace": namespace,
        "name": name,
        "phase": status.get("phase"),
        "conditions": [
            {
                "type": item.get("type"),
                "status": item.get("status"),
                "reason": item.get("reason"),
                "message": item.get("message"),
            }
            for item in status.get("conditions", []) or []
        ],
        "init_container_statuses": status.get("initContainerStatuses", []),
        "container_statuses": status.get("containerStatuses", []),
    }


def observe_runtime(
    env: AttemptEnvironment,
    *,
    seconds: int,
    poll_seconds: float,
) -> dict[str, Any]:
    """Observe only concrete Kubernetes execution errors after deployment."""

    deadline = time.monotonic() + seconds
    latest_pods: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    while True:
        try:
            latest_pods = env.kubectl_json(["get", "pods", "--all-namespaces"]).get(
                "items", []
            )
        except Exception as exc:
            raise EnvironmentError(f"Could not inspect deployed pods: {exc}") from exc
        failures = extract_pod_failures(latest_pods)
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_seconds)

    events: list[dict[str, Any]] = []
    try:
        events = env.kubectl_json(["get", "events", "--all-namespaces"]).get(
            "items", []
        )
    except Exception as exc:
        raise EnvironmentError(f"Could not inspect Kubernetes events: {exc}") from exc
    if not failures:
        failures = extract_event_failures(events, latest_pods)
    return {
        "status": "execution_error" if failures else "no_concrete_execution_error",
        "observation_seconds": seconds,
        "failures": failures,
        "pods": [
            _pod_summary(pod)
            for pod in latest_pods
            if _pod_identity(pod)[0] not in _INFRA_NAMESPACES
        ],
        "warning_events": [
            {
                "namespace": item.get("metadata", {}).get("namespace", "default"),
                "reason": item.get("reason"),
                "message": item.get("message"),
                "involved_object": item.get("involvedObject", {}),
            }
            for item in events
            if item.get("type") == "Warning"
            and item.get("metadata", {}).get("namespace", "default")
            not in _INFRA_NAMESPACES
        ],
    }
