from __future__ import annotations

import re
import uuid
from typing import Any

from .environment import AttemptEnvironment, EnvironmentError
from .tasks import BenchmarkTask


_WORKLOADS = ("deployment", "statefulset", "daemonset")


def _resource_items(
    env: AttemptEnvironment, resource: str, namespace: str
) -> list[dict[str, Any]]:
    value = env.kubectl_json(["get", resource, "-n", namespace])
    items = value.get("items", [])
    if not isinstance(items, list):
        raise EnvironmentError(f"kubectl returned malformed {resource} items")
    return [item for item in items if isinstance(item, dict)]


def _name(item: dict[str, Any]) -> str:
    return str(item.get("metadata", {}).get("name", "<unnamed>"))


def _command_detail(result: Any) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_ms": result.duration_ms,
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


def _init_container_diagnostics(
    env: AttemptEnvironment,
    workload: dict[str, Any],
    namespace: str,
    *,
    command_timeout: int,
) -> dict[str, Any]:
    """Best-effort init-container state and logs for a failed rollout."""

    selector = _workload_selector(workload)
    diagnostics: dict[str, Any] = {
        "status": "captured",
        "selector": selector,
        "pods": [],
        "pod_limit": 20,
        "init_container_limit_per_pod": 10,
    }
    if selector is None:
        diagnostics["status"] = "unavailable"
        diagnostics["error"] = "workload has no usable label selector"
        return diagnostics

    try:
        value = env.kubectl_json(
            ["get", "pods", "-n", namespace, "-l", selector]
        )
        raw_pods = value.get("items", [])
        if not isinstance(raw_pods, list):
            raise EnvironmentError("kubectl returned malformed pod items")
        pods = [pod for pod in raw_pods if isinstance(pod, dict)]
        diagnostics["matched_pods"] = len(pods)
        diagnostics["truncated_pods"] = max(len(pods) - 20, 0)
        for pod in pods[:20]:
            pod_name = _name(pod)
            status = pod.get("status", {})
            statuses = status.get("initContainerStatuses", []) or []
            by_name = {
                str(item.get("name")): item
                for item in statuses
                if isinstance(item, dict) and item.get("name")
            }
            init_specs = pod.get("spec", {}).get("initContainers", []) or []
            names = [
                str(item.get("name"))
                for item in init_specs
                if isinstance(item, dict) and item.get("name")
            ]
            for name in by_name:
                if name not in names:
                    names.append(name)
            pod_record: dict[str, Any] = {
                "pod": pod_name,
                "phase": status.get("phase"),
                "reason": status.get("reason"),
                "message": status.get("message"),
                "init_containers": [],
                "truncated_init_containers": max(len(names) - 10, 0),
            }
            for init_name in names[:10]:
                container_status = by_name.get(init_name, {})
                logs = env.kubectl(
                    [
                        "logs",
                        pod_name,
                        "-n",
                        namespace,
                        "-c",
                        init_name,
                        "--tail=200",
                    ],
                    check=False,
                    timeout=command_timeout,
                )
                container_record: dict[str, Any] = {
                    "container": init_name,
                    "ready": container_status.get("ready"),
                    "restart_count": container_status.get("restartCount"),
                    "state": container_status.get("state"),
                    "last_state": container_status.get("lastState"),
                    "image": container_status.get("image"),
                    "image_id": container_status.get("imageID"),
                    "logs": _command_detail(logs),
                }
                restart_count = container_status.get("restartCount")
                if isinstance(restart_count, int) and restart_count > 0:
                    previous_logs = env.kubectl(
                        [
                            "logs",
                            pod_name,
                            "-n",
                            namespace,
                            "-c",
                            init_name,
                            "--previous",
                            "--tail=200",
                        ],
                        check=False,
                        timeout=command_timeout,
                    )
                    container_record["previous_logs"] = _command_detail(previous_logs)
                pod_record["init_containers"].append(container_record)
            diagnostics["pods"].append(pod_record)
    except Exception as exc:
        diagnostics["status"] = "capture_error"
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
    return diagnostics


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


def run_execution_gate(
    task: BenchmarkTask,
    env: AttemptEnvironment,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run generic operational checks as post-deployment ground truth.

    This stage deliberately knows no requirement IDs or task-specific expected
    values. It checks only whether applied resources can become operational,
    and its findings are never returned to the model for regeneration.
    """

    report: dict[str, Any] = {
        "task_id": task.task_id,
        "namespace": task.namespace,
        "status": "running",
        "repair_on_failure": False,
        "task_specific": False,
        "checks": [],
        "failures": [],
        "probe_cleanup": [],
    }
    ready = env.readyz()
    if ready.returncode != 0 or "ok" not in ready.stdout.lower():
        report["status"] = "infrastructure_error"
        report["error"] = (
            ready.stderr.strip() or ready.stdout.strip() or "Kubernetes API not ready"
        )
        return report

    command_timeout = max(int(timeout_seconds), 10)
    kubectl_timeout = max(command_timeout - 5, 5)

    try:
        direct_jobs = [
            item
            for item in _resource_items(env, "jobs", task.namespace)
            if not item.get("metadata", {}).get("ownerReferences")
        ]
        cronjobs = _resource_items(env, "cronjobs", task.namespace)

        # Exercise every declared CronJob once. This is generic execution
        # validation, not a check of task-specific schedule or command details.
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
                timeout=command_timeout,
            )
            check: dict[str, Any] = {
                "type": "cronjob_execution",
                "resource": f"cronjob/{cronjob_name}",
                "probe_job": job_name,
                "create": _command_detail(created),
            }
            try:
                if created.returncode != 0:
                    report["failures"].append(
                        {
                            "type": "cronjob_create_failed",
                            "resource": f"cronjob/{cronjob_name}",
                            "diagnostic": _command_detail(created),
                        }
                    )
                else:
                    waited = env.kubectl(
                        [
                            "wait",
                            "--for=condition=complete",
                            f"job/{job_name}",
                            "-n",
                            task.namespace,
                            f"--timeout={kubectl_timeout}s",
                        ],
                        check=False,
                        timeout=command_timeout,
                    )
                    logs = env.kubectl(
                        [
                            "logs",
                            f"job/{job_name}",
                            "-n",
                            task.namespace,
                            "--all-containers=true",
                            "--tail=200",
                        ],
                        check=False,
                        timeout=command_timeout,
                    )
                    check["wait"] = _command_detail(waited)
                    check["logs"] = _command_detail(logs)
                    if waited.returncode != 0:
                        try:
                            job_status = env.kubectl_json(
                                ["get", "job", job_name, "-n", task.namespace]
                            ).get("status", {})
                        except Exception as exc:
                            job_status = {"inspection_error": f"{type(exc).__name__}: {exc}"}
                        report["failures"].append(
                            {
                                "type": "cronjob_execution_failed",
                                "resource": f"cronjob/{cronjob_name}",
                                "probe_job": job_name,
                                "diagnostic": _command_detail(waited),
                                "logs": logs.stdout,
                                "job_status": job_status,
                            }
                        )
            finally:
                deleted = env.kubectl(
                    [
                        "delete",
                        "job",
                        job_name,
                        "-n",
                        task.namespace,
                        "--ignore-not-found=true",
                        "--wait=true",
                        f"--timeout={kubectl_timeout}s",
                    ],
                    check=False,
                    timeout=command_timeout,
                )
                cleanup = {
                    "resource": f"job/{job_name}",
                    **_command_detail(deleted),
                }
                report["probe_cleanup"].append(cleanup)
                if deleted.returncode != 0:
                    report["status"] = "infrastructure_error"
                    report["error"] = (
                        f"Could not remove execution-gate probe job {job_name}: "
                        f"{deleted.stderr.strip() or deleted.stdout.strip()}"
                    )
            report["checks"].append(check)

        if report["status"] == "infrastructure_error":
            return report

        for resource in _WORKLOADS:
            for item in _resource_items(env, resource, task.namespace):
                resource_name = _name(item)
                waited = env.kubectl(
                    [
                        "rollout",
                        "status",
                        f"{resource}/{resource_name}",
                        "-n",
                        task.namespace,
                        f"--timeout={kubectl_timeout}s",
                    ],
                    check=False,
                    timeout=command_timeout,
                )
                check = {
                    "type": "workload_rollout",
                    "resource": f"{resource}/{resource_name}",
                    "command": _command_detail(waited),
                }
                report["checks"].append(check)
                if waited.returncode != 0:
                    init_diagnostics = _init_container_diagnostics(
                        env,
                        item,
                        task.namespace,
                        command_timeout=command_timeout,
                    )
                    check["init_container_diagnostics"] = init_diagnostics
                    report["failures"].append(
                        {
                            "type": "workload_not_operational",
                            "resource": f"{resource}/{resource_name}",
                            "diagnostic": _command_detail(waited),
                            "init_container_diagnostics": init_diagnostics,
                        }
                    )

        for job in direct_jobs:
            job_name = _name(job)
            waited = env.kubectl(
                [
                    "wait",
                    "--for=condition=complete",
                    f"job/{job_name}",
                    "-n",
                    task.namespace,
                    f"--timeout={kubectl_timeout}s",
                ],
                check=False,
                timeout=command_timeout,
            )
            report["checks"].append(
                {
                    "type": "job_execution",
                    "resource": f"job/{job_name}",
                    "command": _command_detail(waited),
                }
            )
            if waited.returncode != 0:
                logs = env.kubectl(
                    [
                        "logs",
                        f"job/{job_name}",
                        "-n",
                        task.namespace,
                        "--all-containers=true",
                        "--tail=200",
                    ],
                    check=False,
                    timeout=command_timeout,
                )
                report["failures"].append(
                    {
                        "type": "job_execution_failed",
                        "resource": f"job/{job_name}",
                        "diagnostic": _command_detail(waited),
                        "logs": logs.stdout,
                    }
                )

        for service in _resource_items(env, "services", task.namespace):
            service_name = _name(service)
            selector = service.get("spec", {}).get("selector")
            if not isinstance(selector, dict) or not selector:
                continue
            endpoints = env.kubectl_json(
                ["get", "endpoints", service_name, "-n", task.namespace]
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
            check = {
                "type": "service_ready_endpoints",
                "resource": f"service/{service_name}",
                "ready_addresses": ready_addresses,
                "not_ready_addresses": not_ready_addresses,
            }
            report["checks"].append(check)
            if ready_addresses == 0:
                report["failures"].append(
                    {
                        "type": "service_has_no_ready_endpoints",
                        "resource": f"service/{service_name}",
                        "ready_addresses": ready_addresses,
                        "not_ready_addresses": not_ready_addresses,
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
                ]
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
                    timeout=command_timeout,
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
                    report["failures"].append(
                        {
                            "type": "service_target_not_listening",
                            "resource": f"service/{service_name}",
                            "pod": pod_name,
                            "container": container_name,
                            "target_port": target_number,
                            "diagnostic": _command_detail(connected),
                        }
                    )

        for pod in _resource_items(env, "pods", task.namespace):
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
                    }
                )
    except Exception as exc:
        report["status"] = "infrastructure_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["status"] = "failed" if report["failures"] else "passed"
    return report
