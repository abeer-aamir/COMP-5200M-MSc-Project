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
                    report["failures"].append(
                        {
                            "type": "workload_not_operational",
                            "resource": f"{resource}/{resource_name}",
                            "diagnostic": _command_detail(waited),
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
