from __future__ import annotations

from typing import Any

from benchmark.private_tests.core import EvaluationError, Kubectl
from benchmark.private_tests.suites import run_suite

from .environment import AttemptEnvironment
from .tasks import BenchmarkTask


def run_post_execution_verifier(
    task: BenchmarkTask, env: AttemptEnvironment
) -> dict[str, Any]:
    """Run the hidden live oracle. Its failures are terminal, never repair input."""

    ready = env.readyz()
    if ready.returncode != 0 or "ok" not in ready.stdout.lower():
        return {
            "task_id": task.task_id,
            "status": "infrastructure_error",
            "error": ready.stderr.strip() or ready.stdout.strip() or "API not ready",
        }
    try:
        kube = Kubectl(
            context=env.kubectl_context,
            namespace=task.namespace,
            kubeconfig=env.container_kubeconfig,
            command_prefix=["docker", "exec", "-i", env.node_name, "kubectl"],
        )
        return run_suite(task.post_execution_suite, kube)
    except EvaluationError as exc:
        return {
            "task_id": task.task_id,
            "status": "infrastructure_error",
            "error": str(exc),
        }
