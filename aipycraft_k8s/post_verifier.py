from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.private_tests.core import (
    EvaluationError,
    EvaluationInfrastructureError,
    Kubectl,
    SuiteExecutionError,
)
from benchmark.private_tests.suites import run_suite

from .environment import AttemptEnvironment
from .tasks import BenchmarkTask


def run_post_execution_verifier(
    task: BenchmarkTask, env: AttemptEnvironment, candidate_path: Path
) -> dict[str, Any]:
    """Run the hidden live oracle. Its failures are terminal, never repair input."""

    if task.post_execution_suite is None:
        return {
            "task_id": task.task_id,
            "status": "not_scored",
            "reason": "No private evaluator is registered for this generated task set",
        }
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
            command_timeout=env.command_timeout_seconds,
            command_prefix=["docker", "exec", "-i", env.node_name, "kubectl"],
            journal_callback=lambda result: env.journal_external_command(
                result, source="hidden_specification_oracle"
            ),
        )
        return run_suite(task.post_execution_suite, kube, candidate_path)
    except SuiteExecutionError as exc:
        report = exc.partial_report()
        ready_after = env.readyz()
        report["api_readiness_after_error"] = ready_after.audit_dict()
        return report
    except EvaluationInfrastructureError as exc:
        return {
            "task_id": task.task_id,
            "status": "infrastructure_error",
            "error": str(exc),
        }
    except EvaluationError as exc:
        return {
            "task_id": task.task_id,
            "status": "evaluator_error",
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "task_id": task.task_id,
            "status": "evaluator_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
