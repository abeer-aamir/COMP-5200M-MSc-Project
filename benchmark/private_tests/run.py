from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aipycraft_k8s.tasks import CANONICAL_TASK_NAMESPACES, load_tasks

from .core import (
    EvaluationError,
    EvaluationInfrastructureError,
    Kubectl,
    SuiteExecutionError,
)
from .suites import run_suite


TASK_NAMESPACES = {task_id: task.namespace for task_id, task in load_tasks().items()}
if TASK_NAMESPACES != CANONICAL_TASK_NAMESPACES:
    raise RuntimeError("task index and private evaluator registration differ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hidden post-execution checks against an already deployed task."
    )
    parser.add_argument("--task", choices=sorted(TASK_NAMESPACES), required=True)
    parser.add_argument(
        "--context",
        required=True,
        help="Exact kubectl context. An explicit context is required for safety.",
    )
    parser.add_argument("--kubeconfig", type=Path)
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Exact candidate manifest that was deployed into the target cluster.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        kube = Kubectl(
            context=args.context,
            namespace=TASK_NAMESPACES[args.task],
            kubeconfig=args.kubeconfig,
        )
        report = run_suite(args.task, kube, args.candidate)
    except SuiteExecutionError as exc:
        report = exc.partial_report()
    except EvaluationInfrastructureError as exc:
        report = {
            "task_id": args.task,
            "status": "infrastructure_error",
            "error": str(exc),
        }
    except EvaluationError as exc:
        report = {
            "task_id": args.task,
            "status": "evaluator_error",
            "error": str(exc),
        }
    except Exception as exc:
        report = {
            "task_id": args.task,
            "status": "evaluator_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
