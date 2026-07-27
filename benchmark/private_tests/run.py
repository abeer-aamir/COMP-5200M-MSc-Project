from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import EvaluationError, Kubectl
from .suites import run_suite


TASK_NAMESPACES = {
    "pilot-001": "order-system",
    "pilot-002": "pipeline-ns",
}


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
        report = run_suite(args.task, kube)
    except EvaluationError as exc:
        report = {
            "task_id": args.task,
            "status": "infrastructure_error",
            "error": str(exc),
        }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
