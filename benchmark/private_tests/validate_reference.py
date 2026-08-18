from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from aipycraft_k8s.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from aipycraft_k8s.environment import IsolatedKindHarness
from aipycraft_k8s.pipeline import KubernetesAIPyCraftPipeline
from aipycraft_k8s.tasks import CANONICAL_TASK_NAMESPACES, load_task
from aipycraft_k8s.yaml_check import check_yaml_syntax


REFERENCE_ROOT = PROJECT_ROOT / "benchmark" / "private_tests" / "references"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically checkpoint the reference run without exposing a partial JSON file."""

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy a private known-good task reference through the real local "
            "Kind execution and hidden-oracle path. No model or paid API is used."
        )
    )
    parser.add_argument("--task", choices=sorted(CANONICAL_TASK_NAMESPACES), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    args = _args(argv)
    config = load_config(args.config)
    task = load_task(args.task)
    reference = REFERENCE_ROOT / f"{task.task_id}.yaml"
    if not reference.is_file():
        print(f"No private reference manifest exists for {task.task_id}", file=sys.stderr)
        return 2

    source = reference.read_text(encoding="utf-8")
    syntax = check_yaml_syntax(source)
    if not syntax.valid:
        print(f"Reference manifest is invalid YAML: {syntax.error}", file=sys.stderr)
        return 2

    run_id = f"reference{uuid.uuid4().hex[:16]}"
    output_dir = (
        config.environment.cache_root / "reference-validations" / run_id / "attempt-01"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate = output_dir / "candidate.yaml"
    shutil.copyfile(reference, candidate)

    harness = IsolatedKindHarness(
        config.environment,
        command_timeout_seconds=config.pipeline.command_timeout_seconds,
        max_attempts=1,
    )
    pipeline = KubernetesAIPyCraftPipeline(
        config,
        cast(Any, None),
        harness,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "task_id": task.task_id,
        "paid_calls_made": False,
        "reference": str(reference),
        "reference_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "artifacts": str(output_dir),
        "started_at": started_at,
        "status": "running",
    }
    report_path = output_dir / "reference_validation.json"
    _write_report(report_path, report)
    try:
        report["preflight"] = harness.preflight()
        _write_report(report_path, report)
        with harness.attempt(run_id, 1, output_dir) as environment:
            report["evaluation"] = pipeline._evaluate_candidate_once(
                task,
                environment,
                candidate,
                output_dir,
            )
        for name, field in (
            ("setup_stages.json", "setup_stages"),
            ("cleanup.json", "cleanup"),
        ):
            path = output_dir / name
            if path.is_file():
                report[field] = json.loads(path.read_text(encoding="utf-8"))
        report["status"] = (
            "passed"
            if report.get("evaluation", {}).get("result") == "accepted"
            else "failed"
        )
    except BaseException as exc:
        report["status"] = "infrastructure_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        cleanup = output_dir / "cleanup.json"
        if cleanup.is_file():
            report["cleanup"] = json.loads(cleanup.read_text(encoding="utf-8"))

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["duration_ms"] = round((time.monotonic() - started_monotonic) * 1000)
    _write_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
