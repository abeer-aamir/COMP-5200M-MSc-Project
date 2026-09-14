from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


class ResultsLedgerError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultsLedgerError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultsLedgerError(f"Expected an object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _compact_record(
    record: dict[str, Any], *, project_root: Path, source: Path
) -> dict[str, Any]:
    return {
        "run_id": record.get("run_id"),
        "task_id": record.get("task_id"),
        "treatment_id": record.get("treatment_id"),
        "model": record.get("model"),
        "reasoning_effort": record.get("reasoning_effort"),
        "validator_enabled": bool(record.get("validator_enabled")),
        "validator_model": record.get("validator_model"),
        "validator_reasoning_effort": record.get("validator_reasoning_effort"),
        "status": record.get("status"),
        "terminal_attempt_result": record.get("terminal_attempt_result"),
        "attempts": int(record.get("attempts") or 0),
        "regenerations": int(record.get("regenerations") or 0),
        "provider_unavailable_retries": int(
            record.get("provider_unavailable_retries") or 0
        ),
        "validator_unparsable_response_retries": int(
            record.get("validator_unparsable_response_retries") or 0
        ),
        "empty_length_provider_retries": int(
            record.get("empty_length_provider_retries") or 0
        ),
        "deployment_successes": int(record.get("deployment_successes") or 0),
        "hidden_verifier_reached": bool(record.get("hidden_verifier_reached")),
        "hidden_passed": int(record.get("hidden_passed") or 0),
        "hidden_total": int(record.get("hidden_total") or 0),
        "hidden_failed_requirement_ids": list(
            record.get("hidden_failed_requirement_ids") or []
        ),
        "total_tokens": int(record.get("total_tokens") or 0),
        "cost_usd": str(record.get("cost_usd") or "0"),
        "duration_ms": int(record.get("duration_ms") or 0),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "git_commit": record.get("git_commit"),
        "analysis_record": _relative(source, project_root),
        "analysis_record_sha256": _sha256(source),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("treatment_id") or "standalone")].append(row)
    summaries: list[dict[str, Any]] = []
    for treatment_id, group in sorted(grouped.items()):
        passed = sum(int(row["hidden_passed"]) for row in group)
        total = sum(int(row["hidden_total"]) for row in group)
        perfect = sum(
            1
            for row in group
            if int(row["hidden_total"]) > 0
            and int(row["hidden_passed"]) == int(row["hidden_total"])
        )
        summaries.append(
            {
                "treatment_id": treatment_id,
                "rows": len(group),
                "perfect_hidden_rows": perfect,
                "hidden_passed": passed,
                "hidden_total": total,
                "hidden_score_percent": (
                    round(100 * passed / total, 4) if total else None
                ),
                "run_tokens": sum(int(row["total_tokens"]) for row in group),
                "run_cost_usd": str(
                    sum((Decimal(row["cost_usd"]) for row in group), Decimal("0"))
                ),
                "status_counts": dict(
                    sorted(Counter(str(row["status"]) for row in group).items())
                ),
            }
        )
    return summaries


def _apply_adjudication(
    study: dict[str, Any], adjudication: dict[str, Any], *, source_path: Path
) -> dict[str, Any]:
    if adjudication.get("study_name") != study.get("study_name"):
        raise ResultsLedgerError(
            f"{source_path}: study_name does not match the selected study"
        )
    rows = {str(row["run_id"]): dict(row) for row in study["rows"]}
    seen: set[tuple[str, str]] = set()
    for change in adjudication.get("changes", []):
        run_id = str(change.get("run_id"))
        requirement_id = str(change.get("requirement_id"))
        key = (run_id, requirement_id)
        if key in seen:
            raise ResultsLedgerError(f"{source_path}: duplicate change {key}")
        seen.add(key)
        row = rows.get(run_id)
        if row is None:
            raise ResultsLedgerError(f"{source_path}: unknown run_id {run_id}")
        if change.get("task_id") != row.get("task_id"):
            raise ResultsLedgerError(f"{source_path}: task mismatch for {run_id}")
        failed = list(row["hidden_failed_requirement_ids"])
        if requirement_id not in failed:
            raise ResultsLedgerError(
                f"{source_path}: {run_id}/{requirement_id} was not a raw failure"
            )
        failed.remove(requirement_id)
        row["hidden_failed_requirement_ids"] = failed
        row["hidden_passed"] = int(row["hidden_passed"]) + 1

    adjusted_rows = list(rows.values())
    adjusted = _aggregate(adjusted_rows)
    declared = adjudication.get("declared_adjusted_treatments")
    if declared is not None:
        actual = [
            {
                "treatment_id": item["treatment_id"],
                "rows": item["rows"],
                "perfect_hidden_rows": item["perfect_hidden_rows"],
                "hidden_passed": item["hidden_passed"],
                "hidden_total": item["hidden_total"],
                "hidden_score_percent": item["hidden_score_percent"],
            }
            for item in adjusted
        ]
        if actual != declared:
            raise ResultsLedgerError(
                f"{source_path}: declared adjusted totals do not match changes"
            )
    return {
        "adjudication_id": adjudication.get("adjudication_id"),
        "source": source_path.as_posix(),
        "scope": adjudication.get("scope"),
        "changes": len(seen),
        "adjusted_treatments": adjusted,
    }


def build(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    studies_root = project_root / "benchmark" / "study_runs"
    runs_root = project_root / "benchmark" / "aipycraft_runs"
    linked_runs: set[str] = set()
    studies: list[dict[str, Any]] = []

    for manifest_path in sorted(studies_root.glob("*/manifest.json")):
        state_path = manifest_path.parent / "state.json"
        if not state_path.is_file():
            continue
        manifest = _read(manifest_path)
        state = _read(state_path)
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for row_id, row_state in sorted(state.get("rows", {}).items()):
            invocations = list(row_state.get("invocations") or [])
            if not invocations:
                missing.append(row_id)
                continue
            invocation = invocations[-1]
            run_dir = Path(str(invocation.get("run_dir") or ""))
            source = run_dir / "private" / "analysis_record.json"
            if not source.is_file():
                missing.append(row_id)
                continue
            record = _compact_record(_read(source), project_root=project_root, source=source)
            record["row_id"] = row_id
            record["scheduler_status"] = row_state.get("status")
            record["infrastructure_reruns"] = int(
                row_state.get("infrastructure_reruns") or 0
            )
            rows.append(record)
            linked_runs.add(str(record["run_id"]))
        studies.append(
            {
                "study_name": manifest.get("study_name"),
                "status": state.get("status"),
                "manifest": _relative(manifest_path, project_root),
                "manifest_sha256": _sha256(manifest_path),
                "state": _relative(state_path, project_root),
                "state_sha256": _sha256(state_path),
                "shared_initial_generation_cost_usd": str(
                    manifest.get("shared_initial_generation_cost_usd") or "0"
                ),
                "observed_treatment_cost_usd": str(
                    state.get("observed_treatment_cost_usd") or "0"
                ),
                "observed_total_cost_usd": str(
                    state.get("observed_total_cost_usd") or "0"
                ),
                "rows": rows,
                "missing_terminal_analysis_rows": missing,
                "raw_treatments": _aggregate(rows),
            }
        )

    standalone: list[dict[str, Any]] = []
    for source in sorted(runs_root.glob("*/private/analysis_record.json")):
        record = _compact_record(_read(source), project_root=project_root, source=source)
        if str(record["run_id"]) not in linked_runs:
            standalone.append(record)

    adjudications: list[dict[str, Any]] = []
    by_study = {str(study["study_name"]): study for study in studies}
    adjudication_root = project_root / "benchmark" / "study_results" / "adjudications"
    for source in sorted(adjudication_root.glob("*.json")):
        value = _read(source)
        study_name = str(value.get("study_name"))
        study = by_study.get(study_name)
        if study is None:
            raise ResultsLedgerError(f"{source}: unknown study {study_name}")
        expected_state = value.get("source_state_sha256")
        if expected_state and expected_state != study["state_sha256"]:
            raise ResultsLedgerError(f"{source}: source state hash changed")
        result = _apply_adjudication(study, value, source_path=source)
        result["source"] = _relative(source, project_root)
        result["source_sha256"] = _sha256(source)
        adjudications.append(result)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "raw_artifacts_modified": False,
            "study_rows_use_terminal_scheduler_invocation": True,
            "shared_initial_generation_cost_is_reported_once_per_study": True,
            "adjudications_are_versioned_sidecars": True,
        },
        "inventory": {
            "completed_or_partial_studies": len(studies),
            "study_rows": sum(len(study["rows"]) for study in studies),
            "standalone_runs": len(standalone),
            "total_analysis_records": sum(len(study["rows"]) for study in studies)
            + len(standalone),
        },
        "studies": studies,
        "standalone_runs": standalone,
        "standalone_raw_summary": _aggregate(standalone),
        "adjudications": adjudications,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a compact raw-score ledger and validate adjudications"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "study_results" / "results_ledger.json",
    )
    args = parser.parse_args(argv)
    try:
        value = build(args.project_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "output": str(args.output.resolve()),
                    "inventory": value["inventory"],
                    "adjudications": len(value["adjudications"]),
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
