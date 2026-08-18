from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


class AnalysisExportError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisExportError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisExportError(f"Expected an object in {path}")
    return value


def export(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    state = _read(manifest_path.parent / "state.json")
    manifest = _read(manifest_path)
    manifest_rows = {row["row_id"]: row for row in manifest["rows"]}
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row_id, row_state in state.get("rows", {}).items():
        row = manifest_rows[row_id]
        for invocation in row_state.get("invocations", []):
            run_dir = invocation.get("run_dir")
            record_path = (
                Path(str(run_dir)) / "private" / "analysis_record.json"
                if run_dir
                else None
            )
            if record_path is None or not record_path.is_file():
                missing.append(
                    {
                        "row_id": row_id,
                        "invocation": invocation.get("invocation"),
                        "reason": "analysis_record_not_available",
                    }
                )
                continue
            records.append(
                {
                    "study_name": manifest["study_name"],
                    "row_id": row_id,
                    "scheduler_invocation": invocation.get("invocation"),
                    "scheduler_status": row_state.get("status"),
                    "scheduler_infrastructure_reruns": row_state.get(
                        "infrastructure_reruns", 0
                    ),
                    "paired_candidate_id": row["candidate_id"],
                    **_read(record_path),
                }
            )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "analysis_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    csv_path = output_dir / "analysis_records.csv"
    scalar_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    report = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "raw_artifacts_modified": False,
        "records": len(records),
        "missing_records": missing,
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export study analysis rows without modifying raw run artifacts"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(export(args.manifest, args.output), indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
