from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from task_authoring.config import load_env_file

from .candidate_bank import CandidateBankError, create_entry, load_entry
from .cli import PAID_CONFIRMATION
from .config import PROJECT_ROOT, load_config, treatment_id
from .environment import EnvironmentPreparer
from .openrouter import OpenRouterTextClient
from .tasks import load_task, load_tasks


STUDY_SCHEMA_VERSION = 1
DEFAULT_BANK_ROOT = PROJECT_ROOT / "benchmark" / "candidate_bank"
DEFAULT_STUDY_ROOT = PROJECT_ROOT / "benchmark" / "study_runs"


class StudyError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": _utc_now(), **event}) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudyError(f"Expected a JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confirm(value: str) -> None:
    if value != PAID_CONFIRMATION:
        raise StudyError(
            "Paid calls were not confirmed. Pass exactly "
            f"--confirm-paid-calls {PAID_CONFIRMATION}"
        )


def generate_candidate(
    *, config_path: Path, task_id: str, output_root: Path
) -> Path:
    started = time.monotonic()
    config = load_config(config_path)
    task = load_task(task_id)
    load_env_file()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise StudyError("OPENROUTER_API_KEY is missing")
    client = OpenRouterTextClient(api_key, config.api)
    try:
        entry = create_entry(
            task=task, config=config, client=client, output_root=output_root
        )
    except Exception as exc:
        audit_result = getattr(exc, "audit_result", None)
        failure_artifact = None
        raw_response_sha256 = None
        if audit_result is not None:
            failure_dir = (
                output_root.resolve()
                / "failed_generations"
                / f"{task_id}-{uuid.uuid4().hex[:12]}"
            )
            failure_dir.mkdir(parents=True, exist_ok=False)
            raw_response = str(audit_result.raw_text)
            (failure_dir / "raw_response.txt").write_text(
                raw_response, encoding="utf-8"
            )
            _write_json_atomic(failure_dir / "generation.json", audit_result.audit_dict())
            failure_artifact = str(failure_dir)
            raw_response_sha256 = hashlib.sha256(
                raw_response.encode("utf-8")
            ).hexdigest()
        _append_event(
            output_root.resolve() / "generation_events.jsonl",
            {
                "event": "candidate_generation_failed",
                "task_id": task_id,
                "config": str(config.path),
                "duration_ms": round((time.monotonic() - started) * 1000),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "transport_attempts": getattr(exc, "transport_attempts", None),
                "unknown_cost_attempts": getattr(exc, "unknown_cost_attempts", None),
                "audit_result": (
                    audit_result.audit_dict() if audit_result is not None else None
                ),
                "failure_artifact": failure_artifact,
                "raw_response_sha256": raw_response_sha256,
            },
        )
        raise
    _append_event(
        output_root.resolve() / "generation_events.jsonl",
        {
            "event": "candidate_generation_completed",
            "task_id": task_id,
            "config": str(config.path),
            "candidate_id": entry.candidate_id,
            "candidate_record": str(entry.record_path),
            "duration_ms": round((time.monotonic() - started) * 1000),
        },
    )
    return entry.record_path


def create_manifest(
    *,
    name: str,
    candidate_paths: list[Path],
    config_paths: list[Path],
    output_path: Path,
    max_infrastructure_reruns: int,
    max_study_cost_usd: Decimal | None,
) -> dict[str, Any]:
    if not candidate_paths or not config_paths:
        raise StudyError("At least one candidate and treatment config are required")
    loaded_configs = [load_config(path) for path in config_paths]

    def comparison_contract(config: Any) -> dict[str, Any]:
        return {
            "max_regenerations": config.pipeline.max_regenerations,
            "candidate_api_loss_confirmation_replays": (
                config.pipeline.candidate_api_loss_confirmation_replays
            ),
            "runtime_observation_seconds": config.pipeline.runtime_observation_seconds,
            "runtime_poll_seconds": config.pipeline.runtime_poll_seconds,
            "command_timeout_seconds": config.pipeline.command_timeout_seconds,
            "environment_lock_sha256": _sha256_file(config.environment.lock.path),
            "kind_config_sha256": _sha256_file(config.environment.kind_config),
            "repair_prompt_sha256": _sha256_file(config.prompts.regenerate),
            "validator_detailed_prompt_sha256": _sha256_file(
                config.prompts.validator_detailed
            ),
            "validator_verdict_prompt_sha256": _sha256_file(
                config.prompts.validator_verdict_only
            ),
        }

    baseline_contract = comparison_contract(loaded_configs[0])
    for config in loaded_configs[1:]:
        if comparison_contract(config) != baseline_contract:
            raise StudyError(
                "Treatment configs differ outside the validator intervention; "
                "execution, repair, environment, and timeout settings must match"
            )
    configured_treatments = [treatment_id(config) for config in loaded_configs]
    if len(set(configured_treatments)) != len(configured_treatments):
        raise StudyError("Treatment configs contain a duplicate treatment ID")
    rows: list[dict[str, Any]] = []
    seen_treatments: set[str] = set()
    shared_initial_cost = Decimal("0")
    for candidate_index, candidate_path in enumerate(candidate_paths):
        raw = _read_json(candidate_path.resolve())
        task_id = str(raw.get("task_id", ""))
        task = load_task(task_id)
        generation = raw.get("generation", {}) or {}
        shared_initial_cost += Decimal(str(generation.get("cost_usd", "0")))
        rotation = candidate_index % len(config_paths)
        ordered_configs = loaded_configs[rotation:] + loaded_configs[:rotation]
        for config in ordered_configs:
            entry = load_entry(candidate_path, task=task, config=config)
            treatment = treatment_id(config)
            seen_treatments.add(treatment)
            rows.append(
                {
                    "row_id": f"row-{len(rows) + 1:04d}",
                    "task_id": task_id,
                    "candidate_id": entry.candidate_id,
                    "candidate_record": str(entry.record_path),
                    "candidate_record_sha256": entry.provenance()["record_sha256"],
                    "config": str(config.path),
                    "config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
                    "treatment_id": treatment,
                }
            )
    manifest = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_name": name,
        "created_at": _utc_now(),
        "execution_policy": {
            "sequential_per_machine": True,
            "continue_after_row_failure": True,
            "resume_from_atomic_state": True,
            "same_candidate_across_treatments": True,
            "treatment_order": (
                "deterministic rotation by candidate index to counterbalance "
                "machine warm-up and order effects"
            ),
            "infrastructure_rerun_policy": (
                "rerun the same candidate/treatment only after a top-level "
                "infrastructure_error; never rerun candidate_failed automatically"
            ),
            "max_infrastructure_reruns": max_infrastructure_reruns,
            "max_study_cost_usd": (
                None if max_study_cost_usd is None else str(max_study_cost_usd)
            ),
        },
        "treatments": sorted(seen_treatments),
        "shared_initial_generation_cost_usd": str(shared_initial_cost),
        "rows": rows,
    }
    _write_json_atomic(output_path.resolve(), manifest)
    return manifest


def _public_result(stdout: str) -> dict[str, Any] | None:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    results = value.get("results") if isinstance(value, dict) else None
    if isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict):
        return results[0]
    return None


def _initial_state(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "study_name": manifest["study_name"],
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "scheduler_pid": os.getpid(),
        "machine": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "status": "running",
        "rows": {
            row["row_id"]: {
                "status": "pending",
                "task_id": row["task_id"],
                "treatment_id": row["treatment_id"],
                "candidate_id": row["candidate_id"],
                "config": row["config"],
                "invocations": [],
                "infrastructure_reruns": 0,
            }
            for row in manifest["rows"]
        },
    }


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and str(pid) in result.stdout


def run_manifest(manifest_path: Path, *, only_row: str | None = None) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != STUDY_SCHEMA_VERSION:
        raise StudyError("Unsupported study manifest schema")
    run_dir = manifest_path.parent
    state_path = run_dir / "state.json"
    events_path = run_dir / "scheduler_events.jsonl"
    state = (
        _read_json(state_path)
        if state_path.exists()
        else _initial_state(manifest_path, manifest)
    )
    existing_scheduler_pid = state.get("scheduler_pid")
    if (
        existing_scheduler_pid != os.getpid()
        and _pid_alive(existing_scheduler_pid)
        and state.get("status") == "running"
    ):
        raise StudyError(
            f"Scheduler PID {existing_scheduler_pid} is already running this study"
        )
    if state.get("manifest_sha256") != _sha256_file(manifest_path):
        raise StudyError(
            "Manifest changed after scheduler state was created; create a new study"
        )
    state["scheduler_pid"] = os.getpid()
    state["status"] = "running"
    state["updated_at"] = _utc_now()
    _write_json_atomic(state_path, state)
    max_reruns = int(manifest["execution_policy"]["max_infrastructure_reruns"])
    raw_cap = manifest["execution_policy"].get("max_study_cost_usd")
    cost_cap = None if raw_cap is None else Decimal(str(raw_cap))
    observed_treatment_cost = sum(
        Decimal(str(invocation.get("cost_usd", "0")))
        for item in state["rows"].values()
        for invocation in item.get("invocations", [])
    )
    shared_initial_cost = Decimal(
        str(manifest.get("shared_initial_generation_cost_usd", "0"))
    )
    observed_cost = shared_initial_cost + observed_treatment_cost

    selected = [
        row
        for row in manifest["rows"]
        if only_row is None or row["row_id"] == only_row
    ]
    if only_row is not None and not selected:
        raise StudyError(f"Unknown row ID {only_row}")

    for row in selected:
        if _sha256_file(Path(row["candidate_record"])) != row[
            "candidate_record_sha256"
        ]:
            raise StudyError(
                f"Candidate record changed after manifest creation: {row['row_id']}"
            )
        if _sha256_file(Path(row["config"])) != row["config_sha256"]:
            raise StudyError(
                f"Treatment config changed after manifest creation: {row['row_id']}"
            )
        row_state = state["rows"][row["row_id"]]
        if row_state["status"] in {"completed", "candidate_failed"}:
            continue
        if row_state["status"] == "running" and _pid_alive(
            row_state.get("child_pid")
        ):
            _append_event(
                events_path,
                {"event": "live_row_left_untouched", "row_id": row["row_id"]},
            )
            continue
        if row_state["status"] == "running":
            row_state["status"] = "interrupted"
            row_state["interrupted_at"] = _utc_now()
        if cost_cap is not None and observed_cost >= cost_cap:
            state["status"] = "budget_stop"
            state["budget_stop"] = {
                "observed_cost_usd": str(observed_cost),
                "cap_usd": str(cost_cap),
            }
            break

        while True:
            invocation_number = len(row_state["invocations"]) + 1
            stdout_path = run_dir / "logs" / f"{row['row_id']}-{invocation_number}.out"
            stderr_path = run_dir / "logs" / f"{row['row_id']}-{invocation_number}.err"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "aipycraft_k8s.cli",
                "--config",
                row["config"],
                "run",
                "--task",
                row["task_id"],
                "--initial-candidate",
                row["candidate_record"],
                "--confirm-paid-calls",
                PAID_CONFIRMATION,
            ]
            started = time.monotonic()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                duration_ms = round((time.monotonic() - started) * 1000)
                stderr_path.write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
                invocation = {
                    "invocation": invocation_number,
                    "finished_at": _utc_now(),
                    "duration_ms": duration_ms,
                    "exit_code": None,
                    "pipeline_status": "infrastructure_error",
                    "run_id": None,
                    "run_dir": None,
                    "cost_usd": "0",
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "scheduler_error": f"{type(exc).__name__}: {exc}",
                }
                row_state["invocations"].append(invocation)
                row_state["last_pipeline_status"] = "infrastructure_error"
                _append_event(
                    events_path,
                    {
                        "event": "row_launch_failed",
                        "row_id": row["row_id"],
                        **invocation,
                    },
                )
                if row_state["infrastructure_reruns"] < max_reruns:
                    row_state["infrastructure_reruns"] += 1
                    row_state["status"] = "pending_infrastructure_rerun"
                    _write_json_atomic(state_path, state)
                    continue
                row_state["status"] = "infrastructure_error"
                _write_json_atomic(state_path, state)
                break
            row_state.update(
                {
                    "status": "running",
                    "child_pid": process.pid,
                    "started_at": _utc_now(),
                }
            )
            state["updated_at"] = _utc_now()
            _write_json_atomic(state_path, state)
            _append_event(
                events_path,
                {
                    "event": "row_started",
                    "row_id": row["row_id"],
                    "invocation": invocation_number,
                    "child_pid": process.pid,
                },
            )
            stdout, stderr = process.communicate()
            duration_ms = round((time.monotonic() - started) * 1000)
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            result = _public_result(stdout)
            status = (
                str(result.get("status"))
                if result is not None
                else "infrastructure_error"
            )
            cost = Decimal(
                str((result or {}).get("usage_totals", {}).get("cost_usd", "0"))
            )
            observed_cost += cost
            observed_treatment_cost += cost
            invocation = {
                "invocation": invocation_number,
                "finished_at": _utc_now(),
                "duration_ms": duration_ms,
                "exit_code": process.returncode,
                "pipeline_status": status,
                "run_id": (result or {}).get("run_id"),
                "run_dir": (result or {}).get("run_dir"),
                "cost_usd": str(cost),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            row_state["invocations"].append(invocation)
            row_state.pop("child_pid", None)
            row_state["finished_at"] = _utc_now()
            row_state["last_exit_code"] = process.returncode
            row_state["last_pipeline_status"] = status
            _append_event(
                events_path,
                {"event": "row_finished", "row_id": row["row_id"], **invocation},
            )

            if status == "infrastructure_error" and row_state[
                "infrastructure_reruns"
            ] < max_reruns and (cost_cap is None or observed_cost < cost_cap):
                row_state["infrastructure_reruns"] += 1
                row_state["status"] = "pending_infrastructure_rerun"
                state["updated_at"] = _utc_now()
                _write_json_atomic(state_path, state)
                continue
            row_state["status"] = status
            state["updated_at"] = _utc_now()
            _write_json_atomic(state_path, state)
            break

    statuses = [item["status"] for item in state["rows"].values()]
    if state.get("status") != "budget_stop":
        state["status"] = (
            "completed"
            if statuses
            and all(
                status
                not in {
                    "pending",
                    "running",
                    "interrupted",
                    "pending_infrastructure_rerun",
                }
                for status in statuses
            )
            else "incomplete"
        )
    state["shared_initial_generation_cost_usd"] = str(shared_initial_cost)
    state["observed_treatment_cost_usd"] = str(observed_treatment_cost)
    state["observed_total_cost_usd"] = str(observed_cost)
    state["updated_at"] = _utc_now()
    _write_json_atomic(state_path, state)
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired, resumable AIPyCraft Kubernetes experiment runner"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    candidate = sub.add_parser("candidate", help="Generate one immutable candidate")
    candidate.add_argument("--config", type=Path, required=True)
    candidate.add_argument("--task", required=True)
    candidate.add_argument("--output-root", type=Path, default=DEFAULT_BANK_ROOT)
    candidate.add_argument("--confirm-paid-calls", required=True)

    manifest = sub.add_parser("manifest", help="Create a paired study manifest")
    manifest.add_argument("--name", required=True)
    manifest.add_argument("--candidate", type=Path, action="append", required=True)
    manifest.add_argument("--treatment-config", type=Path, action="append", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--max-infrastructure-reruns", type=int, default=1)
    manifest.add_argument("--max-study-cost-usd", type=Decimal)

    run = sub.add_parser("run", help="Run or resume all pending manifest rows")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--row")
    run.add_argument("--confirm-paid-calls", required=True)

    status = sub.add_parser("status", help="Read scheduler state without monitoring")
    status.add_argument("--manifest", type=Path, required=True)

    paired = sub.add_parser(
        "paired", help="Generate candidates, build a manifest, then run every arm"
    )
    paired.add_argument("--name", required=True)
    paired.add_argument("--generator-config", type=Path, required=True)
    paired.add_argument("--treatment-config", type=Path, action="append", required=True)
    paired.add_argument("--task", action="append", required=True)
    paired.add_argument("--replicates", type=int, default=1)
    paired.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    paired.add_argument("--candidate-root", type=Path, default=DEFAULT_BANK_ROOT)
    paired.add_argument("--max-infrastructure-reruns", type=int, default=1)
    paired.add_argument("--max-study-cost-usd", type=Decimal)
    paired.add_argument("--confirm-paid-calls", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "candidate":
            _confirm(args.confirm_paid_calls)
            path = generate_candidate(
                config_path=args.config,
                task_id=args.task,
                output_root=args.output_root,
            )
            print(json.dumps({"status": "completed", "candidate": str(path)}, indent=2))
            return 0
        if args.command == "manifest":
            if args.max_infrastructure_reruns < 0:
                raise StudyError("--max-infrastructure-reruns cannot be negative")
            value = create_manifest(
                name=args.name,
                candidate_paths=args.candidate,
                config_paths=args.treatment_config,
                output_path=args.output,
                max_infrastructure_reruns=args.max_infrastructure_reruns,
                max_study_cost_usd=args.max_study_cost_usd,
            )
            print(json.dumps(value, indent=2))
            return 0
        if args.command == "status":
            state = args.manifest.resolve().parent / "state.json"
            print(json.dumps(_read_json(state), indent=2))
            return 0
        if args.command == "run":
            _confirm(args.confirm_paid_calls)
            value = run_manifest(args.manifest, only_row=args.row)
            print(json.dumps(value, indent=2))
            return 0 if value["status"] == "completed" else 2

        _confirm(args.confirm_paid_calls)
        if args.replicates < 1:
            raise StudyError("--replicates must be positive")
        unknown = sorted(set(args.task) - set(load_tasks()))
        if unknown:
            raise StudyError(f"Unknown tasks: {unknown}")
        study_dir = args.study_root.resolve() / args.name
        study_dir.mkdir(parents=True, exist_ok=True)
        if (study_dir / "manifest.json").exists() or (study_dir / "state.json").exists():
            raise StudyError(
                "Study name already has a manifest/state. Resume it with the run "
                "command instead of generating new candidates."
            )
        generator_config = load_config(args.generator_config)
        preflight_started = time.monotonic()
        try:
            preflight = EnvironmentPreparer(
                generator_config.environment,
                timeout_seconds=max(
                    generator_config.pipeline.command_timeout_seconds, 900
                ),
            ).doctor()
        except Exception as exc:
            _write_json_atomic(
                study_dir / "study_preflight.json",
                {
                    "schema_version": 1,
                    "checked_at": _utc_now(),
                    "duration_ms": round(
                        (time.monotonic() - preflight_started) * 1000
                    ),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "paid_calls_made_before_this_check": False,
                },
            )
            raise
        _write_json_atomic(
            study_dir / "study_preflight.json",
            {
                "schema_version": 1,
                "checked_at": _utc_now(),
                "duration_ms": round((time.monotonic() - preflight_started) * 1000),
                "status": "completed",
                "result": preflight,
                "paid_calls_made_before_this_check": False,
            },
        )
        candidate_paths: list[Path] = []
        shared_cost = Decimal("0")
        for task_id in args.task:
            for _ in range(args.replicates):
                candidate_path = generate_candidate(
                    config_path=args.generator_config,
                    task_id=task_id,
                    output_root=args.candidate_root,
                )
                candidate_paths.append(candidate_path)
                candidate_record = _read_json(candidate_path)
                shared_cost += Decimal(
                    str(
                        (candidate_record.get("generation", {}) or {}).get(
                            "cost_usd", "0"
                        )
                    )
                )
                if (
                    args.max_study_cost_usd is not None
                    and shared_cost >= args.max_study_cost_usd
                ):
                    raise StudyError(
                        "Study cost ceiling was reached during candidate-bank "
                        f"generation ({shared_cost} USD observed). Existing immutable "
                        "candidate entries were retained; no treatments were started."
                    )
        manifest_path = study_dir / "manifest.json"
        create_manifest(
            name=args.name,
            candidate_paths=candidate_paths,
            config_paths=args.treatment_config,
            output_path=manifest_path,
            max_infrastructure_reruns=args.max_infrastructure_reruns,
            max_study_cost_usd=args.max_study_cost_usd,
        )
        value = run_manifest(manifest_path)
        print(json.dumps(value, indent=2))
        return 0 if value["status"] == "completed" else 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
