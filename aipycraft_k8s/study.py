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
MAX_INITIAL_GENERATION_ATTEMPTS = 2


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


def _failed_call_cost(exc: BaseException) -> Decimal:
    """Return provider-reported cost attached to a failed generation call."""

    audit_result = getattr(exc, "audit_result", None)
    if audit_result is None:
        return Decimal("0")
    return Decimal(str(getattr(audit_result, "cost_usd", "0") or "0"))


def _generation_failure_record(exc: BaseException, attempt: int) -> dict[str, Any]:
    audit_result = getattr(exc, "audit_result", None)
    audit = audit_result.audit_dict() if audit_result is not None else None
    return {
        "attempt": attempt,
        "failed_at": _utc_now(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "cost_usd": str(_failed_call_cost(exc)),
        "transport_attempts": getattr(exc, "transport_attempts", None),
        "unknown_cost_attempts": getattr(exc, "unknown_cost_attempts", None),
        "finish_reason": (audit or {}).get("finish_reason"),
        "generation_id": (audit or {}).get("generation_id"),
        "request_id": (audit or {}).get("request_id"),
    }


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
                "transport_attempt_log": list(
                    getattr(exc, "transport_attempt_log", ())
                ),
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
    charge_candidate_generation_cost: bool = True,
    failed_generation_cost_usd: Decimal = Decimal("0"),
    skipped_tasks: list[dict[str, Any]] | None = None,
    candidate_source_metadata: dict[str, Any] | None = None,
    allow_task_description_drift: bool = False,
    write: bool = True,
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
    candidate_generation_cost = Decimal("0")
    for candidate_index, candidate_path in enumerate(candidate_paths):
        raw = _read_json(candidate_path.resolve())
        task_id = str(raw.get("task_id", ""))
        task = load_task(task_id)
        generation = raw.get("generation", {}) or {}
        candidate_generation_cost += Decimal(str(generation.get("cost_usd", "0")))
        rotation = candidate_index % len(config_paths)
        ordered_configs = loaded_configs[rotation:] + loaded_configs[:rotation]
        for config in ordered_configs:
            entry = load_entry(
                candidate_path,
                task=task,
                config=config,
                allow_task_description_drift=allow_task_description_drift,
            )
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
                    "candidate_task_description_drift_allowed": (
                        allow_task_description_drift
                    ),
                    "candidate_task_description_sha256": raw.get(
                        "task_description_sha256"
                    ),
                    "current_task_description_sha256": hashlib.sha256(
                        task.description.encode("utf-8")
                    ).hexdigest(),
                }
            )
    shared_initial_cost = (
        candidate_generation_cost
        if charge_candidate_generation_cost
        else Decimal("0")
    ) + failed_generation_cost_usd
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
        "initial_generation_accounting": {
            "candidate_cost_incurred_in_this_study": charge_candidate_generation_cost,
            "successful_candidate_cost_usd": (
                str(candidate_generation_cost)
                if charge_candidate_generation_cost
                else "0"
            ),
            "failed_candidate_call_cost_usd": str(failed_generation_cost_usd),
            "reused_candidate_historical_cost_usd": (
                "0"
                if charge_candidate_generation_cost
                else str(candidate_generation_cost)
            ),
            "maximum_attempts_per_initial_candidate": MAX_INITIAL_GENERATION_ATTEMPTS,
        },
        "skipped_initial_candidates": skipped_tasks or [],
        "candidate_source_metadata": candidate_source_metadata,
        "candidate_task_description_drift_allowed": allow_task_description_drift,
        "rows": rows,
    }
    if write:
        _write_json_atomic(output_path.resolve(), manifest)
    return manifest


def recover_paired_study(
    plan_path: Path, *, plan_only: bool = False
) -> dict[str, Any]:
    """Resume interrupted candidate banking, then run the original paired study."""

    plan_path = plan_path.resolve()
    plan = _read_json(plan_path)
    required = {
        "schema_version",
        "study_name",
        "generator_config",
        "treatment_configs",
        "tasks",
        "existing_candidates",
        "study_root",
        "candidate_root",
        "max_infrastructure_reruns",
        "max_study_cost_usd",
    }
    if set(plan) != required or plan.get("schema_version") != 1:
        raise StudyError(
            "Recovery plan must be schema 1 with exactly the documented fields"
        )

    def project_path(raw: Any, field: str) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise StudyError(f"Recovery plan {field} must be a non-empty path")
        value = Path(raw)
        return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()

    study_name = str(plan["study_name"])
    tasks = [str(item) for item in plan["tasks"]]
    if not tasks or len(tasks) != len(set(tasks)):
        raise StudyError("Recovery plan tasks must be non-empty and unique")
    unknown = sorted(set(tasks) - set(load_tasks()))
    if unknown:
        raise StudyError(f"Recovery plan contains unknown tasks: {unknown}")
    generator_path = project_path(plan["generator_config"], "generator_config")
    treatment_paths = [
        project_path(item, "treatment_configs")
        for item in plan["treatment_configs"]
    ]
    if not treatment_paths:
        raise StudyError("Recovery plan requires treatment configs")
    study_root = project_path(plan["study_root"], "study_root")
    candidate_root = project_path(plan["candidate_root"], "candidate_root")
    study_dir = study_root / study_name
    manifest_path = study_dir / "manifest.json"
    scheduler_state_path = study_dir / "state.json"
    recovery_state_path = study_dir / "candidate_generation_recovery.json"
    if scheduler_state_path.exists():
        raise StudyError(
            "Study scheduler state already exists; use the ordinary run command"
        )
    if manifest_path.exists() and plan_only:
        return {
            "status": "planned",
            "paid_calls_made": False,
            "manifest_exists": True,
            "manifest": str(manifest_path),
        }
    if manifest_path.exists():
        return run_manifest(manifest_path)

    max_reruns = int(plan["max_infrastructure_reruns"])
    if max_reruns < 0:
        raise StudyError("Recovery max_infrastructure_reruns cannot be negative")
    raw_cap = plan["max_study_cost_usd"]
    cost_cap = None if raw_cap is None else Decimal(str(raw_cap))
    if cost_cap is not None and cost_cap <= 0:
        raise StudyError("Recovery max_study_cost_usd must be positive or null")

    config = load_config(generator_path)
    plan_sha256 = _sha256_file(plan_path)
    candidate_by_task: dict[str, Path] = {}
    recovery_checkpoint: dict[str, Any] = {}
    source_paths = [
        project_path(item, "existing_candidates")
        for item in plan["existing_candidates"]
    ]
    if recovery_state_path.exists():
        recovery_checkpoint = _read_json(recovery_state_path)
        if recovery_checkpoint.get("plan_sha256") != plan_sha256:
            raise StudyError("Recovery plan changed after checkpoint creation")
        source_paths.extend(
            Path(item) for item in recovery_checkpoint.get("candidates", [])
        )

    for candidate_path in source_paths:
        candidate_path = candidate_path.resolve()
        raw = _read_json(candidate_path)
        task_id = str(raw.get("task_id", ""))
        if task_id not in tasks:
            raise StudyError(
                f"Recovery candidate {candidate_path} belongs to unexpected task {task_id}"
            )
        previous = candidate_by_task.get(task_id)
        if previous is not None and previous != candidate_path:
            raise StudyError(f"Multiple recovery candidates were supplied for {task_id}")
        load_entry(candidate_path, task=load_task(task_id), config=config)
        candidate_by_task[task_id] = candidate_path

    initial_existing_tasks = set(candidate_by_task)
    generation_attempts: dict[str, list[dict[str, Any]]] = {
        str(task_id): list(records)
        for task_id, records in (
            recovery_checkpoint.get("generation_attempts", {}) or {}
        ).items()
        if isinstance(records, list)
    }
    skipped_task_ids = {
        str(item.get("task_id"))
        for item in recovery_checkpoint.get("skipped_initial_candidates", [])
        if isinstance(item, dict) and item.get("task_id")
    }
    failed_generation_cost = Decimal(
        str(recovery_checkpoint.get("failed_generation_cost_usd", "0"))
    )

    if plan_only:
        return {
            "status": "planned",
            "paid_calls_made": False,
            "study_name": study_name,
            "generator_model": config.api.model,
            "generator_reasoning": config.api.reasoning_effort,
            "existing_tasks": [
                task_id for task_id in tasks if task_id in candidate_by_task
            ],
            "missing_tasks": [
                task_id
                for task_id in tasks
                if task_id not in candidate_by_task and task_id not in skipped_task_ids
            ],
            "skipped_tasks": [
                task_id for task_id in tasks if task_id in skipped_task_ids
            ],
            "existing_candidates": len(candidate_by_task),
            "max_study_cost_usd": (
                None if cost_cap is None else str(cost_cap)
            ),
        }

    def successful_candidate_cost() -> Decimal:
        return sum(
            (
                Decimal(
                    str(
                        (_read_json(path).get("generation", {}) or {}).get(
                            "cost_usd", "0"
                        )
                    )
                )
                for path in candidate_by_task.values()
            ),
            Decimal("0"),
        )

    def observed_candidate_cost() -> Decimal:
        return successful_candidate_cost() + failed_generation_cost

    def skipped_records() -> list[dict[str, Any]]:
        return [
            {
                "task_id": task_id,
                "status": "initial_candidate_skipped",
                "attempts": generation_attempts.get(task_id, []),
            }
            for task_id in tasks
            if task_id in skipped_task_ids
        ]

    def save_recovery(status: str, **extra: Any) -> None:
        _write_json_atomic(
            recovery_state_path,
            {
                "schema_version": 1,
                "study_name": study_name,
                "plan": str(plan_path),
                "plan_sha256": plan_sha256,
                "status": status,
                "updated_at": _utc_now(),
                "candidates": [
                    str(candidate_by_task[task_id])
                    for task_id in tasks
                    if task_id in candidate_by_task
                ],
                "completed_tasks": [
                    task_id for task_id in tasks if task_id in candidate_by_task
                ],
                "remaining_tasks": [
                    task_id
                    for task_id in tasks
                    if task_id not in candidate_by_task
                    and task_id not in skipped_task_ids
                ],
                "skipped_initial_candidates": skipped_records(),
                "generation_attempts": generation_attempts,
                "maximum_attempts_per_initial_candidate": (
                    MAX_INITIAL_GENERATION_ATTEMPTS
                ),
                "successful_candidate_cost_usd": str(successful_candidate_cost()),
                "failed_generation_cost_usd": str(failed_generation_cost),
                "observed_candidate_cost_usd": str(observed_candidate_cost()),
                **extra,
            },
        )

    save_recovery("preflight")
    preflight_started = time.monotonic()
    try:
        preflight = EnvironmentPreparer(
            config.environment,
            timeout_seconds=max(config.pipeline.command_timeout_seconds, 900),
        ).doctor()
    except Exception as exc:
        save_recovery(
            "preflight_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            preflight_duration_ms=round(
                (time.monotonic() - preflight_started) * 1000
            ),
        )
        raise
    save_recovery(
        "generating",
        preflight_status=preflight.get("status"),
        preflight_duration_ms=round((time.monotonic() - preflight_started) * 1000),
    )

    for task_id in tasks:
        if task_id in candidate_by_task or task_id in skipped_task_ids:
            continue
        task_attempts = generation_attempts.setdefault(task_id, [])
        for attempt in range(
            len(task_attempts) + 1, MAX_INITIAL_GENERATION_ATTEMPTS + 1
        ):
            if cost_cap is not None and observed_candidate_cost() >= cost_cap:
                save_recovery("budget_stop", cap_usd=str(cost_cap))
                raise StudyError(
                    "Recovery candidate-generation cost ceiling was reached"
                )
            try:
                candidate_path = generate_candidate(
                    config_path=generator_path,
                    task_id=task_id,
                    output_root=candidate_root,
                )
            except Exception as exc:
                failure = _generation_failure_record(exc, attempt)
                task_attempts.append(failure)
                failed_generation_cost += Decimal(failure["cost_usd"])
                _append_event(
                    candidate_root.resolve() / "generation_events.jsonl",
                    {
                        "event": (
                            "candidate_generation_retry_scheduled"
                            if attempt < MAX_INITIAL_GENERATION_ATTEMPTS
                            else "initial_candidate_skipped"
                        ),
                        "task_id": task_id,
                        "attempt": attempt,
                        "maximum_attempts": MAX_INITIAL_GENERATION_ATTEMPTS,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failed_call_cost_usd": failure["cost_usd"],
                    },
                )
                if attempt < MAX_INITIAL_GENERATION_ATTEMPTS:
                    save_recovery("generating", retrying_task=task_id)
                    continue
                skipped_task_ids.add(task_id)
                save_recovery("generating", most_recent_skip=task_id)
                break
            candidate_by_task[task_id] = candidate_path.resolve()
            save_recovery("generating")
            break

    ordered_candidates = [
        candidate_by_task[task_id] for task_id in tasks if task_id in candidate_by_task
    ]
    if not ordered_candidates:
        save_recovery("no_candidates")
        raise StudyError("Every initial candidate was skipped after two failed attempts")
    create_manifest(
        name=study_name,
        candidate_paths=ordered_candidates,
        config_paths=treatment_paths,
        output_path=manifest_path,
        max_infrastructure_reruns=max_reruns,
        max_study_cost_usd=cost_cap,
        failed_generation_cost_usd=failed_generation_cost,
        skipped_tasks=skipped_records(),
        candidate_source_metadata={
            "mode": "checkpoint_recovery",
            "plan": str(plan_path),
            "plan_sha256": plan_sha256,
            "preexisting_tasks": sorted(initial_existing_tasks),
        },
    )
    save_recovery("manifest_created", manifest=str(manifest_path))
    return run_manifest(manifest_path)


def reuse_saved_candidates(
    *,
    name: str,
    source_manifest_paths: list[Path],
    config_paths: list[Path],
    tasks: list[str],
    study_root: Path,
    max_infrastructure_reruns: int,
    max_study_cost_usd: Decimal | None,
    allow_task_description_drift: bool = False,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Run validator-only treatments from immutable, previously generated YAML."""

    if not source_manifest_paths:
        raise StudyError("At least one source manifest is required")
    if not config_paths:
        raise StudyError("At least one treatment config is required")
    if not tasks or len(tasks) != len(set(tasks)):
        raise StudyError("Reused-candidate tasks must be non-empty and unique")
    unknown = sorted(set(tasks) - set(load_tasks()))
    if unknown:
        raise StudyError(f"Unknown tasks: {unknown}")
    if max_infrastructure_reruns < 0:
        raise StudyError("--max-infrastructure-reruns cannot be negative")
    if max_study_cost_usd is not None and max_study_cost_usd <= 0:
        raise StudyError("--max-study-cost-usd must be positive or omitted")

    source_metadata: list[dict[str, Any]] = []
    candidates: dict[str, tuple[Path, str]] = {}
    for source_path in source_manifest_paths:
        source_path = source_path.resolve()
        source = _read_json(source_path)
        rows = source.get("rows")
        if not isinstance(rows, list):
            raise StudyError(f"Source manifest has no rows: {source_path}")
        source_metadata.append(
            {
                "manifest": str(source_path),
                "manifest_sha256": _sha256_file(source_path),
                "study_name": source.get("study_name"),
            }
        )
        for row in rows:
            if not isinstance(row, dict):
                continue
            task_id = str(row.get("task_id", ""))
            if task_id not in tasks:
                continue
            raw_candidate_path = Path(str(row.get("candidate_record", "")))
            candidate_path = (
                raw_candidate_path
                if raw_candidate_path.is_absolute()
                else PROJECT_ROOT / raw_candidate_path
            ).resolve()
            expected_sha256 = str(row.get("candidate_record_sha256", ""))
            if not candidate_path.is_file():
                raise StudyError(
                    f"Saved candidate is missing for {task_id}: {candidate_path}"
                )
            actual_sha256 = _sha256_file(candidate_path)
            if not expected_sha256 or actual_sha256 != expected_sha256:
                raise StudyError(
                    f"Saved candidate record integrity failed for {task_id}: "
                    f"{candidate_path}"
                )
            previous = candidates.get(task_id)
            if previous is not None and previous != (candidate_path, actual_sha256):
                raise StudyError(
                    f"Source manifests contain conflicting candidates for {task_id}"
                )
            candidates[task_id] = (candidate_path, actual_sha256)

    missing = [task_id for task_id in tasks if task_id not in candidates]
    if missing:
        raise StudyError(
            "Saved candidate preflight failed before paid calls; missing tasks: "
            + ", ".join(missing)
        )
    candidate_paths = [candidates[task_id][0] for task_id in tasks]
    study_dir = study_root.resolve() / name
    manifest_path = study_dir / "manifest.json"
    state_path = study_dir / "state.json"
    if manifest_path.exists() or state_path.exists():
        raise StudyError(
            "Study name already has a manifest/state. Resume it with the run "
            "command instead of creating another reused-candidate study."
        )

    candidate_source_metadata = {
        "mode": "reused_saved_initial_candidates",
        "historical_generation_cost_excluded_from_this_study": True,
        "task_description_drift_allowed": allow_task_description_drift,
        "source_manifests": source_metadata,
        "selected_candidates": [
            {
                "task_id": task_id,
                "candidate_record": str(candidates[task_id][0]),
                "candidate_record_sha256": candidates[task_id][1],
                "candidate_task_description_sha256": _read_json(
                    candidates[task_id][0]
                ).get("task_description_sha256"),
                "current_task_description_sha256": hashlib.sha256(
                    load_task(task_id).description.encode("utf-8")
                ).hexdigest(),
            }
            for task_id in tasks
        ],
    }
    preview = create_manifest(
        name=name,
        candidate_paths=candidate_paths,
        config_paths=config_paths,
        output_path=manifest_path,
        max_infrastructure_reruns=max_infrastructure_reruns,
        max_study_cost_usd=max_study_cost_usd,
        charge_candidate_generation_cost=False,
        candidate_source_metadata=candidate_source_metadata,
        allow_task_description_drift=allow_task_description_drift,
        write=False,
    )
    if plan_only:
        return {
            "status": "planned",
            "paid_calls_made": False,
            "study_name": name,
            "tasks": tasks,
            "candidate_count": len(candidate_paths),
            "treatments": preview["treatments"],
            "row_count": len(preview["rows"]),
            "source_manifests": source_metadata,
            "historical_candidate_cost_usd": preview[
                "initial_generation_accounting"
            ]["reused_candidate_historical_cost_usd"],
            "charged_initial_generation_cost_usd": "0",
            "max_study_cost_usd": (
                None
                if max_study_cost_usd is None
                else str(max_study_cost_usd)
            ),
        }

    study_dir.mkdir(parents=True, exist_ok=True)
    first_config = load_config(config_paths[0])
    preflight_started = time.monotonic()
    try:
        preflight = EnvironmentPreparer(
            first_config.environment,
            timeout_seconds=max(first_config.pipeline.command_timeout_seconds, 900),
        ).doctor()
    except Exception as exc:
        _write_json_atomic(
            study_dir / "study_preflight.json",
            {
                "schema_version": 1,
                "checked_at": _utc_now(),
                "duration_ms": round((time.monotonic() - preflight_started) * 1000),
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
    _write_json_atomic(
        study_dir / "candidate_reuse_summary.json",
        {
            "schema_version": 1,
            "created_at": _utc_now(),
            "status": "validated",
            "paid_calls_made_for_candidate_reuse": False,
            **candidate_source_metadata,
        },
    )
    create_manifest(
        name=name,
        candidate_paths=candidate_paths,
        config_paths=config_paths,
        output_path=manifest_path,
        max_infrastructure_reruns=max_infrastructure_reruns,
        max_study_cost_usd=max_study_cost_usd,
        charge_candidate_generation_cost=False,
        candidate_source_metadata=candidate_source_metadata,
        allow_task_description_drift=allow_task_description_drift,
    )
    return run_manifest(manifest_path)


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
                "pending_since": _utc_now(),
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
            if row.get("candidate_task_description_drift_allowed"):
                command.append("--allow-candidate-task-description-drift")
            invocation_started_at = _utc_now()
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
                    "started_at": invocation_started_at,
                    "finished_at": _utc_now(),
                    "duration_ms": duration_ms,
                    "process_launch_duration_ms": duration_ms,
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
            process_launched = time.monotonic()
            row_state.update(
                {
                    "status": "running",
                    "child_pid": process.pid,
                    "last_started_at": _utc_now(),
                }
            )
            row_state.setdefault("first_started_at", row_state["last_started_at"])
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
            process_finished = time.monotonic()
            child_process_duration_ms = round((process_finished - started) * 1000)
            scheduler_postprocess_started = time.monotonic()
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
            scheduler_postprocess_duration_ms = round(
                (time.monotonic() - scheduler_postprocess_started) * 1000
            )
            duration_ms = round((time.monotonic() - started) * 1000)
            invocation = {
                "invocation": invocation_number,
                "started_at": invocation_started_at,
                "finished_at": _utc_now(),
                "duration_ms": duration_ms,
                "process_launch_duration_ms": round(
                    (process_launched - started) * 1000
                ),
                "child_process_duration_ms": child_process_duration_ms,
                "process_execution_duration_ms": round(
                    (process_finished - process_launched) * 1000
                ),
                "scheduler_postprocess_duration_ms": scheduler_postprocess_duration_ms,
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

    recover = sub.add_parser(
        "recover-paired",
        help="Reuse checkpointed candidates, generate only missing tasks, then run",
    )
    recover.add_argument("--plan", type=Path, required=True)
    recover.add_argument("--confirm-paid-calls")
    recover.add_argument("--plan-only", action="store_true")

    reuse = sub.add_parser(
        "reuse",
        help=(
            "Validate and reuse saved initial candidates, then run only the "
            "requested treatment configs"
        ),
    )
    reuse.add_argument("--name", required=True)
    reuse.add_argument("--source-manifest", type=Path, action="append", required=True)
    reuse.add_argument("--treatment-config", type=Path, action="append", required=True)
    reuse.add_argument("--task", action="append", required=True)
    reuse.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    reuse.add_argument("--max-infrastructure-reruns", type=int, default=1)
    reuse.add_argument("--max-study-cost-usd", type=Decimal)
    reuse.add_argument("--confirm-paid-calls")
    reuse.add_argument("--plan-only", action="store_true")
    reuse.add_argument(
        "--allow-task-description-drift",
        action="store_true",
        help=(
            "Allow immutable candidates from an intentional plaintext rewrite; "
            "source and current hashes remain in the manifest"
        ),
    )
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
        if args.command == "recover-paired":
            if not args.plan_only:
                _confirm(args.confirm_paid_calls)
            value = recover_paired_study(args.plan, plan_only=args.plan_only)
            print(json.dumps(value, indent=2))
            return 0 if value["status"] in {"completed", "planned"} else 2
        if args.command == "reuse":
            if not args.plan_only:
                _confirm(args.confirm_paid_calls)
            value = reuse_saved_candidates(
                name=args.name,
                source_manifest_paths=args.source_manifest,
                config_paths=args.treatment_config,
                tasks=args.task,
                study_root=args.study_root,
                max_infrastructure_reruns=args.max_infrastructure_reruns,
                max_study_cost_usd=args.max_study_cost_usd,
                allow_task_description_drift=args.allow_task_description_drift,
                plan_only=args.plan_only,
            )
            print(json.dumps(value, indent=2))
            return 0 if value["status"] in {"completed", "planned"} else 2

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
        failed_generation_cost = Decimal("0")
        generation_attempts: dict[str, list[dict[str, Any]]] = {}
        skipped_candidates: list[dict[str, Any]] = []

        def successful_generation_cost() -> Decimal:
            return sum(
                (
                    Decimal(
                        str(
                            (_read_json(path).get("generation", {}) or {}).get(
                                "cost_usd", "0"
                            )
                        )
                    )
                    for path in candidate_paths
                ),
                Decimal("0"),
            )

        def write_generation_summary(status: str) -> None:
            _write_json_atomic(
                study_dir / "candidate_generation_summary.json",
                {
                    "schema_version": 1,
                    "study_name": args.name,
                    "status": status,
                    "updated_at": _utc_now(),
                    "maximum_attempts_per_initial_candidate": (
                        MAX_INITIAL_GENERATION_ATTEMPTS
                    ),
                    "successful_candidates": [str(path) for path in candidate_paths],
                    "generation_attempts": generation_attempts,
                    "skipped_initial_candidates": skipped_candidates,
                    "successful_candidate_cost_usd": str(
                        successful_generation_cost()
                    ),
                    "failed_generation_cost_usd": str(failed_generation_cost),
                    "observed_candidate_cost_usd": str(
                        successful_generation_cost() + failed_generation_cost
                    ),
                },
            )

        for task_id in args.task:
            for replicate in range(1, args.replicates + 1):
                attempt_key = (
                    task_id
                    if args.replicates == 1
                    else f"{task_id}#replicate-{replicate}"
                )
                task_attempts = generation_attempts.setdefault(attempt_key, [])
                generated = False
                for attempt in range(1, MAX_INITIAL_GENERATION_ATTEMPTS + 1):
                    observed = successful_generation_cost() + failed_generation_cost
                    if (
                        args.max_study_cost_usd is not None
                        and observed >= args.max_study_cost_usd
                    ):
                        write_generation_summary("budget_stop")
                        raise StudyError(
                            "Study cost ceiling was reached during candidate-bank "
                            f"generation ({observed} USD observed). Existing immutable "
                            "candidate entries were retained; no treatments were started."
                        )
                    try:
                        candidate_path = generate_candidate(
                            config_path=args.generator_config,
                            task_id=task_id,
                            output_root=args.candidate_root,
                        )
                    except Exception as exc:
                        failure = _generation_failure_record(exc, attempt)
                        task_attempts.append(failure)
                        failed_generation_cost += Decimal(failure["cost_usd"])
                        _append_event(
                            args.candidate_root.resolve() / "generation_events.jsonl",
                            {
                                "event": (
                                    "candidate_generation_retry_scheduled"
                                    if attempt < MAX_INITIAL_GENERATION_ATTEMPTS
                                    else "initial_candidate_skipped"
                                ),
                                "task_id": task_id,
                                "replicate": replicate,
                                "attempt": attempt,
                                "maximum_attempts": MAX_INITIAL_GENERATION_ATTEMPTS,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "failed_call_cost_usd": failure["cost_usd"],
                            },
                        )
                        write_generation_summary("generating")
                        continue
                    candidate_paths.append(candidate_path)
                    generated = True
                    write_generation_summary("generating")
                    break
                if not generated:
                    skipped_candidates.append(
                        {
                            "task_id": task_id,
                            "replicate": replicate,
                            "status": "initial_candidate_skipped",
                            "attempts": task_attempts,
                        }
                    )
                    write_generation_summary("generating")
        if not candidate_paths:
            write_generation_summary("no_candidates")
            raise StudyError(
                "Every initial candidate was skipped after two failed attempts"
            )
        manifest_path = study_dir / "manifest.json"
        create_manifest(
            name=args.name,
            candidate_paths=candidate_paths,
            config_paths=args.treatment_config,
            output_path=manifest_path,
            max_infrastructure_reruns=args.max_infrastructure_reruns,
            max_study_cost_usd=args.max_study_cost_usd,
            failed_generation_cost_usd=failed_generation_cost,
            skipped_tasks=skipped_candidates,
        )
        write_generation_summary("manifest_created")
        value = run_manifest(manifest_path)
        print(json.dumps(value, indent=2))
        return 0 if value["status"] == "completed" else 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
