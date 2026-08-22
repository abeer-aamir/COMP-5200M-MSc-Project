from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import AppConfig
from .openrouter import TextGenerationClient
from .tasks import BenchmarkTask


CANDIDATE_BANK_SCHEMA_VERSION = 1


class CandidateBankError(ValueError):
    pass


def _hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def initial_user_prompt(task: BenchmarkTask) -> str:
    return (
        "<public_task>\n"
        f"{task.description.rstrip()}\n"
        "</public_task>\n"
    )


@dataclass(frozen=True)
class CandidateBankEntry:
    record_path: Path
    candidate_id: str
    task_id: str
    raw_text: str
    finish_reason: str | None
    raw_response_sha256: str
    metadata: dict[str, Any]
    source: str = "candidate_bank"

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_BANK_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "record_path": str(self.record_path),
            "record_sha256": _hash_file(self.record_path),
            "raw_response_sha256": self.raw_response_sha256,
            "task_id": self.task_id,
            "source": self.source,
            "shared_initial_generation_usage": self.metadata.get("generation"),
        }


def create_entry(
    *,
    task: BenchmarkTask,
    config: AppConfig,
    client: TextGenerationClient,
    output_root: Path,
) -> CandidateBankEntry:
    system_prompt = config.prompts.generate.read_text(encoding="utf-8")
    user_prompt = initial_user_prompt(task)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    result = client.complete(system_prompt, user_prompt)
    duration_ms = round((time.monotonic() - started) * 1000)
    candidate_id = (
        f"{task.task_id}-{config.api.model.replace('/', '_').replace(':', '_')}-"
        f"{_hash_text(result.raw_text)[:12]}-{uuid.uuid4().hex[:8]}"
    )
    entry_dir = output_root.resolve() / candidate_id
    try:
        entry_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise CandidateBankError(
            f"Candidate entry already exists and is immutable: {entry_dir}"
        ) from exc

    raw_path = entry_dir / "raw_response.txt"
    raw_path.write_text(result.raw_text, encoding="utf-8")
    record = {
        "schema_version": CANDIDATE_BANK_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        # Retained for schema-1 readers that used created_at as the request start.
        "created_at": started_at,
        "duration_ms": duration_ms,
        "task_id": task.task_id,
        "task_description_sha256": _hash_text(task.description),
        "model": config.api.model,
        "provider_only": list(config.api.provider_only),
        "allow_fallbacks": config.api.allow_fallbacks,
        "temperature": config.api.temperature,
        "reasoning_effort": config.api.reasoning_effort,
        "transport_retries": config.api.transport_retries,
        "input_usd_per_million": str(config.api.input_usd_per_million),
        "output_usd_per_million": str(config.api.output_usd_per_million),
        "api_base": config.api.base_url,
        "generate_prompt": str(config.prompts.generate),
        "generate_prompt_sha256": _hash_text(system_prompt),
        "initial_user_prompt_sha256": _hash_text(user_prompt),
        "raw_response": "raw_response.txt",
        "raw_response_sha256": _hash_text(result.raw_text),
        "finish_reason": result.finish_reason,
        "generation": result.audit_dict(),
        "accounting_policy": (
            "Shared initial-generation usage is recorded once in this entry and "
            "excluded from treatment-run totals. Add it once per paired unit, or "
            "allocate it equally to arms, according to the preregistered analysis."
        ),
    }
    record_path = entry_dir / "candidate.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return load_entry(record_path, task=task, config=config)


def _project_or_absolute(path: Path, base: Path) -> Path:
    if path.is_absolute():
        raise CandidateBankError("Banked raw response path must be record-relative")
    candidate = (base / path).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise CandidateBankError("Banked raw response path escapes its entry") from exc
    return candidate


def load_entry(
    path: Path,
    *,
    task: BenchmarkTask,
    config: AppConfig,
) -> CandidateBankEntry:
    record_path = path.resolve()
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateBankError(f"Could not read candidate entry {path}: {exc}") from exc
    required = {
        "schema_version",
        "candidate_id",
        "created_at",
        "duration_ms",
        "task_id",
        "task_description_sha256",
        "model",
        "provider_only",
        "allow_fallbacks",
        "temperature",
        "reasoning_effort",
        "transport_retries",
        "input_usd_per_million",
        "output_usd_per_million",
        "api_base",
        "generate_prompt",
        "generate_prompt_sha256",
        "initial_user_prompt_sha256",
        "raw_response",
        "raw_response_sha256",
        "finish_reason",
        "generation",
        "accounting_policy",
    }
    optional = {"started_at", "finished_at"}
    if (
        not isinstance(record, dict)
        or not required.issubset(record)
        or not set(record).issubset(required | optional)
    ):
        raise CandidateBankError("Malformed or unsupported candidate-bank record")
    if record["schema_version"] != CANDIDATE_BANK_SCHEMA_VERSION:
        raise CandidateBankError("Unsupported candidate-bank schema version")
    if record["task_id"] != task.task_id:
        raise CandidateBankError("Candidate entry belongs to a different task")
    if record["task_description_sha256"] != _hash_text(task.description):
        raise CandidateBankError("Task description changed after candidate generation")
    if record["model"] != config.api.model:
        raise CandidateBankError("Candidate model differs from treatment model")
    if record["provider_only"] != list(config.api.provider_only):
        raise CandidateBankError("Candidate provider policy differs from treatment")
    if record["allow_fallbacks"] != config.api.allow_fallbacks:
        raise CandidateBankError("Candidate fallback policy differs from treatment")
    if record["temperature"] != config.api.temperature:
        raise CandidateBankError("Candidate temperature differs from treatment")
    if record["reasoning_effort"] != config.api.reasoning_effort:
        raise CandidateBankError("Candidate reasoning effort differs from treatment")
    if record["transport_retries"] != config.api.transport_retries:
        raise CandidateBankError("Candidate transport retries differ from treatment")
    if record["input_usd_per_million"] != str(config.api.input_usd_per_million):
        raise CandidateBankError("Candidate input price snapshot differs from treatment")
    if record["output_usd_per_million"] != str(config.api.output_usd_per_million):
        raise CandidateBankError("Candidate output price snapshot differs from treatment")
    if record["api_base"] != config.api.base_url:
        raise CandidateBankError("Candidate API endpoint differs from treatment")

    system_prompt = config.prompts.generate.read_text(encoding="utf-8")
    if record["generate_prompt_sha256"] != _hash_text(system_prompt):
        raise CandidateBankError("Generation prompt changed after candidate generation")
    if record["initial_user_prompt_sha256"] != _hash_text(initial_user_prompt(task)):
        raise CandidateBankError("Initial user prompt changed after candidate generation")

    raw_path = _project_or_absolute(Path(str(record["raw_response"])), record_path.parent)
    try:
        raw_text = raw_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CandidateBankError(f"Could not read banked raw response: {exc}") from exc
    raw_hash = _hash_text(raw_text)
    if raw_hash != record["raw_response_sha256"]:
        raise CandidateBankError("Banked raw response failed its integrity check")
    if not isinstance(record["candidate_id"], str) or not record["candidate_id"]:
        raise CandidateBankError("Candidate ID is invalid")
    if record["finish_reason"] is not None and not isinstance(
        record["finish_reason"], str
    ):
        raise CandidateBankError("Candidate finish_reason is invalid")
    if not isinstance(record["generation"], dict):
        raise CandidateBankError("Candidate generation audit is invalid")
    return CandidateBankEntry(
        record_path=record_path,
        candidate_id=record["candidate_id"],
        task_id=record["task_id"],
        raw_text=raw_text,
        finish_reason=record["finish_reason"],
        raw_response_sha256=raw_hash,
        metadata=record,
    )


def load_saved_yaml(path: Path, *, task: BenchmarkTask) -> CandidateBankEntry:
    """Load a user-supplied saved response without claiming bank provenance."""

    candidate_path = path.resolve()
    try:
        size_bytes = candidate_path.stat().st_size
        if size_bytes > 10 * 1024 * 1024:
            raise CandidateBankError("Saved initial YAML exceeds the 10 MiB safety limit")
        raw_text = candidate_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CandidateBankError(f"Could not read saved initial YAML {path}: {exc}") from exc
    if not raw_text.strip():
        raise CandidateBankError("Saved initial YAML is empty")
    raw_hash = _hash_text(raw_text)
    return CandidateBankEntry(
        record_path=candidate_path,
        candidate_id=f"saved-{task.task_id}-{raw_hash[:12]}",
        task_id=task.task_id,
        raw_text=raw_text,
        finish_reason="stop",
        raw_response_sha256=raw_hash,
        metadata={
            "schema_version": 1,
            "source": "saved_yaml",
            "task_id": task.task_id,
            "task_description_sha256": _hash_text(task.description),
            "raw_response_sha256": raw_hash,
            "size_bytes": size_bytes,
            "generation": None,
            "accounting_policy": (
                "Generation usage is unknown and excluded from treatment totals; "
                "use an immutable candidate-bank record for final paired studies."
            ),
        },
        source="saved_yaml",
    )


def load_initial_candidate(
    path: Path,
    *,
    task: BenchmarkTask,
    config: AppConfig,
) -> CandidateBankEntry:
    """Accept an immutable bank record or a directly saved YAML/text response."""

    if path.suffix.lower() == ".json":
        return load_entry(path, task=task, config=config)
    return load_saved_yaml(path, task=task)
