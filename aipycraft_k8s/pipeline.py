from __future__ import annotations

import hashlib
import inspect
import json
import platform
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from .analysis import build_analysis_record
from .candidate_bank import CandidateBankEntry
from .candidate_diff import candidate_semantic_diff
from .commands import CommandError, CommandRunner
from .config import AppConfig, PROJECT_ROOT, treatment_id
from .diagnostics import observe_runtime
from .environment import EnvironmentError, IsolatedKindHarness
from .execution_verifier import run_execution_gate
from .openrouter import (
    GenerationResult,
    ProviderError,
    TextGenerationClient,
)
from .post_verifier import run_post_execution_verifier
from .pre_validator import (
    AiValidationError,
    correction_feedback,
    parse_decision,
    validator_user_prompt,
)
from .tasks import BenchmarkTask
from .yaml_check import check_yaml_syntax


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact_integrity(root: Path) -> dict[str, Any]:
    excluded = {"summary.json", "artifact_integrity.json"}
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "scope": "all run artifacts except mutable summaries and this manifest",
        "excluded": sorted(excluded),
        "file_count": len(files),
        "aggregate_sha256": aggregate,
        "files": files,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_REPAIR_FEEDBACK_MAX_BYTES = 48_000


def _compact_feedback(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 12,
    string_limit: int = 4_000,
    list_limit: int = 20,
    dict_limit: int = 50,
    truncation: dict[str, int] | None = None,
) -> Any:
    """Bound untrusted execution evidence before placing it in a model prompt."""

    if truncation is None:
        truncation = {}

    def record(label: str, amount: int = 1) -> None:
        truncation[label] = truncation.get(label, 0) + amount

    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        record("truncated_strings")
        record("omitted_string_characters", len(value) - string_limit)
        return value[:string_limit] + (
            f"\n<... {len(value) - string_limit} characters omitted>"
        )
    if value is None or isinstance(value, (bool, int, float)):
        return value
    # Preserve deep scalar leaves (exit codes, reasons, states, and log text).
    # Only nested collections are replaced at the depth boundary.
    if depth >= max_depth:
        record("omitted_nested_collections")
        return "<nested diagnostic collection omitted>"
    if isinstance(value, list):
        compact = [
            _compact_feedback(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                string_limit=string_limit,
                list_limit=list_limit,
                dict_limit=dict_limit,
                truncation=truncation,
            )
            for item in value[:list_limit]
        ]
        if len(value) > list_limit:
            record("omitted_list_items", len(value) - list_limit)
            compact.append({"omitted_items": len(value) - list_limit})
        return compact
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        compact = {
            str(key): _compact_feedback(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                string_limit=string_limit,
                list_limit=list_limit,
                dict_limit=dict_limit,
                truncation=truncation,
            )
            for key, item in items[:dict_limit]
        }
        if len(items) > dict_limit:
            record("omitted_dict_fields", len(items) - dict_limit)
            compact["omitted_fields"] = len(items) - dict_limit
        return compact
    return str(value)


def _bounded_feedback_json(value: Any) -> tuple[str, dict[str, Any]]:
    """Serialize exact prompt feedback under one whole-payload byte budget."""

    source = json.dumps(value, sort_keys=True, default=str)
    profiles = (
        (4_000, 20, 50, 12),
        (2_000, 12, 30, 11),
        (1_000, 8, 20, 10),
        (500, 6, 15, 9),
        (250, 4, 10, 8),
    )
    source_utf8_bytes = len(source.encode("utf-8"))
    rendered = ""
    profile_used: tuple[int, int, int, int] | None = None
    truncation: dict[str, int] = {}
    for string_limit, list_limit, dict_limit, max_depth in profiles:
        candidate_truncation: dict[str, int] = {}
        compact = _compact_feedback(
            value,
            string_limit=string_limit,
            list_limit=list_limit,
            dict_limit=dict_limit,
            max_depth=max_depth,
            truncation=candidate_truncation,
        )
        rendered = json.dumps(compact, indent=2, sort_keys=True, default=str)
        profile_used = (string_limit, list_limit, dict_limit, max_depth)
        truncation = candidate_truncation
        if len(rendered.encode("utf-8")) <= _REPAIR_FEEDBACK_MAX_BYTES:
            break
    whole_payload_truncated = False
    if len(rendered.encode("utf-8")) > _REPAIR_FEEDBACK_MAX_BYTES:
        whole_payload_truncated = True
        compact_rendered = rendered
        compact_utf8_bytes = len(compact_rendered.encode("utf-8"))

        def envelope(excerpt: str) -> str:
            return json.dumps(
                {
                    "diagnostic_excerpt": excerpt,
                    "notice": (
                        "The bounded diagnostic exceeded the whole-payload budget; "
                        "the complete raw evidence remains in the run artifacts."
                    ),
                    "omitted_compacted_utf8_bytes": max(
                        compact_utf8_bytes - len(excerpt.encode("utf-8")), 0
                    ),
                },
                indent=2,
                sort_keys=True,
            )

        # JSON escaping can make a character excerpt substantially larger than
        # its source bytes. Binary-search the exact serialized envelope so the
        # persisted/sent payload itself, not just its pre-escape excerpt, is bounded.
        low = 0
        high = len(compact_rendered)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = envelope(compact_rendered[:middle])
            if len(candidate.encode("utf-8")) <= _REPAIR_FEEDBACK_MAX_BYTES:
                low = middle
            else:
                high = middle - 1
        rendered = envelope(compact_rendered[:low])
        truncation["omitted_utf8_bytes"] = max(
            compact_utf8_bytes
            - len(compact_rendered[:low].encode("utf-8")),
            0,
        )
    utf8_bytes = len(rendered.encode("utf-8"))
    metadata = {
        "source_sha256": _sha256_text(source),
        "source_characters": len(source),
        "source_utf8_bytes": source_utf8_bytes,
        "prompt_feedback_sha256": _sha256_text(rendered),
        "characters": len(rendered),
        "utf8_bytes": utf8_bytes,
        "max_utf8_bytes": _REPAIR_FEEDBACK_MAX_BYTES,
        "within_budget": utf8_bytes <= _REPAIR_FEEDBACK_MAX_BYTES,
        "truncated": bool(truncation) or whole_payload_truncated,
        "whole_payload_truncated": whole_payload_truncated,
        "truncation_counts": dict(sorted(truncation.items())),
        "compaction_profile": {
            "string_limit": profile_used[0],
            "list_limit": profile_used[1],
            "dict_limit": profile_used[2],
            "max_depth": profile_used[3],
        }
        if profile_used
        else None,
    }
    return rendered, metadata


def _git_provenance() -> dict[str, Any]:
    try:
        runner = CommandRunner()
        commit = runner.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            timeout=10,
            check=False,
        )
        status = runner.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            timeout=10,
            check=False,
        )
        if commit.returncode or status.returncode:
            raise RuntimeError(commit.stderr.strip() or status.stderr.strip())
        return {
            "commit": commit.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
            "status_sha256": _sha256_text(status.stdout),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _source_state_snapshot(extra_paths: list[Path]) -> dict[str, Any]:
    """Capture exact active source/config text for dirty-run reproducibility."""

    candidates: set[Path] = set(extra_paths)
    for relative_root, patterns in (
        ("aipycraft_k8s", ("*.py",)),
        (
            "benchmark/private_tests",
            ("*.py", "*.md", "*.yaml", "*.yml", "*.json", "*.txt"),
        ),
        ("benchmark/tasks", ("*.json", "*.txt", "*.md")),
        ("benchmark/prompts", ("*.txt",)),
        ("benchmark/environment", ("*.json", "*.yaml", "Dockerfile", "*.sh")),
    ):
        root = PROJECT_ROOT / relative_root
        for pattern in patterns:
            candidates.update(path for path in root.rglob(pattern) if path.is_file())
    for name in (
        "requirements.txt",
        "requirements-aipycraft-k8s.txt",
        "pyproject.toml",
    ):
        path = PROJECT_ROOT / name
        if path.is_file():
            candidates.add(path)

    files: list[dict[str, Any]] = []
    for path in sorted(candidates, key=lambda value: str(value).lower()):
        resolved = path.resolve()
        if not resolved.is_file():
            continue
        try:
            relative = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            relative = str(resolved)
        content = resolved.read_text(encoding="utf-8")
        files.append(
            {
                "path": relative,
                "sha256": _sha256_text(content),
                "utf8_bytes": len(content.encode("utf-8")),
                "content": content,
            }
        )
    digest_input = "\n".join(
        f"{item['path']}\0{item['sha256']}" for item in files
    )
    return {
        "schema_version": 1,
        "scope": "active pipeline, prompts, task registry, evaluator, and environment inputs",
        "files": files,
        "file_count": len(files),
        "aggregate_sha256": _sha256_text(digest_input),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _api_ready(result: Any) -> bool:
    return result.returncode == 0 and "ok" in result.stdout.lower()


def _nested_truthy_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if value.get(key) is True:
            return True
        return any(_nested_truthy_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_nested_truthy_key(item, key) for item in value)
    return False


def _validator_ground_truth(
    *,
    satisfies: bool,
    candidate_correct: bool | None,
    basis: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Compare one validator verdict with evidence obtained after deployment."""

    predicted_defect = not satisfies
    if candidate_correct is None:
        return {
            "status": "unscored",
            "basis": basis,
            "reason": reason,
            "validator_predicted_defect": predicted_defect,
            "candidate_correct": None,
            "classification": None,
            "metric_scope": "candidate_level_downstream_outcome_agreement",
            "citation_correctness": None,
        }
    candidate_defective = not candidate_correct
    classification = {
        (True, True): "true_positive",
        (True, False): "false_positive",
        (False, True): "false_negative",
        (False, False): "true_negative",
    }[(predicted_defect, candidate_defective)]
    return {
        "status": "scored",
        "basis": basis,
        "reason": reason,
        "validator_predicted_defect": predicted_defect,
        "candidate_correct": candidate_correct,
        "classification": classification,
        "metric_scope": "candidate_level_downstream_outcome_agreement",
        "citation_correctness": None,
    }


@dataclass(frozen=True)
class PipelineRun:
    summary: dict[str, Any]
    run_dir: Path

    def public_summary(self) -> dict[str, Any]:
        summary = self.summary
        return {
            "run_id": summary["run_id"],
            "task_id": summary["task_id"],
            "status": summary["status"],
            "mode": summary["mode"],
            "attempts": len(summary["attempts"]),
            "regenerations": summary["regenerations"],
            "duplicate_generation_responses": summary[
                "duplicate_generation_responses"
            ],
            "usage_totals": summary["usage_totals"],
            "unknown_cost_failures": summary["unknown_cost_failures"],
            "usage_by_role": summary.get("usage_by_role"),
            "ai_pre_validation": summary.get("ai_pre_validation_public"),
            "execution_gate": summary.get("execution_gate_public"),
            "post_execution": summary.get("post_execution_public"),
            "run_dir": str(self.run_dir),
            "fatal_error": summary.get("fatal_error"),
        }


class KubernetesAIPyCraftPipeline:
    def __init__(
        self,
        config: AppConfig,
        client: TextGenerationClient,
        harness: IsolatedKindHarness,
        *,
        post_verifier: Callable[..., dict[str, Any]] = run_post_execution_verifier,
        execution_verifier: Callable[..., dict[str, Any]] = run_execution_gate,
        runtime_observer: Callable[..., dict[str, Any]] = observe_runtime,
        validator_client: TextGenerationClient | None = None,
        key_context_supplier: Callable[[], dict[str, Any]] | None = None,
    ):
        self.config = config
        self.client = client
        self.harness = harness
        self.post_verifier = post_verifier
        self.execution_verifier = execution_verifier
        self.runtime_observer = runtime_observer
        self.validator_client = validator_client or client
        self.key_context_supplier = key_context_supplier

    @staticmethod
    def _initial_user_prompt(task: BenchmarkTask) -> str:
        return (
            "<public_task>\n"
            f"{task.description.rstrip()}\n"
            "</public_task>\n"
        )

    def _repair_user_prompt(
        self, task: BenchmarkTask, previous: str, trigger: str, failure: str
    ) -> str:
        return (
            "<public_task>\n"
            f"{task.description.rstrip()}\n\n"
            "</public_task>\n\n"
            "<previous_response>\n"
            f"{previous.rstrip()}\n\n"
            "</previous_response>\n\n"
            f'<correction_feedback trigger="{trigger}">\n'
            f"{failure.rstrip()}\n"
            "</correction_feedback>\n"
        )

    @staticmethod
    def _failure_payload(kind: str, detail: Any) -> dict[str, Any]:
        if kind == "yaml_syntax":
            text = str(detail)
        elif kind == "generation_incomplete":
            text = str(detail)
        elif kind == "ai_validator":
            text = str(detail)
        elif kind == "deployment":
            audit = detail if isinstance(detail, dict) else detail.audit_dict()
            text, metadata = _bounded_feedback_json(audit)
            return {"text": text, **metadata}
        elif kind == "runtime":
            text, metadata = _bounded_feedback_json(
                {
                    "message": "The deployed candidate produced concrete runtime failures.",
                    "failures": detail.get("failures", []),
                    "container_logs": detail.get("container_logs", []),
                    "warning_events": detail.get("warning_events", []),
                }
            )
            return {"text": text, **metadata}
        elif kind == "execution_gate":
            text, metadata = _bounded_feedback_json(
                detail.get("repair_feedback")
                or {
                    "message": "The candidate failed generic operational checks.",
                    "failures": detail.get("failures", []),
                }
            )
            return {"text": text, **metadata}
        elif kind == "candidate_api_disruption":
            text, metadata = _bounded_feedback_json(detail)
            return {"text": text, **metadata}
        else:
            raise ValueError(f"Unknown failure kind {kind}")

        encoded = text.encode("utf-8")
        source_utf8_bytes = len(encoded)
        if len(encoded) > _REPAIR_FEEDBACK_MAX_BYTES:
            text = encoded[:_REPAIR_FEEDBACK_MAX_BYTES].decode(
                "utf-8", errors="ignore"
            )
            truncated = True
        else:
            truncated = False
        return {
            "text": text,
            "source_sha256": _sha256_text(str(detail)),
            "source_characters": len(str(detail)),
            "source_utf8_bytes": source_utf8_bytes,
            "prompt_feedback_sha256": _sha256_text(text),
            "characters": len(text),
            "utf8_bytes": len(text.encode("utf-8")),
            "max_utf8_bytes": _REPAIR_FEEDBACK_MAX_BYTES,
            "within_budget": True,
            "truncated": truncated,
            "whole_payload_truncated": truncated,
            "truncation_counts": (
                {"omitted_utf8_bytes": source_utf8_bytes - len(text.encode("utf-8"))}
                if truncated
                else {}
            ),
            "compaction_profile": None,
        }

    @staticmethod
    def _failure_text(kind: str, detail: Any) -> str:
        """Backward-compatible accessor used by focused unit tests."""

        return str(KubernetesAIPyCraftPipeline._failure_payload(kind, detail)["text"])

    @staticmethod
    def _usage_totals(
        results: list[GenerationResult],
        *,
        terminal_transport_attempts: int = 0,
        unknown_cost_failures: int = 0,
    ) -> dict[str, Any]:
        cost = sum((item.cost_usd for item in results), Decimal("0"))
        return {
            "requests": len(results),
            "http_post_attempts": sum(
                item.transport_retries + 1 for item in results
            )
            + terminal_transport_attempts,
            "prompt_tokens": sum(item.prompt_tokens for item in results),
            "completion_tokens": sum(item.completion_tokens for item in results),
            "total_tokens": sum(item.total_tokens for item in results),
            "reasoning_tokens": sum(item.reasoning_tokens for item in results),
            "cached_tokens": sum(item.cached_tokens for item in results),
            "response_characters": sum(len(item.raw_text) for item in results),
            "response_utf8_bytes": sum(
                len(item.raw_text.encode("utf-8")) for item in results
            ),
            "response_lines": sum(len(item.raw_text.splitlines()) for item in results),
            "suspicious_provider_usage_responses": sum(
                bool(item.usage_consistency_issues()) for item in results
            ),
            "latency_ms": sum(item.latency_ms for item in results),
            "transport_attempt_duration_ms": sum(
                int(attempt.get("duration_ms", 0) or 0)
                for item in results
                for attempt in item.transport_attempt_log
            ),
            "transport_retry_sleep_duration_ms": sum(
                int(attempt.get("retry_sleep_duration_ms", 0) or 0)
                for item in results
                for attempt in item.transport_attempt_log
            ),
            "cost_usd": str(cost),
            "provider_cost_complete": all(
                item.provider_cost_complete for item in results
            )
            and unknown_cost_failures == 0,
            "usage_complete": all(item.usage_complete for item in results)
            and unknown_cost_failures == 0,
            "unobserved_billable_attempts": sum(
                item.unobserved_billable_attempts for item in results
            )
            + unknown_cost_failures,
        }

    @staticmethod
    def _public_post(report: dict[str, Any] | None) -> dict[str, Any] | None:
        if not report:
            return None
        return {
            "status": report.get("status"),
            "passed": report.get("passed"),
            "failed": report.get("failed"),
            "total": report.get("total"),
        }

    def _invoke_post_verifier(
        self,
        task: BenchmarkTask,
        environment: Any,
        candidate_path: Path,
    ) -> dict[str, Any]:
        """Pass the exact deployed source when supported, preserving test injectors."""

        try:
            inspect.signature(self.post_verifier).bind(
                task, environment, candidate_path
            )
        except (TypeError, ValueError):
            return self.post_verifier(task, environment)
        return self.post_verifier(task, environment, candidate_path)

    @staticmethod
    def _model_audit(
        result: GenerationResult,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        return {
            **result.audit_dict(),
            "raw_response_sha256": _sha256_text(result.raw_text),
            "local_prompt_metrics": {
                "system_characters": len(system_prompt),
                "system_utf8_bytes": len(system_prompt.encode("utf-8")),
                "system_lines": len(system_prompt.splitlines()),
                "user_characters": len(user_prompt),
                "user_utf8_bytes": len(user_prompt.encode("utf-8")),
                "user_lines": len(user_prompt.splitlines()),
            },
        }

    def _check_task_environment_invariants(
        self, task: BenchmarkTask, generate_prompt: str, repair_prompt: str
    ) -> None:
        lock = self.config.environment.lock
        if task.kubernetes_version != lock.kubernetes_minor:
            raise EnvironmentError(
                "Task Kubernetes version does not match the frozen cluster"
            )
        busybox_tags = {
            image.tag for image in lock.preload_images if image.tag.endswith("busybox:1.36.1")
        }
        if not busybox_tags or task.execution_image != "busybox:1.36.1":
            raise EnvironmentError(
                "Task execution image does not match the frozen BusyBox image"
            )
        for label, prompt in (("generate", generate_prompt), ("repair", repair_prompt)):
            if "Kubernetes 1.35" not in prompt or "busybox:1.36.1" not in prompt:
                raise EnvironmentError(
                    f"The {label} prompt drifted from the frozen environment"
                )

    def _evaluate_candidate_once(
        self,
        task: BenchmarkTask,
        environment: Any,
        candidate_path: Path,
        evidence_dir: Path,
    ) -> dict[str, Any]:
        """Deploy and evaluate one unchanged candidate in one fresh cluster."""

        evaluation: dict[str, Any] = {
            "result": "running",
            "deployment": None,
            "deployment_failure_api_readiness": None,
            "runtime": None,
            "runtime_failure_api_readiness": None,
            "execution_gate": None,
            "execution_failure_api_readiness": None,
            "post_execution": None,
            "error": None,
            "api_loss_stage": None,
            "stage_timings_ms": {},
        }
        stage_started = time.monotonic()
        deployment = environment.deploy(candidate_path)
        evaluation["stage_timings_ms"]["deployment"] = round(
            (time.monotonic() - stage_started) * 1000
        )
        evaluation["deployment"] = deployment.audit_dict()
        _write_json(evidence_dir / "deployment.json", deployment.audit_dict())
        if deployment.returncode != 0:
            readiness = environment.readyz()
            evaluation["deployment_failure_api_readiness"] = readiness.audit_dict()
            _write_json(
                evidence_dir / "deployment_failure_api_readiness.json",
                readiness.audit_dict(),
            )
            if not _api_ready(readiness):
                evaluation["result"] = "candidate_api_loss"
                evaluation["api_loss_stage"] = "deployment"
                evaluation["error"] = (
                    readiness.stderr.strip()
                    or readiness.stdout.strip()
                    or "Kubernetes API readiness check failed"
                )
                return evaluation
            evaluation["result"] = "deployment_error"
            return evaluation

        stage_started = time.monotonic()
        try:
            runtime = self.runtime_observer(
                environment,
                seconds=self.config.pipeline.runtime_observation_seconds,
                poll_seconds=self.config.pipeline.runtime_poll_seconds,
            )
        except (EnvironmentError, CommandError) as exc:
            readiness = environment.readyz()
            evaluation["runtime_failure_api_readiness"] = readiness.audit_dict()
            _write_json(
                evidence_dir / "runtime_failure_api_readiness.json",
                readiness.audit_dict(),
            )
            evaluation["error"] = str(exc)
            if not _api_ready(readiness):
                evaluation["result"] = "candidate_api_loss"
                evaluation["api_loss_stage"] = "runtime_observation"
            else:
                evaluation["result"] = "infrastructure_error"
            evaluation["stage_timings_ms"]["runtime_observation"] = round(
                (time.monotonic() - stage_started) * 1000
            )
            return evaluation
        evaluation["stage_timings_ms"]["runtime_observation"] = round(
            (time.monotonic() - stage_started) * 1000
        )
        evaluation["runtime"] = runtime
        _write_json(evidence_dir / "runtime.json", runtime)
        if runtime.get("failures"):
            evaluation["result"] = "runtime_error"
            return evaluation

        stage_started = time.monotonic()
        execution = self.execution_verifier(
            task,
            environment,
            timeout_seconds=self.config.pipeline.command_timeout_seconds,
        )
        evaluation["stage_timings_ms"]["generic_execution_gate"] = round(
            (time.monotonic() - stage_started) * 1000
        )
        evaluation["execution_gate"] = execution
        _write_json(evidence_dir / "execution_gate.json", execution)
        if execution.get("status") == "failed":
            evaluation["result"] = "execution_gate_error"
            return evaluation
        if execution.get("status") != "passed":
            readiness = environment.readyz()
            evaluation["execution_failure_api_readiness"] = readiness.audit_dict()
            _write_json(
                evidence_dir / "execution_failure_api_readiness.json",
                readiness.audit_dict(),
            )
            evaluation["error"] = execution.get(
                "error", "Generic execution gate failed"
            )
            if not _api_ready(readiness):
                evaluation["result"] = "candidate_api_loss"
                evaluation["api_loss_stage"] = "execution_gate"
            else:
                evaluation["result"] = "infrastructure_error"
            return evaluation

        stage_started = time.monotonic()
        post = self._invoke_post_verifier(task, environment, candidate_path)
        evaluation["stage_timings_ms"]["hidden_specification_oracle"] = round(
            (time.monotonic() - stage_started) * 1000
        )
        evaluation["post_execution"] = post
        _write_json(evidence_dir / "post_execution.json", post)
        if post.get("status") == "passed":
            evaluation["result"] = "accepted"
        elif post.get("status") == "failed":
            evaluation["result"] = "specification_discrepancy"
        else:
            evaluation["result"] = "post_verifier_infrastructure_error"
            evaluation["error"] = post.get(
                "error", "Hidden specification verifier failed"
            )
        return evaluation

    def run(
        self,
        task: BenchmarkTask,
        *,
        initial_candidate: CandidateBankEntry | None = None,
    ) -> PipelineRun:
        run_id = _run_id()
        run_dir = self.config.pipeline.output_root / run_id
        private_dir = run_dir / "private"
        private_dir.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        role_generator = "aipycraft_generator"
        role_validator = "ai_pre_validator"
        model_results: list[GenerationResult] = []
        results_by_role: dict[str, list[GenerationResult]] = {
            role_generator: [],
            role_validator: [],
        }
        unknown_cost_by_role = {role_generator: 0, role_validator: 0}
        terminal_transport_by_role = {role_generator: 0, role_validator: 0}
        aipycraft_response_hashes: list[str] = []
        attempts: list[dict[str, Any]] = []
        attempt_start_times: dict[int, float] = {}
        current_attempt_dir: Path | None = None
        active_role = role_generator
        active_system_prompt = ""
        active_user_prompt = ""

        def record_model_result(role: str, result: GenerationResult) -> None:
            model_results.append(result)
            results_by_role[role].append(result)
            unknown_cost_by_role[role] += result.unobserved_billable_attempts

        summary: dict[str, Any] = {
            "run_id": run_id,
            "task_id": task.task_id,
            "status": "running",
            "mode": "paid" if self.client.billable else "replay",
            "treatment_id": treatment_id(self.config),
            "started_at": _utc_now(),
            "config": str(self.config.path),
            "config_sha256": _sha256_file(self.config.path),
            "environment_lock": str(self.config.environment.lock.path),
            "environment_lock_sha256": _sha256_file(
                self.config.environment.lock.path
            ),
            "kind_config": str(self.config.environment.kind_config),
            "kind_config_sha256": _sha256_file(self.config.environment.kind_config),
            "task_description": str(task.description_path),
            "task_description_sha256": _sha256_text(task.description),
            "model": self.config.api.model,
            "provider_only": list(self.config.api.provider_only),
            "allow_fallbacks": self.config.api.allow_fallbacks,
            "temperature": self.config.api.temperature,
            "reasoning_effort": self.config.api.reasoning_effort,
            "transport_retries": self.config.api.transport_retries,
            "api_base": self.config.api.base_url,
            "response_format": "raw_text",
            "initial_candidate_source": (
                initial_candidate.source
                if initial_candidate is not None
                else "model_request"
            ),
            "candidate_bank": (
                initial_candidate.provenance()
                if initial_candidate is not None
                else None
            ),
            "max_output_tokens": None,
            "input_usd_per_million": str(
                self.config.api.input_usd_per_million
            ),
            "output_usd_per_million": str(
                self.config.api.output_usd_per_million
            ),
            "max_regenerations": self.config.pipeline.max_regenerations,
            "candidate_api_loss_confirmation_replays": (
                self.config.pipeline.candidate_api_loss_confirmation_replays
            ),
            "environment_setup_retries": (
                self.config.pipeline.environment_setup_retries
            ),
            "runtime_observation_seconds": (
                self.config.pipeline.runtime_observation_seconds
            ),
            "runtime_poll_seconds": self.config.pipeline.runtime_poll_seconds,
            "command_timeout_seconds": self.config.pipeline.command_timeout_seconds,
            "kind_create_timeout_seconds": max(
                300, self.config.pipeline.command_timeout_seconds * 3
            ),
            "max_aipycraft_generations": self.config.pipeline.max_regenerations + 1,
            "regenerations": 0,
            "regeneration_policy": {
                "eligible_triggers": [
                    "generation_incomplete",
                    "yaml_syntax",
                    "ai_validator",
                    "deployment",
                    "runtime",
                    "execution_gate",
                    "candidate_api_disruption",
                ],
                "validator_repair_scope": (
                    "cited_defects_and_required_dependent_changes_only"
                    if self.config.ai_validator.feedback_mode == "detailed"
                    else "negative_verdict_without_defect_details"
                ),
                "candidate_execution_failures_regenerate": True,
                "confirmed_candidate_api_disruption_regenerates": True,
                "same_candidate_api_loss_confirmation_is_not_regeneration": True,
                "cleanup_verified_environment_setup_retry_is_not_regeneration": True,
                "hidden_or_confirmed_infrastructure_failures_regenerate": False,
                "transport_retry_is_same_request_not_candidate_regeneration": True,
            },
            "duplicate_generation_responses": 0,
            "ai_validator": {
                "enabled": self.config.ai_validator.enabled,
                "feedback_mode": self.config.ai_validator.feedback_mode,
                "shadow_mode": self.config.ai_validator.shadow_mode,
                "model": self.config.ai_validator.api.model,
                "provider_only": list(self.config.ai_validator.api.provider_only),
                "reasoning_effort": self.config.ai_validator.api.reasoning_effort,
                "temperature": self.config.ai_validator.api.temperature,
                "transport_retries": self.config.ai_validator.api.transport_retries,
                "input_usd_per_million": str(
                    self.config.ai_validator.api.input_usd_per_million
                ),
                "output_usd_per_million": str(
                    self.config.ai_validator.api.output_usd_per_million
                ),
                "intervention_active": (
                    self.config.ai_validator.enabled
                    and not self.config.ai_validator.shadow_mode
                ),
                "same_model_and_endpoint_as_aipycraft": (
                    self.config.ai_validator.api == self.config.api
                ),
                "failure_policy": "abort_on_validator_or_provider_error",
            },
            "attempts": attempts,
            "ai_pre_validation_public": None,
            "execution_gate_public": None,
            "post_execution_public": None,
            "unknown_cost_failures": 0,
            "stage_timings_ms": {},
            "provenance": {
                "git": _git_provenance(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "requirements_sha256": _sha256_file(PROJECT_ROOT / "requirements.txt"),
                "pipeline_requirements_sha256": _sha256_file(
                    PROJECT_ROOT / "requirements-aipycraft-k8s.txt"
                ),
                "node_image": {
                    "tag": self.config.environment.lock.node_image,
                    "source": self.config.environment.lock.node_image_source,
                    "id": None,
                    "id_source": "machine_preparation_receipt",
                    "dockerfile_sha256": _sha256_file(
                        self.config.environment.lock.node_image_dockerfile
                    ),
                    "entrypoint_sha256": _sha256_file(
                        self.config.environment.lock.node_image_entrypoint
                    ),
                },
                "hidden_evaluator": {
                    "core_sha256": _sha256_file(
                        PROJECT_ROOT / "benchmark/private_tests/core.py"
                    ),
                    "suites_sha256": _sha256_file(
                        PROJECT_ROOT / "benchmark/private_tests/suites.py"
                    ),
                },
                "generic_execution_verifier_sha256": _sha256_file(
                    PROJECT_ROOT / "aipycraft_k8s/execution_verifier.py"
                ),
                "ai_pre_validator_sha256": _sha256_file(
                    PROJECT_ROOT / "aipycraft_k8s/pre_validator.py"
                ),
            },
            "fatal_error": None,
            "failure_origin": None,
            "failure_stage": None,
        }
        source_snapshot_started = time.monotonic()
        source_state = _source_state_snapshot(
            [
                self.config.path,
                self.config.environment.lock.path,
                self.config.environment.kind_config,
                self.config.prompts.generate,
                self.config.prompts.regenerate,
                self.config.prompts.validator_detailed,
                self.config.prompts.validator_verdict_only,
            ]
        )
        summary["stage_timings_ms"]["source_snapshot"] = round(
            (time.monotonic() - source_snapshot_started) * 1000
        )
        _write_json(private_dir / "source_state.json", source_state)
        summary["provenance"]["source_state"] = {
            "artifact": "source_state.json",
            "schema_version": source_state["schema_version"],
            "file_count": source_state["file_count"],
            "aggregate_sha256": source_state["aggregate_sha256"],
        }
        _write_json(private_dir / "summary.json", summary)

        def queue_regeneration(
            attempt: dict[str, Any],
            attempt_dir: Path,
            *,
            trigger: str,
            failure_kind: str,
            detail: Any,
            evidence_artifacts: list[str],
        ) -> str:
            feedback_started = time.monotonic()
            payload = self._failure_payload(failure_kind, detail)
            feedback_text = str(payload.pop("text"))
            record = {
                "schema_version": 1,
                "trigger": trigger,
                "source_attempt_result": attempt.get("result"),
                "feedback_policy": (
                    "full raw evidence retained privately; exact bounded payload "
                    "below is sent to the next generator request"
                ),
                "prompt_feedback": feedback_text,
                "prompt_feedback_metrics": payload,
                "full_evidence_artifacts": [
                    name for name in evidence_artifacts if (attempt_dir / name).exists()
                ],
                "hidden_oracle_evidence_included": False,
            }
            attempt["regeneration_trigger"] = trigger
            attempt["regeneration_feedback"] = {
                key: value for key, value in record.items() if key != "prompt_feedback"
            }
            attempt["finished_at"] = _utc_now()
            started_at = attempt_start_times.get(int(attempt["attempt"]))
            if started_at is not None:
                attempt["duration_ms"] = round(
                    (time.monotonic() - started_at) * 1000
                )
            attempt["stage_timings_ms"]["regeneration_feedback_preparation"] = round(
                (time.monotonic() - feedback_started) * 1000
            )
            _write_json(attempt_dir / "regeneration_feedback.json", record)
            summary["regenerations"] += 1
            _write_json(private_dir / "summary.json", summary)
            return feedback_text

        try:
            preflight_started = time.monotonic()
            try:
                preflight = self.harness.preflight()
            finally:
                summary["stage_timings_ms"]["environment_preflight"] = round(
                    (time.monotonic() - preflight_started) * 1000
                )
            summary["provenance"]["node_image"]["id"] = (
                preflight.get("checks", {}).get("node_image", {}).get("id")
            )
            _write_json(private_dir / "environment_preflight.json", preflight)
            if self.key_context_supplier:
                key_before_started = time.monotonic()
                try:
                    summary["openrouter_key_before"] = self.key_context_supplier()
                except Exception as exc:
                    summary["openrouter_key_before_error"] = str(exc)
                finally:
                    summary["stage_timings_ms"]["openrouter_key_before"] = round(
                        (time.monotonic() - key_before_started) * 1000
                    )

            generate_system = self.config.prompts.generate.read_text(encoding="utf-8")
            repair_system = self.config.prompts.regenerate.read_text(encoding="utf-8")
            validator_path = (
                self.config.prompts.validator_detailed
                if self.config.ai_validator.feedback_mode == "detailed"
                else self.config.prompts.validator_verdict_only
            )
            validator_system = validator_path.read_text(encoding="utf-8")
            self._check_task_environment_invariants(
                task, generate_system, repair_system
            )
            summary["generate_prompt_sha256"] = _sha256_text(generate_system)
            summary["repair_prompt_sha256"] = _sha256_text(repair_system)
            summary["validator_prompt"] = str(validator_path)
            summary["validator_prompt_sha256"] = _sha256_text(validator_system)
            previous = ""
            previous_valid_candidate: str | None = None
            repair_trigger = ""
            repair_failure = ""
            terminal = False
            auxiliary_cluster_count = 0
            for index in range(self.config.pipeline.max_regenerations + 1):
                number = index + 1
                attempt_start_times[number] = time.monotonic()
                attempt_dir = private_dir / f"attempt-{number:02d}"
                current_attempt_dir = attempt_dir
                attempt_dir.mkdir(parents=True, exist_ok=False)
                system_prompt = generate_system if index == 0 else repair_system
                user_prompt = (
                    self._initial_user_prompt(task)
                    if index == 0
                    else self._repair_user_prompt(
                        task, previous, repair_trigger, repair_failure
                    )
                )
                active_role = role_generator
                active_system_prompt = system_prompt
                active_user_prompt = user_prompt
                (attempt_dir / "system_prompt.txt").write_text(
                    system_prompt, encoding="utf-8"
                )
                (attempt_dir / "user_prompt.txt").write_text(
                    user_prompt, encoding="utf-8"
                )
                attempt: dict[str, Any] = {
                    "attempt": number,
                    "started_at": _utc_now(),
                    "system_prompt_sha256": _sha256_text(system_prompt),
                    "user_prompt_sha256": _sha256_text(user_prompt),
                    "generation": None,
                    "pre_execution": None,
                    "candidate_semantic_diff": None,
                    "ai_pre_validation": None,
                    "deployment": None,
                    "deployment_failure_api_readiness": None,
                    "runtime": None,
                    "execution_gate": None,
                    "post_execution": None,
                    "candidate_evaluations": [],
                    "environment_attempts": [],
                    "environment_setup_retries": 0,
                    "api_loss_confirmation_replays": 0,
                    "validator_ground_truth": None,
                    "validator_outcome_agreement": None,
                    "regeneration_trigger": None,
                    "regeneration_feedback": None,
                    "stage_timings_ms": {},
                    "result": "running",
                }
                attempts.append(attempt)
                _write_json(private_dir / "summary.json", summary)
                generation_started = time.monotonic()
                banked_initial = index == 0 and initial_candidate is not None
                if banked_initial:
                    raw_text = initial_candidate.raw_text
                    finish_reason = initial_candidate.finish_reason
                    attempt["stage_timings_ms"]["initial_candidate_load"] = round(
                        (time.monotonic() - generation_started) * 1000
                    )
                    generation_audit = {
                        "source": initial_candidate.source,
                        "candidate_id": initial_candidate.candidate_id,
                        "candidate_record": str(initial_candidate.record_path),
                        "candidate_record_sha256": initial_candidate.provenance()[
                            "record_sha256"
                        ],
                        "raw_response_sha256": initial_candidate.raw_response_sha256,
                        "finish_reason": finish_reason,
                        "shared_initial_generation_usage": (
                            initial_candidate.metadata.get("generation")
                        ),
                        "charged_to_treatment_run": False,
                        "local_prompt_metrics": {
                            "system_characters": len(system_prompt),
                            "system_utf8_bytes": len(system_prompt.encode("utf-8")),
                            "system_lines": len(system_prompt.splitlines()),
                            "user_characters": len(user_prompt),
                            "user_utf8_bytes": len(user_prompt.encode("utf-8")),
                            "user_lines": len(user_prompt.splitlines()),
                        },
                    }
                else:
                    try:
                        generation = self.client.complete(system_prompt, user_prompt)
                    finally:
                        attempt["stage_timings_ms"]["generator_request"] = round(
                            (time.monotonic() - generation_started) * 1000
                        )
                    record_model_result(role_generator, generation)
                    raw_text = generation.raw_text
                    finish_reason = generation.finish_reason
                    generation_audit = self._model_audit(
                        generation, system_prompt, user_prompt
                    )
                raw_response_sha256 = _sha256_text(raw_text)
                duplicate_of_attempt = next(
                    (
                        prior_index + 1
                        for prior_index, prior_hash in enumerate(
                            aipycraft_response_hashes
                        )
                        if prior_hash == raw_response_sha256
                    ),
                    None,
                )
                aipycraft_response_hashes.append(raw_response_sha256)
                if duplicate_of_attempt is not None:
                    summary["duplicate_generation_responses"] += 1
                previous = raw_text
                (attempt_dir / "raw_response.txt").write_text(
                    raw_text, encoding="utf-8"
                )
                attempt["generation"] = {
                    **generation_audit,
                    "duplicate_of_attempt": duplicate_of_attempt,
                }
                _write_json(attempt_dir / "generation.json", attempt["generation"])
                _write_json(private_dir / "summary.json", summary)

                if finish_reason != "stop":
                    attempt["result"] = "incomplete_generation"
                    detail = (
                        "The provider returned finish_reason="
                        f"{finish_reason or '<missing>'}; the response may be "
                        "incomplete and was not treated as a candidate manifest."
                    )
                    _write_json(
                        attempt_dir / "generation_completeness.json",
                        {"complete": False, "error": detail},
                    )
                    if index < self.config.pipeline.max_regenerations:
                        repair_trigger = "generation_incomplete"
                        repair_failure = queue_regeneration(
                            attempt,
                            attempt_dir,
                            trigger=repair_trigger,
                            failure_kind="generation_incomplete",
                            detail=detail,
                            evidence_artifacts=[
                                "generation.json",
                                "generation_completeness.json",
                                "raw_response.txt",
                            ],
                        )
                        continue
                    summary["status"] = "candidate_failed"
                    terminal = True
                    break

                syntax_started = time.monotonic()
                syntax = check_yaml_syntax(raw_text)
                attempt["stage_timings_ms"]["yaml_pre_execution"] = round(
                    (time.monotonic() - syntax_started) * 1000
                )
                attempt["pre_execution"] = syntax.audit_dict()
                _write_json(attempt_dir / "pre_execution.json", syntax.audit_dict())
                (attempt_dir / "candidate.yaml").write_text(
                    syntax.candidate, encoding="utf-8"
                )
                _write_json(private_dir / "summary.json", summary)
                if not syntax.valid:
                    attempt["result"] = "yaml_syntax_error"
                    if index < self.config.pipeline.max_regenerations:
                        repair_trigger = "yaml_syntax"
                        repair_failure = queue_regeneration(
                            attempt,
                            attempt_dir,
                            trigger=repair_trigger,
                            failure_kind="yaml_syntax",
                            detail=(
                                syntax.error or "unknown YAML syntax error"
                            ),
                            evidence_artifacts=[
                                "pre_execution.json",
                                "candidate.yaml",
                                "raw_response.txt",
                            ],
                        )
                        continue
                    summary["status"] = "candidate_failed"
                    terminal = True
                    break

                semantic_diff_started = time.monotonic()
                semantic_diff = candidate_semantic_diff(
                    previous_valid_candidate, syntax.candidate
                )
                attempt["stage_timings_ms"]["candidate_semantic_diff"] = round(
                    (time.monotonic() - semantic_diff_started) * 1000
                )
                previous_valid_candidate = syntax.candidate
                attempt["candidate_semantic_diff"] = semantic_diff
                _write_json(attempt_dir / "candidate_semantic_diff.json", semantic_diff)

                validation_decision_satisfies: bool | None = None
                if self.config.ai_validator.enabled:
                    validation_user = validator_user_prompt(
                        task.description, syntax.candidate
                    )
                    active_role = role_validator
                    active_system_prompt = validator_system
                    active_user_prompt = validation_user
                    (attempt_dir / "ai_validator_system_prompt.txt").write_text(
                        validator_system, encoding="utf-8"
                    )
                    (attempt_dir / "ai_validator_user_prompt.txt").write_text(
                        validation_user, encoding="utf-8"
                    )
                    validator_started = time.monotonic()
                    try:
                        validation_result = self.validator_client.complete(
                            validator_system, validation_user
                        )
                        attempt["stage_timings_ms"]["ai_pre_validator_request"] = (
                            round((time.monotonic() - validator_started) * 1000)
                        )
                    except ProviderError as exc:
                        attempt["stage_timings_ms"]["ai_pre_validator_request"] = (
                            round((time.monotonic() - validator_started) * 1000)
                        )
                        if not self.config.ai_validator.shadow_mode:
                            raise
                        rejected_call: dict[str, Any] | None = None
                        if exc.audit_result is not None:
                            audit = exc.audit_result
                            record_model_result(role_validator, audit)
                            (attempt_dir / "ai_validator_raw_response.txt").write_text(
                                audit.raw_text, encoding="utf-8"
                            )
                            rejected_call = {
                                **self._model_audit(
                                    audit, validator_system, validation_user
                                ),
                                "validation_error": str(exc),
                            }
                        else:
                            terminal_transport_by_role[role_validator] += (
                                exc.transport_attempts
                            )
                        unknown_cost_by_role[role_validator] += (
                            exc.unknown_cost_attempts
                        )
                        validation_record = {
                            "enabled": True,
                            "feedback_mode": self.config.ai_validator.feedback_mode,
                            "shadow_mode": True,
                            "status": "shadow_provider_error",
                            "error": str(exc),
                            "model_call": rejected_call,
                            "transport_attempts": exc.transport_attempts,
                            "transport_attempt_log": list(
                                exc.transport_attempt_log
                            ),
                            "unknown_cost_attempts": exc.unknown_cost_attempts,
                        }
                        attempt["ai_pre_validation"] = validation_record
                        summary["ai_pre_validation_public"] = {
                            "enabled": True,
                            "feedback_mode": self.config.ai_validator.feedback_mode,
                            "shadow_mode": True,
                            "status": "shadow_provider_error",
                        }
                        _write_json(
                            attempt_dir / "ai_pre_validation.json", validation_record
                        )
                        validation_result = None
                    if validation_result is None:
                        pass
                    else:
                        record_model_result(role_validator, validation_result)
                        (attempt_dir / "ai_validator_raw_response.txt").write_text(
                            validation_result.raw_text, encoding="utf-8"
                        )
                        validation_audit = self._model_audit(
                            validation_result, validator_system, validation_user
                        )
                        validation_record = {
                            "enabled": True,
                            "feedback_mode": self.config.ai_validator.feedback_mode,
                            "shadow_mode": self.config.ai_validator.shadow_mode,
                            "status": "response_received",
                            "model_call": validation_audit,
                        }
                        attempt["ai_pre_validation"] = validation_record
                        _write_json(
                            attempt_dir / "ai_pre_validation.json", validation_record
                        )
                        validator_parse_started = time.monotonic()
                        try:
                            if validation_result.finish_reason != "stop":
                                raise AiValidationError(
                                    "AI validator returned finish_reason="
                                    f"{validation_result.finish_reason or '<missing>'}"
                                )
                            decision = parse_decision(
                                validation_result.raw_text,
                                self.config.ai_validator.feedback_mode,
                            )
                        except AiValidationError as exc:
                            if not self.config.ai_validator.shadow_mode:
                                raise
                            validation_record = {
                                **validation_record,
                                "status": "shadow_validator_error",
                                "error": str(exc),
                            }
                            attempt["ai_pre_validation"] = validation_record
                            summary["ai_pre_validation_public"] = {
                                "enabled": True,
                                "feedback_mode": (
                                    self.config.ai_validator.feedback_mode
                                ),
                                "shadow_mode": True,
                                "status": "shadow_validator_error",
                            }
                            _write_json(
                                attempt_dir / "ai_pre_validation.json",
                                validation_record,
                            )
                        else:
                            validation_decision_satisfies = decision.satisfies
                            validation_record = {
                                **validation_record,
                                "status": (
                                    "passed" if decision.satisfies else "rejected"
                                ),
                                "decision": decision.audit_dict(),
                            }
                            attempt["ai_pre_validation"] = validation_record
                            _write_json(
                                attempt_dir / "ai_pre_validation.json",
                                validation_record,
                            )
                            summary["ai_pre_validation_public"] = {
                                "enabled": True,
                                "feedback_mode": (
                                    self.config.ai_validator.feedback_mode
                                ),
                                "shadow_mode": self.config.ai_validator.shadow_mode,
                                "status": validation_record["status"],
                                "satisfies": decision.satisfies,
                                "problem_count": len(decision.problems),
                                "intervened": (
                                    not decision.satisfies
                                    and not self.config.ai_validator.shadow_mode
                                ),
                            }
                            if (
                                not decision.satisfies
                                and not self.config.ai_validator.shadow_mode
                            ):
                                attempt["validator_ground_truth"] = (
                                    _validator_ground_truth(
                                        satisfies=False,
                                        candidate_correct=None,
                                        basis="not_deployed_active_validator_rejection",
                                        reason=(
                                            "The active validator blocked deployment; "
                                            "live correctness is unobserved"
                                        ),
                                    )
                                )
                                attempt["validator_outcome_agreement"] = attempt[
                                    "validator_ground_truth"
                                ]
                                _write_json(
                                    attempt_dir / "validator_ground_truth.json",
                                    attempt["validator_ground_truth"],
                                )
                                _write_json(
                                    attempt_dir / "validator_outcome_agreement.json",
                                    attempt["validator_outcome_agreement"],
                                )
                                attempt["result"] = "ai_pre_validation_rejected"
                                if index < self.config.pipeline.max_regenerations:
                                    repair_trigger = "ai_validator"
                                    repair_failure = queue_regeneration(
                                        attempt,
                                        attempt_dir,
                                        trigger=repair_trigger,
                                        failure_kind="ai_validator",
                                        detail=correction_feedback(
                                            decision,
                                            self.config.ai_validator.feedback_mode,
                                        ),
                                        evidence_artifacts=[
                                            "ai_pre_validation.json",
                                            "ai_validator_raw_response.txt",
                                            "candidate.yaml",
                                        ],
                                    )
                                    continue
                                summary["status"] = "candidate_failed"
                                terminal = True
                                break
                        finally:
                            attempt["stage_timings_ms"][
                                "ai_pre_validator_decision_parse"
                            ] = round(
                                (time.monotonic() - validator_parse_started) * 1000
                            )
                else:
                    disabled_record = {
                        "enabled": False,
                        "feedback_mode": self.config.ai_validator.feedback_mode,
                        "shadow_mode": False,
                        "status": "disabled",
                    }
                    attempt["ai_pre_validation"] = disabled_record
                    summary["ai_pre_validation_public"] = disabled_record
                    _write_json(
                        attempt_dir / "ai_pre_validation.json", disabled_record
                    )

                _write_json(private_dir / "summary.json", summary)
                candidate_evaluation: dict[str, Any] | None = None
                maximum_confirmations = (
                    self.config.pipeline.candidate_api_loss_confirmation_replays
                )
                evaluation_index = 0
                setup_retry_index = 0
                while evaluation_index <= maximum_confirmations:
                    if evaluation_index == 0 and setup_retry_index == 0:
                        cluster_attempt_number = number
                        evidence_dir = attempt_dir
                    else:
                        auxiliary_cluster_count += 1
                        cluster_attempt_number = (
                            self.config.pipeline.max_regenerations
                            + 1
                            + auxiliary_cluster_count
                        )
                        if evaluation_index == 0:
                            evidence_name = (
                                f"environment-retry-{setup_retry_index:02d}"
                            )
                        elif setup_retry_index == 0:
                            evidence_name = (
                                f"api-loss-confirmation-{evaluation_index:02d}"
                            )
                        else:
                            evidence_name = (
                                f"api-loss-confirmation-{evaluation_index:02d}-"
                                f"environment-retry-{setup_retry_index:02d}"
                            )
                        evidence_dir = attempt_dir / evidence_name
                        evidence_dir.mkdir(parents=True, exist_ok=False)
                        (evidence_dir / "candidate.yaml").write_text(
                            syntax.candidate, encoding="utf-8"
                        )
                        if evaluation_index > 0 and setup_retry_index == 0:
                            attempt["api_loss_confirmation_replays"] += 1

                    cluster_started = time.monotonic()
                    environment_record: dict[str, Any] = {
                        "evaluation_number": evaluation_index + 1,
                        "environment_setup_retry_number": setup_retry_index,
                        "cluster_attempt_number": cluster_attempt_number,
                        "evidence_dir": str(evidence_dir.relative_to(private_dir)),
                        "started_at": _utc_now(),
                        "status": "running",
                        "setup": None,
                        "cleanup": None,
                    }
                    attempt["environment_attempts"].append(environment_record)
                    _write_json(
                        attempt_dir / "environment_attempts.json",
                        attempt["environment_attempts"],
                    )
                    evaluation_finished: float | None = None
                    environment_ready = False
                    environment_error: BaseException | None = None
                    try:
                        with self.harness.attempt(
                            run_id,
                            cluster_attempt_number,
                            evidence_dir,
                        ) as environment:
                            environment_ready = True
                            setup_finished = time.monotonic()
                            evaluation = self._evaluate_candidate_once(
                                task,
                                environment,
                                attempt_dir / "candidate.yaml",
                                evidence_dir,
                            )
                            evaluation = {
                                **evaluation,
                                "evaluation_number": evaluation_index + 1,
                                "cluster_attempt_number": cluster_attempt_number,
                                "candidate_sha256": _sha256_text(syntax.candidate),
                                "evidence_dir": str(
                                    evidence_dir.relative_to(private_dir)
                                ),
                            }
                            evaluation["stage_timings_ms"][
                                "environment_setup"
                            ] = round((setup_finished - cluster_started) * 1000)
                            evaluation_finished = time.monotonic()
                            attempt["candidate_evaluations"].append(evaluation)
                            _write_json(
                                attempt_dir / "candidate_evaluations.json",
                                attempt["candidate_evaluations"],
                            )
                        environment_record["status"] = "completed"
                        environment_record["evaluation_result"] = evaluation.get(
                            "result"
                        )
                    except BaseException as exc:
                        environment_record["status"] = "failed"
                        environment_record["error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        environment_error = exc
                    finally:
                        cluster_finished = time.monotonic()
                        environment_record["finished_at"] = _utc_now()
                        environment_record["duration_ms"] = round(
                            (cluster_finished - cluster_started) * 1000
                        )
                        setup_path = evidence_dir / "setup_stages.json"
                        if setup_path.is_file():
                            try:
                                setup_detail = json.loads(
                                    setup_path.read_text(encoding="utf-8")
                                )
                                environment_record["setup"] = {
                                    "artifact": str(
                                        setup_path.relative_to(attempt_dir)
                                    ),
                                    "status": setup_detail.get("status"),
                                    "current_stage": setup_detail.get(
                                        "current_stage"
                                    ),
                                    "total_duration_ms": (
                                        (setup_detail.get("total") or {}).get(
                                            "duration_ms"
                                        )
                                    ),
                                    "completed_stage_duration_ms": setup_detail.get(
                                        "completed_stage_duration_ms"
                                    ),
                                    "host_pause_suspected": _nested_truthy_key(
                                        setup_detail, "host_pause_suspected"
                                    ),
                                    "docker_network_selection": (
                                        (
                                            setup_detail.get(
                                                "precheck_and_network"
                                            )
                                            or {}
                                        ).get("network_selection")
                                    ),
                                }
                            except Exception as checkpoint_exc:
                                environment_record["setup"] = {
                                    "artifact": str(
                                        setup_path.relative_to(attempt_dir)
                                    ),
                                    "read_error": (
                                        f"{type(checkpoint_exc).__name__}: "
                                        f"{checkpoint_exc}"
                                    ),
                                }
                        cleanup_path = evidence_dir / "cleanup.json"
                        if cleanup_path.is_file():
                            try:
                                cleanup_detail = json.loads(
                                    cleanup_path.read_text(encoding="utf-8")
                                )
                                environment_record["cleanup"] = {
                                    "artifact": str(
                                        cleanup_path.relative_to(attempt_dir)
                                    ),
                                    "started_at": cleanup_detail.get("started_at"),
                                    "finished_at": cleanup_detail.get("finished_at"),
                                    "duration_ms": cleanup_detail.get("duration_ms"),
                                    "errors": cleanup_detail.get(
                                        "cleanup_errors", []
                                    ),
                                    "warnings": cleanup_detail.get(
                                        "warnings", []
                                    ),
                                    "host_pause_suspected": _nested_truthy_key(
                                        cleanup_detail, "host_pause_suspected"
                                    ),
                                }
                            except Exception as checkpoint_exc:
                                environment_record["cleanup"] = {
                                    "artifact": str(
                                        cleanup_path.relative_to(attempt_dir)
                                    ),
                                    "read_error": (
                                        f"{type(checkpoint_exc).__name__}: "
                                        f"{checkpoint_exc}"
                                    ),
                                }
                        _write_json(
                            attempt_dir / "environment_attempts.json",
                            attempt["environment_attempts"],
                        )
                        _write_json(private_dir / "summary.json", summary)
                    if environment_error is not None:
                        cleanup_record = environment_record.get("cleanup")
                        cleanup_verified = (
                            isinstance(cleanup_record, dict)
                            and not cleanup_record.get("read_error")
                            and cleanup_record.get("errors") == []
                        )
                        retry_eligible = (
                            not environment_ready
                            and isinstance(
                                environment_error, (EnvironmentError, CommandError)
                            )
                            and cleanup_verified
                            and setup_retry_index
                            < self.config.pipeline.environment_setup_retries
                        )
                        environment_record["cleanup_verified_for_retry"] = (
                            cleanup_verified
                        )
                        environment_record["retry_same_candidate"] = retry_eligible
                        if retry_eligible:
                            environment_record["retry_reason"] = (
                                "environment_setup_failed_before_candidate_evaluation"
                            )
                            setup_retry_index += 1
                            attempt["environment_setup_retries"] += 1
                            _write_json(
                                attempt_dir / "environment_attempts.json",
                                attempt["environment_attempts"],
                            )
                            _write_json(private_dir / "summary.json", summary)
                            continue
                        if environment_ready:
                            environment_record["retry_reason"] = (
                                "candidate_evaluation_had_started"
                            )
                        elif not cleanup_verified:
                            environment_record["retry_reason"] = (
                                "cleanup_not_verified"
                            )
                        else:
                            environment_record["retry_reason"] = (
                                "environment_setup_retry_limit_reached"
                            )
                        _write_json(
                            attempt_dir / "environment_attempts.json",
                            attempt["environment_attempts"],
                        )
                        _write_json(private_dir / "summary.json", summary)
                        raise environment_error
                    if evaluation_finished is None:
                        raise EnvironmentError(
                            "Candidate environment ended before evaluation started"
                        )
                    evaluation["stage_timings_ms"]["environment_cleanup"] = round(
                        (cluster_finished - evaluation_finished) * 1000
                    )
                    evaluation["stage_timings_ms"]["cluster_total"] = round(
                        (cluster_finished - cluster_started) * 1000
                    )
                    _write_json(
                        attempt_dir / "candidate_evaluations.json",
                        attempt["candidate_evaluations"],
                    )
                    _write_json(private_dir / "summary.json", summary)
                    if (
                        evaluation["result"] == "candidate_api_loss"
                        and evaluation_index < maximum_confirmations
                    ):
                        evaluation_index += 1
                        setup_retry_index = 0
                        continue
                    candidate_evaluation = evaluation
                    break

                if candidate_evaluation is None:
                    raise EnvironmentError(
                        "Candidate evaluation ended without a terminal observation"
                    )

                for field in (
                    "deployment",
                    "deployment_failure_api_readiness",
                    "runtime",
                    "runtime_failure_api_readiness",
                    "execution_gate",
                    "execution_failure_api_readiness",
                    "post_execution",
                ):
                    attempt[field] = candidate_evaluation.get(field)
                if candidate_evaluation["evidence_dir"] != attempt_dir.name:
                    artifact_names = {
                        "deployment": "deployment.json",
                        "deployment_failure_api_readiness": (
                            "deployment_failure_api_readiness.json"
                        ),
                        "runtime": "runtime.json",
                        "runtime_failure_api_readiness": (
                            "runtime_failure_api_readiness.json"
                        ),
                        "execution_gate": "execution_gate.json",
                        "execution_failure_api_readiness": (
                            "execution_failure_api_readiness.json"
                        ),
                        "post_execution": "post_execution.json",
                    }
                    for field, filename in artifact_names.items():
                        value = candidate_evaluation.get(field)
                        if value is not None:
                            _write_json(attempt_dir / filename, value)

                execution = candidate_evaluation.get("execution_gate")
                if isinstance(execution, dict):
                    summary["execution_gate_public"] = {
                        "status": execution.get("status"),
                        "checks": len(execution.get("checks", [])),
                        "failures": len(execution.get("failures", [])),
                        "repair_on_failure": execution.get(
                            "repair_on_failure", True
                        ),
                        "task_specific": execution.get("task_specific", False),
                    }
                post = candidate_evaluation.get("post_execution")
                if isinstance(post, dict):
                    summary["post_execution_public"] = self._public_post(post)

                def record_validator_ground_truth(
                    candidate_correct: bool | None,
                    basis: str,
                    reason: str | None = None,
                ) -> None:
                    if validation_decision_satisfies is None:
                        return
                    attempt["validator_ground_truth"] = _validator_ground_truth(
                        satisfies=validation_decision_satisfies,
                        candidate_correct=candidate_correct,
                        basis=basis,
                        reason=reason,
                    )
                    attempt["validator_outcome_agreement"] = attempt[
                        "validator_ground_truth"
                    ]
                    _write_json(
                        attempt_dir / "validator_ground_truth.json",
                        attempt["validator_ground_truth"],
                    )
                    _write_json(
                        attempt_dir / "validator_outcome_agreement.json",
                        attempt["validator_outcome_agreement"],
                    )

                evaluation_result = candidate_evaluation["result"]
                if evaluation_result == "candidate_api_loss":
                    attempt["result"] = "confirmed_candidate_api_disruption"
                    record_validator_ground_truth(
                        False, "confirmed_repeated_candidate_api_disruption"
                    )
                    if index < self.config.pipeline.max_regenerations:
                        repair_trigger = "candidate_api_disruption"
                        api_disruption_feedback = {
                            "message": (
                                "The unchanged candidate repeatedly made a fresh, "
                                "previously healthy Kubernetes API unavailable. "
                                "Repair candidate behavior that can disrupt the cluster."
                            ),
                            "candidate_sha256": _sha256_text(syntax.candidate),
                            "evaluations": [
                                {
                                    "api_loss_stage": item.get("api_loss_stage"),
                                    "error": item.get("error"),
                                    "deployment": item.get("deployment"),
                                    "deployment_failure_api_readiness": item.get(
                                        "deployment_failure_api_readiness"
                                    ),
                                    "runtime_failure_api_readiness": item.get(
                                        "runtime_failure_api_readiness"
                                    ),
                                    "execution_failure_api_readiness": item.get(
                                        "execution_failure_api_readiness"
                                    ),
                                }
                                for item in attempt["candidate_evaluations"]
                            ],
                        }
                        repair_failure = queue_regeneration(
                            attempt,
                            attempt_dir,
                            trigger=repair_trigger,
                            failure_kind="candidate_api_disruption",
                            detail=api_disruption_feedback,
                            evidence_artifacts=[
                                "candidate_evaluations.json",
                                "deployment.json",
                                "deployment_failure_api_readiness.json",
                                "runtime_failure_api_readiness.json",
                                "execution_failure_api_readiness.json",
                            ],
                        )
                        continue
                    summary["status"] = "candidate_failed"
                    terminal = True
                    break

                if evaluation_result == "deployment_error":
                    attempt["result"] = "deployment_error"
                    record_validator_ground_truth(False, "deployment_error")
                    if index < self.config.pipeline.max_regenerations:
                        repair_trigger = "deployment"
                        repair_failure = queue_regeneration(
                            attempt,
                            attempt_dir,
                            trigger=repair_trigger,
                            failure_kind="deployment",
                            detail={
                                "message": (
                                    "kubectl apply rejected the candidate while the "
                                    "Kubernetes API remained healthy. Repair the "
                                    "reported candidate manifest error."
                                ),
                                "apply_command": candidate_evaluation["deployment"],
                                "api_readiness_after_failure": candidate_evaluation.get(
                                    "deployment_failure_api_readiness"
                                ),
                            },
                            evidence_artifacts=[
                                "deployment.json",
                                "deployment_failure_api_readiness.json",
                            ],
                        )
                        continue
                    summary["status"] = "candidate_failed"
                    terminal = True
                    break

                if evaluation_result == "runtime_error":
                    attempt["result"] = "runtime_error"
                    record_validator_ground_truth(False, "runtime_error")
                    if index < self.config.pipeline.max_regenerations:
                        repair_trigger = "runtime"
                        repair_failure = queue_regeneration(
                            attempt,
                            attempt_dir,
                            trigger=repair_trigger,
                            failure_kind="runtime",
                            detail=candidate_evaluation["runtime"],
                            evidence_artifacts=["runtime.json"],
                        )
                        continue
                    summary["status"] = "candidate_failed"
                    terminal = True
                    break

                if evaluation_result == "execution_gate_error":
                    attempt["result"] = "execution_gate_error"
                    record_validator_ground_truth(False, "execution_gate_error")
                    if index < self.config.pipeline.max_regenerations:
                        repair_trigger = "execution_gate"
                        repair_failure = queue_regeneration(
                            attempt,
                            attempt_dir,
                            trigger=repair_trigger,
                            failure_kind="execution_gate",
                            detail=candidate_evaluation["execution_gate"],
                            evidence_artifacts=["execution_gate.json"],
                        )
                        continue
                    summary["status"] = "candidate_failed"
                    terminal = True
                    break

                if evaluation_result == "infrastructure_error":
                    attempt["result"] = "evaluation_infrastructure_error"
                    record_validator_ground_truth(
                        None,
                        "evaluation_infrastructure_error",
                        candidate_evaluation.get("error"),
                    )
                    summary["status"] = "infrastructure_error"
                    summary["fatal_error"] = candidate_evaluation.get(
                        "error", "Candidate evaluation infrastructure failed"
                    )
                    summary["failure_origin"] = "environment"
                    summary["failure_stage"] = "candidate_evaluation"
                    terminal = True
                    break

                if evaluation_result == "accepted":
                    attempt["result"] = "accepted"
                    record_validator_ground_truth(True, "hidden_oracle_passed")
                    summary["status"] = "completed"
                    terminal = True
                    break

                if evaluation_result == "specification_discrepancy":
                    attempt["result"] = "specification_discrepancy"
                    record_validator_ground_truth(False, "hidden_oracle_failed")
                    summary["status"] = "specification_discrepancy"
                    terminal = True
                    break

                attempt["result"] = "post_verifier_infrastructure_error"
                record_validator_ground_truth(
                    None,
                    "post_verifier_infrastructure_error",
                    candidate_evaluation.get("error"),
                )
                summary["status"] = "infrastructure_error"
                summary["fatal_error"] = candidate_evaluation.get(
                    "error", "Hidden specification verifier failed"
                )
                summary["failure_origin"] = "hidden_evaluator"
                summary["failure_stage"] = "hidden_specification_oracle"
                terminal = True
                break

            if not terminal:
                summary["status"] = "candidate_failed"
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            summary["fatal_error"] = "Run interrupted by user"
            summary["failure_origin"] = "user"
            summary["failure_stage"] = active_role
            if attempts and attempts[-1]["result"] == "running":
                attempts[-1]["result"] = "interrupted"
                if current_attempt_dir is not None:
                    _write_json(
                        current_attempt_dir / "interruption.json",
                        {"status": "interrupted", "message": "Run interrupted by user"},
                    )
        except ProviderError as exc:
            summary["status"] = "provider_error"
            summary["fatal_error"] = str(exc)
            provider_stage = (
                "ai_pre_validation" if active_role == role_validator else "generation"
            )
            summary["failure_origin"] = "provider"
            summary["failure_stage"] = provider_stage
            if exc.audit_result is not None:
                audit = exc.audit_result
                record_model_result(active_role, audit)
                rejected = {
                    **self._model_audit(
                        audit, active_system_prompt, active_user_prompt
                    ),
                    "validation_error": str(exc),
                }
                if attempts and current_attempt_dir is not None:
                    attempt = attempts[-1]
                    if active_role == role_validator:
                        (current_attempt_dir / "ai_validator_raw_response.txt").write_text(
                            audit.raw_text, encoding="utf-8"
                        )
                        attempt["ai_pre_validation"] = {
                            "enabled": True,
                            "feedback_mode": self.config.ai_validator.feedback_mode,
                            "status": "provider_response_rejected",
                            "model_call": rejected,
                        }
                        summary["ai_pre_validation_public"] = {
                            "enabled": True,
                            "feedback_mode": self.config.ai_validator.feedback_mode,
                            "status": "provider_response_rejected",
                        }
                        _write_json(
                            current_attempt_dir / "ai_pre_validation.json",
                            attempt["ai_pre_validation"],
                        )
                        attempt["result"] = "ai_validator_provider_error"
                    else:
                        (current_attempt_dir / "raw_response.txt").write_text(
                            audit.raw_text, encoding="utf-8"
                        )
                        attempt["generation"] = rejected
                        _write_json(
                            current_attempt_dir / "generation.json", rejected
                        )
                        attempt["result"] = "provider_response_rejected"
            else:
                terminal_transport_by_role[active_role] += exc.transport_attempts
                if attempts and attempts[-1]["result"] == "running":
                    attempts[-1]["result"] = (
                        "ai_validator_provider_error"
                        if active_role == role_validator
                        else "generator_provider_error"
                    )
            unknown_cost_by_role[active_role] += exc.unknown_cost_attempts
            summary["provider_failure"] = {
                "role": active_role,
                "stage": provider_stage,
                "attempt": attempts[-1].get("attempt") if attempts else None,
                "transport_attempts": exc.transport_attempts,
                "transport_attempt_log": list(exc.transport_attempt_log),
                "unknown_cost_attempts": exc.unknown_cost_attempts,
            }
        except AiValidationError as exc:
            summary["status"] = "validator_error"
            summary["fatal_error"] = str(exc)
            summary["failure_origin"] = "validator"
            summary["failure_stage"] = "ai_pre_validation"
            if attempts and current_attempt_dir is not None:
                attempt = attempts[-1]
                attempt["result"] = "ai_validator_error"
                existing = attempt.get("ai_pre_validation") or {}
                attempt["ai_pre_validation"] = {
                    **existing,
                    "enabled": True,
                    "feedback_mode": self.config.ai_validator.feedback_mode,
                    "status": "validator_error",
                    "error": str(exc),
                }
                summary["ai_pre_validation_public"] = {
                    "enabled": True,
                    "feedback_mode": self.config.ai_validator.feedback_mode,
                    "status": "validator_error",
                }
                _write_json(
                    current_attempt_dir / "ai_pre_validation.json",
                    attempt["ai_pre_validation"],
                )
        except (EnvironmentError, CommandError) as exc:
            summary["status"] = "infrastructure_error"
            summary["fatal_error"] = str(exc)
            summary["failure_origin"] = "environment"
            summary["failure_stage"] = (
                "candidate_environment" if attempts else "environment_preflight"
            )
            if attempts and attempts[-1]["result"] == "running":
                attempts[-1]["result"] = "environment_error"
                attempts[-1]["infrastructure_error"] = str(exc)
                if current_attempt_dir is not None:
                    _write_json(
                        current_attempt_dir / "infrastructure_error.json",
                        {"error": str(exc)},
                    )
        except Exception as exc:
            summary["status"] = "infrastructure_error"
            summary["fatal_error"] = f"{type(exc).__name__}: {exc}"
            summary["failure_origin"] = "pipeline"
            summary["failure_stage"] = active_role
            traceback_text = traceback.format_exc()
            traceback_path = private_dir / "pipeline_traceback.txt"
            traceback_path.write_text(traceback_text, encoding="utf-8")
            summary["private_traceback_artifact"] = traceback_path.name
            if attempts and attempts[-1]["result"] == "running":
                attempts[-1]["result"] = "pipeline_error"
                attempts[-1]["infrastructure_error"] = summary["fatal_error"]
                if current_attempt_dir is not None:
                    (current_attempt_dir / "pipeline_traceback.txt").write_text(
                        traceback_text, encoding="utf-8"
                    )
                    _write_json(
                        current_attempt_dir / "infrastructure_error.json",
                        {"error": summary["fatal_error"]},
                    )
        finally:
            finished_marker = _utc_now()
            finished_monotonic = time.monotonic()
            for attempt in attempts:
                if "finished_at" not in attempt:
                    attempt["finished_at"] = finished_marker
                    attempt_started = attempt_start_times.get(
                        int(attempt.get("attempt", 0))
                    )
                    if attempt_started is not None:
                        attempt["duration_ms"] = round(
                            (finished_monotonic - attempt_started) * 1000
                        )
                if attempt.get("result") == "running":
                    attempt["result"] = "pipeline_terminated_without_result"
            if self.key_context_supplier:
                key_after_started = time.monotonic()
                try:
                    summary["openrouter_key_after"] = self.key_context_supplier()
                except Exception as exc:
                    summary["openrouter_key_after_error"] = str(exc)
                finally:
                    summary["stage_timings_ms"]["openrouter_key_after"] = round(
                        (time.monotonic() - key_after_started) * 1000
                    )
            summary["unknown_cost_failures"] = sum(unknown_cost_by_role.values())
            summary["usage_totals"] = self._usage_totals(
                model_results,
                terminal_transport_attempts=sum(
                    terminal_transport_by_role.values()
                ),
                unknown_cost_failures=summary["unknown_cost_failures"],
            )
            summary["usage_by_role"] = {
                role: self._usage_totals(
                    results,
                    terminal_transport_attempts=terminal_transport_by_role[role],
                    unknown_cost_failures=unknown_cost_by_role[role],
                )
                for role, results in results_by_role.items()
            }
            summary["finished_at"] = finished_marker
            summary["duration_ms"] = round((time.monotonic() - started) * 1000)
            finalization_timings: dict[str, int] = {}
            analysis_started = time.monotonic()
            try:
                summary["analysis_record"] = build_analysis_record(summary)
                _write_json(
                    private_dir / "analysis_record.json", summary["analysis_record"]
                )
            except Exception as exc:
                summary["analysis_record_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                finalization_timings["analysis_record"] = round(
                    (time.monotonic() - analysis_started) * 1000
                )
            integrity_started = time.monotonic()
            try:
                integrity = _artifact_integrity(private_dir)
                _write_json(private_dir / "artifact_integrity.json", integrity)
                summary["artifact_integrity"] = {
                    "artifact": "artifact_integrity.json",
                    "file_count": integrity["file_count"],
                    "aggregate_sha256": integrity["aggregate_sha256"],
                }
            except Exception as exc:
                summary["artifact_integrity_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                finalization_timings["artifact_integrity"] = round(
                    (time.monotonic() - integrity_started) * 1000
                )
            summary["finalization_timings_ms"] = finalization_timings
            _write_json(private_dir / "summary.json", summary)
        return PipelineRun(summary=summary, run_dir=run_dir)
