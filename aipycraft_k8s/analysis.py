from __future__ import annotations

from collections import Counter
from typing import Any


_REGENERATION_RESULTS = {
    "incomplete_generation",
    "yaml_syntax_error",
    "ai_pre_validation_rejected",
}


def _counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _failure_label(failure: dict[str, Any]) -> str:
    for key in ("type", "reason", "kind", "state"):
        value = failure.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unspecified"


def build_analysis_record(summary: dict[str, Any]) -> dict[str, Any]:
    """Build one stable, graph-friendly row without discarding raw evidence."""

    attempts = [
        item for item in summary.get("attempts", []) if isinstance(item, dict)
    ]
    attempt_results = [str(item.get("result", "unknown")) for item in attempts]
    eligible_regeneration_results = [
        value for value in attempt_results if value in _REGENERATION_RESULTS
    ]
    actual_regeneration_triggers = [
        value
        for index, value in enumerate(attempt_results)
        if value in _REGENERATION_RESULTS and index < len(attempt_results) - 1
    ]
    validator_statuses: list[str] = []
    runtime_failure_labels: list[str] = []
    execution_statuses: list[str] = []
    execution_failure_labels: list[str] = []
    hidden_failed_requirement_ids: list[str] = []
    yaml_valid_candidates = 0
    deployment_attempts = 0
    deployment_successes = 0
    runtime_observations = 0
    execution_gate_runs = 0
    hidden_verifier_runs = 0
    hidden_passed = 0
    hidden_failed = 0
    hidden_total = 0
    init_diagnostic_pods = 0
    init_diagnostic_containers = 0

    for attempt in attempts:
        pre_execution = attempt.get("pre_execution")
        if isinstance(pre_execution, dict) and pre_execution.get("valid") is True:
            yaml_valid_candidates += 1

        validation = attempt.get("ai_pre_validation")
        if isinstance(validation, dict):
            status = validation.get("status")
            if isinstance(status, str):
                validator_statuses.append(status)

        deployment = attempt.get("deployment")
        if isinstance(deployment, dict):
            deployment_attempts += 1
            if deployment.get("returncode") == 0:
                deployment_successes += 1

        runtime = attempt.get("runtime")
        if isinstance(runtime, dict):
            runtime_observations += 1
            for failure in runtime.get("failures", []) or []:
                if isinstance(failure, dict):
                    runtime_failure_labels.append(_failure_label(failure))

        execution = attempt.get("execution_gate")
        if isinstance(execution, dict):
            execution_gate_runs += 1
            status = execution.get("status")
            if isinstance(status, str):
                execution_statuses.append(status)
            for failure in execution.get("failures", []) or []:
                if not isinstance(failure, dict):
                    continue
                execution_failure_labels.append(_failure_label(failure))
                diagnostics = failure.get("init_container_diagnostics")
                if not isinstance(diagnostics, dict):
                    continue
                pods = diagnostics.get("pods", []) or []
                init_diagnostic_pods += len(pods) if isinstance(pods, list) else 0
                if isinstance(pods, list):
                    init_diagnostic_containers += sum(
                        len(pod.get("init_containers", []) or [])
                        for pod in pods
                        if isinstance(pod, dict)
                    )

        post = attempt.get("post_execution")
        if isinstance(post, dict):
            hidden_verifier_runs += 1
            hidden_passed += int(post.get("passed") or 0)
            hidden_failed += int(post.get("failed") or 0)
            hidden_total += int(post.get("total") or 0)
            for outcome in post.get("outcomes", []) or []:
                if not isinstance(outcome, dict) or outcome.get("passed") is not False:
                    continue
                requirement_id = outcome.get("requirement_id")
                if isinstance(requirement_id, str):
                    hidden_failed_requirement_ids.append(requirement_id)

    usage = summary.get("usage_totals", {}) or {}
    generator_usage = (summary.get("usage_by_role", {}) or {}).get(
        "aipycraft_generator", {}
    ) or {}
    validator_usage = (summary.get("usage_by_role", {}) or {}).get(
        "ai_pre_validator", {}
    ) or {}
    provenance = summary.get("provenance", {}) or {}
    git = provenance.get("git", {}) or {}
    hidden_evaluator = provenance.get("hidden_evaluator", {}) or {}

    return {
        "schema_version": 1,
        "run_id": summary.get("run_id"),
        "task_id": summary.get("task_id"),
        "status": summary.get("status"),
        "terminal_attempt_result": attempt_results[-1] if attempt_results else None,
        "mode": summary.get("mode"),
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "duration_ms": summary.get("duration_ms"),
        "model": summary.get("model"),
        "provider_only": summary.get("provider_only"),
        "allow_fallbacks": summary.get("allow_fallbacks"),
        "reasoning_effort": summary.get("reasoning_effort"),
        "temperature": summary.get("temperature"),
        "max_output_tokens": summary.get("max_output_tokens"),
        "transport_retries": summary.get("transport_retries"),
        "max_regenerations": summary.get("max_regenerations"),
        "runtime_observation_seconds": summary.get("runtime_observation_seconds"),
        "runtime_poll_seconds": summary.get("runtime_poll_seconds"),
        "command_timeout_seconds": summary.get("command_timeout_seconds"),
        "kind_create_timeout_seconds": summary.get("kind_create_timeout_seconds"),
        "validator_enabled": (summary.get("ai_validator", {}) or {}).get("enabled"),
        "validator_feedback_mode": (summary.get("ai_validator", {}) or {}).get(
            "feedback_mode"
        ),
        "attempts": len(attempts),
        "regenerations": int(summary.get("regenerations") or 0),
        "duplicate_generation_responses": int(
            summary.get("duplicate_generation_responses") or 0
        ),
        "attempt_result_counts": _counts(attempt_results),
        "regeneration_trigger_counts": _counts(actual_regeneration_triggers),
        "regeneration_eligible_failure_counts": _counts(
            eligible_regeneration_results
        ),
        "yaml_valid_candidates": yaml_valid_candidates,
        "validator_status_counts": _counts(validator_statuses),
        "deployment_attempts": deployment_attempts,
        "deployment_successes": deployment_successes,
        "runtime_observations": runtime_observations,
        "runtime_failure_counts": _counts(runtime_failure_labels),
        "execution_gate_runs": execution_gate_runs,
        "execution_gate_status_counts": _counts(execution_statuses),
        "execution_failure_counts": _counts(execution_failure_labels),
        "init_diagnostic_pods": init_diagnostic_pods,
        "init_diagnostic_containers": init_diagnostic_containers,
        "hidden_verifier_reached": hidden_verifier_runs > 0,
        "hidden_verifier_runs": hidden_verifier_runs,
        "hidden_passed": hidden_passed,
        "hidden_failed": hidden_failed,
        "hidden_total": hidden_total,
        "hidden_failed_requirement_ids": sorted(hidden_failed_requirement_ids),
        "model_responses": int(usage.get("requests") or 0),
        "generator_responses": int(generator_usage.get("requests") or 0),
        "validator_responses": int(validator_usage.get("requests") or 0),
        "http_post_attempts": int(usage.get("http_post_attempts") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
        "cached_tokens": int(usage.get("cached_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "model_latency_ms": int(usage.get("latency_ms") or 0),
        "response_characters": int(usage.get("response_characters") or 0),
        "response_utf8_bytes": int(usage.get("response_utf8_bytes") or 0),
        "response_lines": int(usage.get("response_lines") or 0),
        "suspicious_provider_usage_responses": int(
            usage.get("suspicious_provider_usage_responses") or 0
        ),
        "cost_usd": str(usage.get("cost_usd") or "0"),
        "provider_cost_complete": bool(usage.get("provider_cost_complete")),
        "usage_complete": bool(usage.get("usage_complete")),
        "unobserved_billable_attempts": int(
            usage.get("unobserved_billable_attempts") or 0
        ),
        "generator_tokens": int(generator_usage.get("total_tokens") or 0),
        "generator_latency_ms": int(generator_usage.get("latency_ms") or 0),
        "generator_cost_usd": str(generator_usage.get("cost_usd") or "0"),
        "validator_tokens": int(validator_usage.get("total_tokens") or 0),
        "validator_latency_ms": int(validator_usage.get("latency_ms") or 0),
        "validator_cost_usd": str(validator_usage.get("cost_usd") or "0"),
        "fatal_error": summary.get("fatal_error"),
        "config_sha256": summary.get("config_sha256"),
        "environment_lock_sha256": summary.get("environment_lock_sha256"),
        "generate_prompt_sha256": summary.get("generate_prompt_sha256"),
        "repair_prompt_sha256": summary.get("repair_prompt_sha256"),
        "validator_prompt_sha256": summary.get("validator_prompt_sha256"),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "generic_execution_verifier_sha256": provenance.get(
            "generic_execution_verifier_sha256"
        ),
        "ai_pre_validator_sha256": provenance.get("ai_pre_validator_sha256"),
        "hidden_evaluator_core_sha256": hidden_evaluator.get("core_sha256"),
        "hidden_evaluator_suites_sha256": hidden_evaluator.get("suites_sha256"),
    }
