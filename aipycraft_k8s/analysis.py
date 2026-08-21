from __future__ import annotations

from collections import Counter
from typing import Any


_REGENERATION_RESULTS = {
    "incomplete_generation",
    "yaml_syntax_error",
    "ai_pre_validation_rejected",
    "deployment_error",
    "runtime_error",
    "execution_gate_error",
    "confirmed_candidate_api_disruption",
}

_PRE_DEPLOYMENT_TRIGGERS = {
    "generation_incomplete",
    "yaml_syntax",
    "ai_validator",
}

_EXECUTION_TRIGGERS = {
    "deployment",
    "runtime",
    "execution_gate",
    "candidate_api_disruption",
}


def _counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _failure_label(failure: dict[str, Any]) -> str:
    for key in ("type", "reason", "kind", "state"):
        value = failure.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unspecified"


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _nested_truthy_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if value.get(key) is True:
            return True
        return any(_nested_truthy_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_nested_truthy_key(item, key) for item in value)
    return False


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
        str(attempt.get("regeneration_trigger"))
        for attempt in attempts
        if attempt.get("regeneration_trigger")
    ]
    regeneration_source_results = [
        str(attempt.get("result", "unknown"))
        for attempt in attempts
        if attempt.get("regeneration_trigger")
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
    runtime_container_log_records = 0
    execution_gate_runs = 0
    hidden_verifier_runs = 0
    hidden_passed = 0
    hidden_failed = 0
    hidden_total = 0
    init_diagnostic_pods = 0
    init_diagnostic_containers = 0
    pod_diagnostic_pods = 0
    pod_diagnostic_containers = 0
    main_diagnostic_containers = 0
    cluster_attempts = 0
    candidate_api_loss_events = 0
    api_loss_confirmation_replays = 0
    validator_ground_truth_statuses: list[str] = []
    validator_ground_truth_bases: list[str] = []
    validator_classifications: list[str] = []
    validator_scored_predicted_defects: list[bool] = []
    validator_valid_verdicts: list[bool] = []
    validator_classifications_by_attempt: dict[str, list[str]] = {}
    feedback_utf8_bytes = 0
    feedback_source_utf8_bytes = 0
    feedback_omitted_utf8_bytes = 0
    feedback_truncated = 0
    stage_timing_totals: Counter[str] = Counter()
    environment_attempt_statuses: list[str] = []
    environment_setup_statuses: list[str] = []
    environment_attempts_started = 0
    environment_cleanup_failures = 0
    cleanup_verified_environment_retries = 0
    docker_network_subnets: list[str] = []
    docker_network_selection_bases: list[str] = []
    host_pause_suspected = False
    semantic_diff_added_resources = 0
    semantic_diff_removed_resources = 0
    semantic_diff_modified_resources = 0
    semantic_diff_changed_field_paths = 0

    for stage, duration in (summary.get("stage_timings_ms", {}) or {}).items():
        if (
            isinstance(stage, str)
            and isinstance(duration, (int, float))
            and not isinstance(duration, bool)
        ):
            stage_timing_totals[stage] += round(duration)

    for attempt in attempts:
        semantic_diff = attempt.get("candidate_semantic_diff")
        if isinstance(semantic_diff, dict) and semantic_diff.get("basis") != "initial_candidate":
            semantic_diff_added_resources += len(
                semantic_diff.get("added_resources", []) or []
            )
            semantic_diff_removed_resources += len(
                semantic_diff.get("removed_resources", []) or []
            )
            semantic_diff_modified_resources += len(
                semantic_diff.get("modified_resources", []) or []
            )
            semantic_diff_changed_field_paths += int(
                semantic_diff.get("changed_field_path_count") or 0
            )
        environment_records = attempt.get("environment_attempts", []) or []
        if isinstance(environment_records, list):
            for environment_record in environment_records:
                if not isinstance(environment_record, dict):
                    continue
                environment_attempts_started += 1
                environment_status = environment_record.get("status")
                if isinstance(environment_status, str):
                    environment_attempt_statuses.append(environment_status)
                setup_record = environment_record.get("setup")
                if isinstance(setup_record, dict):
                    setup_status = setup_record.get("status")
                    if isinstance(setup_status, str):
                        environment_setup_statuses.append(setup_status)
                    network_selection = setup_record.get("docker_network_selection")
                    if isinstance(network_selection, dict):
                        selected_subnet = network_selection.get("selected_subnet")
                        selection_basis = network_selection.get("selection_basis")
                        if isinstance(selected_subnet, str):
                            docker_network_subnets.append(selected_subnet)
                        if isinstance(selection_basis, str):
                            docker_network_selection_bases.append(selection_basis)
                cleanup_record = environment_record.get("cleanup")
                if isinstance(cleanup_record, dict):
                    errors = cleanup_record.get("errors", []) or []
                    if isinstance(errors, list) and errors:
                        environment_cleanup_failures += 1
                cleanup_verified_environment_retries += int(
                    environment_record.get("retry_same_candidate") is True
                )
                host_pause_suspected = host_pause_suspected or _nested_truthy_key(
                    environment_record, "host_pause_suspected"
                )

        pre_execution = attempt.get("pre_execution")
        if isinstance(pre_execution, dict) and pre_execution.get("valid") is True:
            yaml_valid_candidates += 1

        validation = attempt.get("ai_pre_validation")
        if isinstance(validation, dict):
            status = validation.get("status")
            if isinstance(status, str):
                validator_statuses.append(status)
            decision = validation.get("decision")
            if isinstance(decision, dict) and isinstance(
                decision.get("satisfies"), bool
            ):
                validator_valid_verdicts.append(bool(decision["satisfies"]))

        regeneration_feedback = attempt.get("regeneration_feedback")
        if isinstance(regeneration_feedback, dict):
            metrics = regeneration_feedback.get("prompt_feedback_metrics", {}) or {}
            if isinstance(metrics, dict):
                feedback_utf8_bytes += int(metrics.get("utf8_bytes") or 0)
                feedback_source_utf8_bytes += int(
                    metrics.get("source_utf8_bytes") or 0
                )
                counts = metrics.get("truncation_counts", {}) or {}
                explicitly_omitted = 0
                if isinstance(counts, dict):
                    explicitly_omitted = int(
                        counts.get("omitted_utf8_bytes") or 0
                    )
                # Profile compaction (for example, shortening a long container
                # log) predates the final whole-envelope truncation and therefore
                # does not always populate omitted_utf8_bytes.  Preserve an
                # aggregate lower bound from the exact source/prompt byte sizes so
                # those lossy repair payloads are not reported as omitting zero.
                source_bytes = int(metrics.get("source_utf8_bytes") or 0)
                prompt_bytes = int(metrics.get("utf8_bytes") or 0)
                net_source_reduction = max(source_bytes - prompt_bytes, 0)
                feedback_omitted_utf8_bytes += max(
                    explicitly_omitted, net_source_reduction
                )
                feedback_truncated += int(bool(metrics.get("truncated")))

        for stage, duration in (attempt.get("stage_timings_ms", {}) or {}).items():
            if (
                isinstance(stage, str)
                and isinstance(duration, (int, float))
                and not isinstance(duration, bool)
            ):
                stage_timing_totals[stage] += round(duration)

        truth = attempt.get("validator_ground_truth")
        if isinstance(truth, dict):
            status = truth.get("status")
            basis = truth.get("basis")
            classification = truth.get("classification")
            if isinstance(status, str):
                validator_ground_truth_statuses.append(status)
            if isinstance(basis, str):
                validator_ground_truth_bases.append(basis)
            if isinstance(classification, str):
                validator_classifications.append(classification)
                validator_classifications_by_attempt.setdefault(
                    str(attempt.get("attempt")), []
                ).append(classification)
            predicted_defect = truth.get("validator_predicted_defect")
            if isinstance(predicted_defect, bool):
                if isinstance(classification, str):
                    validator_scored_predicted_defects.append(predicted_defect)

        raw_evaluations = attempt.get("candidate_evaluations")
        if isinstance(raw_evaluations, list) and raw_evaluations:
            evaluations = [item for item in raw_evaluations if isinstance(item, dict)]
        else:
            # Backward-compatible analysis for historical artifacts.
            evaluations = [attempt]
        api_loss_confirmation_replays += int(
            attempt.get("api_loss_confirmation_replays") or 0
        )
        for evaluation in evaluations:
            host_pause_suspected = host_pause_suspected or _nested_truthy_key(
                evaluation, "host_pause_suspected"
            )
            for stage, duration in (
                evaluation.get("stage_timings_ms", {}) or {}
            ).items():
                if (
                    isinstance(stage, str)
                    and isinstance(duration, (int, float))
                    and not isinstance(duration, bool)
                ):
                    stage_timing_totals[stage] += round(duration)

            if evaluation.get("result") == "candidate_api_loss":
                candidate_api_loss_events += 1

            deployment = evaluation.get("deployment")
            if isinstance(deployment, dict):
                cluster_attempts += 1
                deployment_attempts += 1
                if deployment.get("returncode") == 0:
                    deployment_successes += 1

            runtime = evaluation.get("runtime")
            if isinstance(runtime, dict):
                runtime_observations += 1
                logs = runtime.get("container_logs", []) or []
                if isinstance(logs, list):
                    runtime_container_log_records += len(logs)
                for failure in runtime.get("failures", []) or []:
                    if isinstance(failure, dict):
                        runtime_failure_labels.append(_failure_label(failure))

            execution = evaluation.get("execution_gate")
            if isinstance(execution, dict):
                execution_gate_runs += 1
                status = execution.get("status")
                if isinstance(status, str):
                    execution_statuses.append(status)
                for failure in execution.get("failures", []) or []:
                    if not isinstance(failure, dict):
                        continue
                    execution_failure_labels.append(_failure_label(failure))
                    diagnostics = failure.get("pod_diagnostics")
                    if isinstance(diagnostics, dict):
                        pods = diagnostics.get("pods", []) or []
                        if isinstance(pods, list):
                            pod_diagnostic_pods += len(pods)
                            for pod in pods:
                                if not isinstance(pod, dict):
                                    continue
                                containers = pod.get("containers", []) or []
                                if not isinstance(containers, list):
                                    continue
                                typed_containers = [
                                    item for item in containers if isinstance(item, dict)
                                ]
                                pod_diagnostic_containers += len(typed_containers)
                                init_count = sum(
                                    item.get("container_type") == "init"
                                    for item in typed_containers
                                )
                                main_count = sum(
                                    item.get("container_type") == "main"
                                    for item in typed_containers
                                )
                                init_diagnostic_containers += init_count
                                main_diagnostic_containers += main_count
                                if init_count:
                                    init_diagnostic_pods += 1
                        continue

                    # Schema-2 compatibility for historical gate artifacts.
                    diagnostics = failure.get("init_container_diagnostics")
                    if isinstance(diagnostics, dict):
                        pods = diagnostics.get("pods", []) or []
                        if isinstance(pods, list):
                            pod_diagnostic_pods += len(pods)
                            init_diagnostic_pods += len(pods)
                            old_init_count = sum(
                                len(pod.get("init_containers", []) or [])
                                for pod in pods
                                if isinstance(pod, dict)
                            )
                            pod_diagnostic_containers += old_init_count
                            init_diagnostic_containers += old_init_count

            post = evaluation.get("post_execution")
            if isinstance(post, dict):
                hidden_verifier_runs += 1
                hidden_passed += int(post.get("passed") or 0)
                hidden_failed += int(post.get("failed") or 0)
                hidden_total += int(post.get("total") or 0)
                for outcome in post.get("outcomes", []) or []:
                    if (
                        not isinstance(outcome, dict)
                        or outcome.get("passed") is not False
                    ):
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

    classification_counts = Counter(validator_classifications)
    true_positive = classification_counts["true_positive"]
    false_positive = classification_counts["false_positive"]
    false_negative = classification_counts["false_negative"]
    true_negative = classification_counts["true_negative"]

    return {
        "schema_version": 3,
        "run_id": summary.get("run_id"),
        "task_id": summary.get("task_id"),
        "status": summary.get("status"),
        "terminal_attempt_result": attempt_results[-1] if attempt_results else None,
        "mode": summary.get("mode"),
        "treatment_id": summary.get("treatment_id"),
        "initial_candidate_source": summary.get("initial_candidate_source"),
        "candidate_bank_id": (summary.get("candidate_bank") or {}).get(
            "candidate_id"
        ),
        "candidate_bank_record_sha256": (summary.get("candidate_bank") or {}).get(
            "record_sha256"
        ),
        "candidate_bank_raw_response_sha256": (
            summary.get("candidate_bank") or {}
        ).get("raw_response_sha256"),
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
        "candidate_api_loss_confirmation_replays_per_candidate": summary.get(
            "candidate_api_loss_confirmation_replays"
        ),
        "environment_setup_retries_per_environment": summary.get(
            "environment_setup_retries"
        ),
        "runtime_observation_seconds": summary.get("runtime_observation_seconds"),
        "runtime_poll_seconds": summary.get("runtime_poll_seconds"),
        "command_timeout_seconds": summary.get("command_timeout_seconds"),
        "kind_create_timeout_seconds": summary.get("kind_create_timeout_seconds"),
        "validator_enabled": (summary.get("ai_validator", {}) or {}).get("enabled"),
        "validator_feedback_mode": (summary.get("ai_validator", {}) or {}).get(
            "feedback_mode"
        ),
        "validator_shadow_mode": (summary.get("ai_validator", {}) or {}).get(
            "shadow_mode"
        ),
        "validator_intervention_active": (
            summary.get("ai_validator", {}) or {}
        ).get("intervention_active"),
        "attempts": len(attempts),
        "regenerations": int(summary.get("regenerations") or 0),
        "duplicate_generation_responses": int(
            summary.get("duplicate_generation_responses") or 0
        ),
        "attempt_result_counts": _counts(attempt_results),
        "regeneration_trigger_counts": _counts(actual_regeneration_triggers),
        "regeneration_source_result_counts": _counts(regeneration_source_results),
        "regeneration_eligible_failure_counts": _counts(
            eligible_regeneration_results
        ),
        "pre_deployment_regenerations": sum(
            value in _PRE_DEPLOYMENT_TRIGGERS
            for value in actual_regeneration_triggers
        ),
        "execution_feedback_regenerations": sum(
            value in _EXECUTION_TRIGGERS for value in actual_regeneration_triggers
        ),
        "validator_rejections": sum(not value for value in validator_valid_verdicts),
        "validator_negative_verdicts": sum(
            not value for value in validator_valid_verdicts
        ),
        "validator_interventions": sum(
            value == "ai_pre_validation_rejected" for value in attempt_results
        ),
        "yaml_valid_candidates": yaml_valid_candidates,
        "validator_status_counts": _counts(validator_statuses),
        "validator_ground_truth_status_counts": _counts(
            validator_ground_truth_statuses
        ),
        "validator_ground_truth_basis_counts": _counts(
            validator_ground_truth_bases
        ),
        "validator_classification_counts": _counts(validator_classifications),
        "validator_scored_candidates": len(validator_classifications),
        "validator_unscored_candidates": max(
            len(validator_valid_verdicts) - len(validator_classifications), 0
        ),
        "validator_valid_verdict_candidates": len(validator_valid_verdicts),
        "validator_ground_truth_coverage": _ratio(
            len(validator_classifications), len(validator_valid_verdicts)
        ),
        "validator_rejected_candidates_scored": sum(
            validator_scored_predicted_defects
        ),
        "validator_rejected_ground_truth_coverage": _ratio(
            sum(validator_scored_predicted_defects),
            sum(not satisfies for satisfies in validator_valid_verdicts),
        ),
        "validator_accepted_candidates_scored": sum(
            not value for value in validator_scored_predicted_defects
        ),
        "validator_accepted_ground_truth_coverage": _ratio(
            sum(not value for value in validator_scored_predicted_defects),
            sum(satisfies for satisfies in validator_valid_verdicts),
        ),
        "validator_observed_accuracy": _ratio(
            true_positive + true_negative, len(validator_classifications)
        ),
        "validator_observed_precision": _ratio(
            true_positive, true_positive + false_positive
        ),
        "validator_observed_recall": _ratio(
            true_positive, true_positive + false_negative
        ),
        "validator_observed_specificity": _ratio(
            true_negative, true_negative + false_positive
        ),
        "validator_observed_false_positive_rate": _ratio(
            false_positive, false_positive + true_negative
        ),
        "validator_observed_false_negative_rate": _ratio(
            false_negative, false_negative + true_positive
        ),
        "validator_metric_scope": (
            "candidate-level downstream outcome agreement; classifications do "
            "not establish that each cited defect was correct"
        ),
        "validator_candidate_outcome_agreement_counts": _counts(
            validator_classifications
        ),
        "validator_outcome_agreement_by_attempt": {
            key: _counts(values)
            for key, values in sorted(validator_classifications_by_attempt.items())
        },
        "validator_first_candidate_outcome_agreement": next(
            (
                attempt.get("validator_ground_truth", {}).get("classification")
                for attempt in attempts
                if attempt.get("attempt") == 1
                and isinstance(attempt.get("validator_ground_truth"), dict)
            ),
            None,
        ),
        "cluster_attempts": environment_attempts_started or cluster_attempts,
        "environment_attempts_started": (
            environment_attempts_started or cluster_attempts
        ),
        "environment_attempt_status_counts": _counts(
            environment_attempt_statuses
        ),
        "environment_setup_status_counts": _counts(environment_setup_statuses),
        "environment_cleanup_failures": environment_cleanup_failures,
        "cleanup_verified_environment_retries": (
            cleanup_verified_environment_retries
        ),
        "docker_network_subnet_counts": _counts(docker_network_subnets),
        "docker_network_selection_basis_counts": _counts(
            docker_network_selection_bases
        ),
        "host_pause_suspected": host_pause_suspected,
        "candidate_api_loss_events": candidate_api_loss_events,
        "api_loss_confirmation_replays": api_loss_confirmation_replays,
        "deployment_attempts": deployment_attempts,
        "deployment_successes": deployment_successes,
        "runtime_observations": runtime_observations,
        "runtime_container_log_records": runtime_container_log_records,
        "runtime_failure_counts": _counts(runtime_failure_labels),
        "execution_gate_runs": execution_gate_runs,
        "execution_gate_status_counts": _counts(execution_statuses),
        "execution_failure_counts": _counts(execution_failure_labels),
        "init_diagnostic_pods": init_diagnostic_pods,
        "init_diagnostic_containers": init_diagnostic_containers,
        "pod_diagnostic_pods": pod_diagnostic_pods,
        "pod_diagnostic_containers": pod_diagnostic_containers,
        "main_diagnostic_containers": main_diagnostic_containers,
        "hidden_verifier_reached": hidden_verifier_runs > 0,
        "hidden_verifier_runs": hidden_verifier_runs,
        "hidden_passed": hidden_passed,
        "hidden_failed": hidden_failed,
        "hidden_total": hidden_total,
        "hidden_failed_requirement_ids": sorted(hidden_failed_requirement_ids),
        "repair_feedback_utf8_bytes": feedback_utf8_bytes,
        "repair_feedback_source_utf8_bytes": feedback_source_utf8_bytes,
        "repair_feedback_omitted_utf8_bytes": feedback_omitted_utf8_bytes,
        "repair_feedback_payloads_truncated": feedback_truncated,
        "semantic_diff_added_resources": semantic_diff_added_resources,
        "semantic_diff_removed_resources": semantic_diff_removed_resources,
        "semantic_diff_modified_resources": semantic_diff_modified_resources,
        "semantic_diff_changed_field_paths": semantic_diff_changed_field_paths,
        "stage_timing_totals_ms": dict(sorted(stage_timing_totals.items())),
        "accepted_attempt": next(
            (
                attempt.get("attempt")
                for attempt in attempts
                if attempt.get("result") == "accepted"
            ),
            None,
        ),
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
        "transport_attempt_duration_ms": int(
            usage.get("transport_attempt_duration_ms") or 0
        ),
        "transport_retry_sleep_duration_ms": int(
            usage.get("transport_retry_sleep_duration_ms") or 0
        ),
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
        "failure_origin": summary.get("failure_origin"),
        "failure_stage": summary.get("failure_stage"),
        "provider_failure_role": (
            summary.get("provider_failure", {}) or {}
        ).get("role"),
        "provider_failure_stage": (
            summary.get("provider_failure", {}) or {}
        ).get("stage"),
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
        "source_state_sha256": (
            provenance.get("source_state", {}) or {}
        ).get("aggregate_sha256"),
    }
