from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from task_authoring.config import load_env_file

from .commands import CommandError
from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config, treatment_id
from .environment import EnvironmentError, EnvironmentPreparer, IsolatedKindHarness
from .openrouter import (
    OpenRouterTextClient,
    ProviderError,
    ReplayTextClient,
    safe_key_budget_context,
)
from .pipeline import KubernetesAIPyCraftPipeline
from .tasks import TaskError, load_tasks


PAID_CONFIRMATION = "I_ACCEPT_PAID_OPENROUTER_CALLS"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, pre-validate, deploy, and evaluate Kubernetes YAML with "
            "bounded pre-execution and execution-feedback correction."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "plan", help="Show the frozen experiment settings; no network or Docker calls."
    )
    subparsers.add_parser(
        "prepare",
        help="Download checksum-locked harness files and pull images; no model calls.",
    )
    subparsers.add_parser(
        "doctor", help="Verify the prepared Docker/kind environment; no model calls."
    )
    subparsers.add_parser(
        "smoke",
        help="Create, test, and delete a real isolated cluster; no model calls.",
    )

    run = subparsers.add_parser("run", help="Run the paid OpenRouter pipeline.")
    run.add_argument("--task", default="all")
    run.add_argument("--confirm-paid-calls", required=True)

    replay = subparsers.add_parser(
        "replay", help="Use ordered local .txt responses instead of an API."
    )
    replay.add_argument("--task", default="all")
    replay.add_argument("--fixtures", type=Path, required=True)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _select_tasks(selection: str) -> list[Any]:
    tasks = load_tasks()
    if selection == "all":
        return [tasks[key] for key in sorted(tasks)]
    if selection not in tasks:
        raise TaskError(f"Unknown task {selection!r}; choose all or {sorted(tasks)}")
    return [tasks[selection]]


def _plan(config: Any) -> dict[str, Any]:
    lock = config.environment.lock
    max_generations = config.pipeline.max_regenerations + 1
    max_cluster_attempts = max_generations * (
        config.pipeline.candidate_api_loss_confirmation_replays + 1
    )
    max_validator_responses = max_generations if config.ai_validator.enabled else 0
    max_model_responses = max_generations + max_validator_responses
    return {
        "status": "planned",
        "paid_calls_made": False,
        "treatment_id": treatment_id(config),
        "model": config.api.model,
        "provider_only": list(config.api.provider_only),
        "allow_fallbacks": config.api.allow_fallbacks,
        "temperature": config.api.temperature,
        "reasoning_effort": config.api.reasoning_effort,
        "max_aipycraft_generations_per_task": max_generations,
        "max_ai_validator_responses_per_task": max_validator_responses,
        "max_model_responses_per_task": max_model_responses,
        "max_http_post_attempts_per_task": (
            max_model_responses * (config.api.transport_retries + 1)
        ),
        "max_candidate_clusters_per_task": max_cluster_attempts,
        "candidate_api_loss_confirmation_replays": (
            config.pipeline.candidate_api_loss_confirmation_replays
        ),
        "application_spend_cap_usd": None,
        "pricing_usd_per_million": {
            "input": str(config.api.input_usd_per_million),
            "output": str(config.api.output_usd_per_million),
        },
        "pre_execution_candidate_checks": [
            "YAML syntax composition only (deterministic parser)",
            (
                "public plaintext-to-YAML AI review"
                if config.ai_validator.enabled
                else "AI review disabled"
            ),
        ],
        "candidate_regeneration": {
            "eligible_triggers": [
                "incomplete generation",
                "YAML syntax error",
                "AI-validator rejection",
                "candidate-caused kubectl apply failure",
                "concrete candidate runtime failure",
                "generic execution-gate failure",
                "confirmed repeated candidate-associated Kubernetes API loss",
            ],
            "maximum_regenerations": config.pipeline.max_regenerations,
            "validator_repair_scope": (
                "fix cited defects and required dependent changes only"
                if config.ai_validator.feedback_mode == "detailed"
                else "negative verdict provides no cited defect details"
            ),
            "transport_retry_note": (
                "the separately logged transport retry resends the same request; "
                "it is not candidate regeneration"
            ),
        },
        "ai_validator": {
            "enabled": config.ai_validator.enabled,
            "shadow_mode": config.ai_validator.shadow_mode,
            "intervention_active": (
                config.ai_validator.enabled and not config.ai_validator.shadow_mode
            ),
            "model": config.api.model,
            "provider_only": list(config.api.provider_only),
            "feedback_mode": config.ai_validator.feedback_mode,
            "failure_threshold": (
                "clear missing or contradicted requirement only; runtime uncertainty "
                "alone passes"
            ),
            "rejection_policy": (
                "log verdict and deploy unchanged"
                if config.ai_validator.shadow_mode
                else "regenerate before deployment"
            ),
            "shadow_error_policy": (
                "log validator/provider error and continue unchanged candidate"
                if config.ai_validator.shadow_mode
                else "not applicable"
            ),
        },
        "generic_execution_gate": {
            "timing": "after apply and initial runtime observation",
            "wait_strategy": (
                "failed-aware polling for Jobs and controllers under one shared "
                "deadline; terminal failure stops the wait immediately"
            ),
            "checks": [
                "declared CronJobs complete when exercised once",
                "controllers and direct Jobs become operational",
                (
                    "selector-based Services obtain ready endpoints and expose a "
                    "listening TCP target"
                ),
                "standalone Pods become Ready or Succeeded",
            ],
            "rollout_failure_diagnostics": (
                "capture bounded main/init container commands, states, current and "
                "previous logs, Pod events, Service endpoints, and NetworkPolicy "
                "context before disposable probe cleanup"
            ),
            "failure_policy": (
                "persist full evidence privately and send an exact, whole-payload-"
                "bounded candidate-attributable subset for regeneration"
            ),
            "task_specific": False,
        },
        "deployment": "kubectl apply --validate=false with no namespace override",
        "analysis_artifact": (
            "private/analysis_record.json: stable per-run metrics row plus detailed "
            "summary and attempt artifacts"
        ),
        "post_execution_failure_policy": {
            "deployment_runtime_and_generic_execution": (
                "regenerate with bounded generic diagnostics when candidate-caused"
            ),
            "hidden_specification_oracle": (
                "log discrepancy; never send to the model or regenerate"
            ),
            "candidate_time_api_loss": (
                "replay the exact candidate once in a fresh cluster without a model "
                "call; regenerate only after repeated candidate-associated loss"
            ),
            "confirmed_infrastructure_failure": (
                "mark the trial inconclusive; never send it to the model"
            ),
        },
        "environment": {
            "kind": lock.kind_version,
            "kubernetes": lock.node_image,
            "calico": lock.calico_version,
            "internal_docker_network_per_attempt": True,
            "fresh_cluster_per_deployment_attempt": True,
            "configured_extra_mounts": False,
            "configured_extra_port_mappings": False,
            "api_server_host_published": False,
            "api_access": "docker_exec_only",
            "kind_create_timeout_seconds": max(
                300, config.pipeline.command_timeout_seconds * 3
            ),
        },
    }


def _aggregate_usage(results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "requests",
        "http_post_attempts",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "response_characters",
        "response_utf8_bytes",
        "response_lines",
        "suspicious_provider_usage_responses",
        "latency_ms",
        "unobserved_billable_attempts",
    )
    totals: dict[str, Any] = {
        field: sum(int(item["usage_totals"].get(field, 0)) for item in results)
        for field in fields
    }
    totals["cost_usd"] = str(
        sum(
            (Decimal(str(item["usage_totals"].get("cost_usd", "0"))) for item in results),
            Decimal("0"),
        )
    )
    totals["provider_cost_complete"] = all(
        bool(item["usage_totals"].get("provider_cost_complete")) for item in results
    )
    totals["usage_complete"] = all(
        bool(item["usage_totals"].get("usage_complete")) for item in results
    )
    return totals


def _exit_for(results: list[dict[str, Any]]) -> int:
    statuses = {item["status"] for item in results}
    if statuses == {"completed"}:
        return 0
    if "interrupted" in statuses:
        return 130
    if statuses & {"provider_error", "validator_error", "infrastructure_error"}:
        return 3
    return 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "plan":
            _print(_plan(config))
            return 0

        preparer = EnvironmentPreparer(
            config.environment,
            timeout_seconds=max(config.pipeline.command_timeout_seconds, 900),
        )
        if args.command == "prepare":
            _print(preparer.prepare())
            return 0
        if args.command == "doctor":
            _print(preparer.doctor())
            return 0
        if args.command == "smoke":
            harness = IsolatedKindHarness(
                config.environment,
                command_timeout_seconds=config.pipeline.command_timeout_seconds,
                max_attempts=(
                    (config.pipeline.max_regenerations + 1)
                    * (
                        config.pipeline.candidate_api_loss_confirmation_replays
                        + 1
                    )
                ),
            )
            _print(harness.smoke())
            return 0

        tasks = _select_tasks(args.task)
        harness = IsolatedKindHarness(
            config.environment,
            command_timeout_seconds=config.pipeline.command_timeout_seconds,
            max_attempts=(
                (config.pipeline.max_regenerations + 1)
                * (
                    config.pipeline.candidate_api_loss_confirmation_replays + 1
                )
            ),
        )
        key_supplier = None
        if args.command == "run":
            if args.confirm_paid_calls != PAID_CONFIRMATION:
                raise ProviderError(
                    "Paid calls were not confirmed. Pass exactly "
                    f"--confirm-paid-calls {PAID_CONFIRMATION}"
                )
            load_env_file()
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            client = OpenRouterTextClient(api_key, config.api)
            key_supplier = lambda: safe_key_budget_context(client.get_key_status())
        else:
            client = ReplayTextClient(args.fixtures, config.api.model)

        public_results: list[dict[str, Any]] = []
        for task in tasks:
            pipeline = KubernetesAIPyCraftPipeline(
                config,
                client,
                harness,
                key_context_supplier=key_supplier,
            )
            public_results.append(pipeline.run(task).public_summary())
        output = {
            "status": (
                "completed"
                if all(item["status"] == "completed" for item in public_results)
                else "completed_with_failures"
            ),
            "usage_totals": _aggregate_usage(public_results),
            "results": public_results,
        }
        _print(output)
        return _exit_for(public_results)
    except (ConfigError, TaskError, ProviderError, EnvironmentError, CommandError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
