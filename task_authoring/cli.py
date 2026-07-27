from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, DEFAULT_ENV_PATH, PROJECT_ROOT, load_config, load_env_file
from .openrouter import (
    OpenRouterClient,
    ReplayClient,
    derive_effective_local_cap,
    key_budget_context,
)
from .pipeline import TaskAuthoringPipeline


PAID_CONFIRMATION = "I_ACCEPT_PAID_OPENROUTER_CALLS"
DEFAULT_BRIEF = PROJECT_ROOT / "benchmark" / "pilot_brief.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "pilot_runs"
DEFAULT_FIXTURES = PROJECT_ROOT / "benchmark" / "mock_responses"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Three-role Kubernetes benchmark task-authoring pilot"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", type=int, help="Override task_count (maximum three)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Show configuration and make no network calls")
    mock = subparsers.add_parser("mock-run", help="Run with committed replay fixtures")
    mock.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    paid = subparsers.add_parser("run", help="Run paid OpenRouter generation")
    paid.add_argument(
        "--confirm-paid-calls",
        required=True,
        help=f"Must be exactly {PAID_CONFIRMATION}",
    )
    return parser


def _load(args: argparse.Namespace):
    config = load_config(args.config)
    if args.tasks is not None:
        if not 1 <= args.tasks <= config.max_task_count:
            raise ValueError(f"--tasks must be from 1 to {config.max_task_count}")
        config = replace(config, task_count=args.tasks)
    brief = args.brief.read_text(encoding="utf-8")
    return config, brief


def _plan(config) -> dict[str, object]:
    calls_per_task = 3 * (config.max_revision_rounds + 1)
    # These deliberately large prompt envelopes provide a useful pre-run upper estimate.
    prompt_bytes = {
        "spec_generator": 60000,
        "plaintext_writer": 50000,
        "critic": 90000,
    }
    estimated = Decimal("0")
    role_estimates: dict[str, str] = {}
    for name, role in config.roles.items():
        input_cost = (
            Decimal(prompt_bytes[name])
            * role.input_usd_per_million
            / Decimal(1_000_000)
        )
        output_cost = (
            Decimal(role.max_output_tokens)
            * role.output_usd_per_million
            / Decimal(1_000_000)
        )
        per_call = (input_cost + output_cost) * config.reservation_safety_multiplier
        role_estimates[name] = f"{per_call:.6f}"
        estimated += per_call * (config.max_revision_rounds + 1) * config.task_count
    return {
        "network_calls": 0,
        "paid_calls": 0,
        "configured_budget_usd": str(config.budget_usd),
        "account_key_limit_policy": "larger or uncapped keys accepted; local cap remains authoritative",
        "task_count": config.task_count,
        "maximum_model_calls": calls_per_task * config.task_count,
        "conservative_role_reservation_usd": role_estimates,
        "conservative_full_run_estimate_usd": f"{estimated:.6f}",
        "models": {name: role.model for name, role in config.roles.items()},
        "target_kubernetes_version": config.target_kubernetes_version,
        "kind_node_image": config.kind_node_image,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config, brief = _load(args)
        if args.command == "plan":
            print(json.dumps(_plan(config), indent=2))
            return 0

        if args.command == "mock-run":
            client = ReplayClient(args.fixtures)
            pipeline = TaskAuthoringPipeline(config, client, args.output_root)
            summary = pipeline.run(brief)
            print(json.dumps(summary, indent=2))
            return 0

        if args.confirm_paid_calls != PAID_CONFIRMATION:
            raise ValueError(
                f"Paid run refused. Pass --confirm-paid-calls {PAID_CONFIRMATION} exactly."
            )
        load_env_file(DEFAULT_ENV_PATH)
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError(f"Add OPENROUTER_API_KEY to {DEFAULT_ENV_PATH}")
        client = OpenRouterClient(api_key, config.api_base)
        key_status = client.get_key_status()  # Free preflight; no model request.
        effective_cap = derive_effective_local_cap(key_status, config.budget_usd)
        pipeline = TaskAuthoringPipeline(
            config,
            client,
            args.output_root,
            budget_cap_usd=effective_cap,
            key_budget_context=key_budget_context(key_status),
        )
        summary = pipeline.run(brief)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
