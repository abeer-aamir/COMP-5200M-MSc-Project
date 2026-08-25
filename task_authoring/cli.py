from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_ENV_PATH,
    DIFFICULTY_LEVELS,
    PROJECT_ROOT,
    load_config,
    load_env_file,
)
from .openrouter import (
    OpenRouterClient,
    ReplayClient,
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
    parser.add_argument("--tasks", type=int, help="Override task count")
    parser.add_argument(
        "--difficulty",
        choices=DIFFICULTY_LEVELS,
        help="Difficulty for a homogeneous task run",
    )
    parser.add_argument(
        "--difficulty-plan",
        help=(
            "Comma-separated campaign counts, for example "
            "easy:5,medium:5,hard:5,very_hard:5"
        ),
    )
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


def _task_plan(args: argparse.Namespace, config) -> list[tuple[str, str]]:
    if args.difficulty_plan:
        if args.tasks is not None or args.difficulty is not None:
            raise ValueError(
                "--difficulty-plan cannot be combined with --tasks or --difficulty"
            )
        counts: dict[str, int] = {}
        for raw_entry in args.difficulty_plan.split(","):
            parts = raw_entry.strip().split(":", 1)
            if len(parts) != 2 or parts[0] not in DIFFICULTY_LEVELS:
                raise ValueError(
                    "difficulty plan entries must use easy:N, medium:N, hard:N, "
                    "or very_hard:N"
                )
            try:
                count = int(parts[1])
            except ValueError as exc:
                raise ValueError("difficulty plan counts must be integers") from exc
            if count < 1 or parts[0] in counts:
                raise ValueError(
                    "difficulty plan counts must be positive and levels unique"
                )
            counts[parts[0]] = count
        task_prefixes = {"very_hard": "very-hard"}
        plan = [
            (f"{task_prefixes.get(level, level)}-{index:03d}", level)
            for level in DIFFICULTY_LEVELS
            for index in range(1, counts.get(level, 0) + 1)
        ]
    else:
        count = config.task_count if args.tasks is None else args.tasks
        level = args.difficulty or config.default_difficulty
        prefix = (
            "very-hard"
            if args.difficulty is not None and level == "very_hard"
            else level if args.difficulty is not None else "pilot"
        )
        plan = [(f"{prefix}-{index:03d}", level) for index in range(1, count + 1)]
    if not 1 <= len(plan) <= config.max_task_count:
        raise ValueError(f"task plan must contain from 1 to {config.max_task_count} tasks")
    return plan


def _load(args: argparse.Namespace):
    config = load_config(args.config)
    plan = _task_plan(args, config)
    config = replace(config, task_count=len(plan))
    brief = args.brief.read_text(encoding="utf-8")
    return config, brief, plan


def _plan(config, task_plan: list[tuple[str, str]]) -> dict[str, object]:
    calls_per_task = 3 * (config.max_revision_rounds + 1)
    return {
        "network_calls": 0,
        "paid_calls": 0,
        "configured_budget_usd": (
            None if config.budget_usd is None else str(config.budget_usd)
        ),
        "account_key_limit_policy": (
            "no local run cap; the externally configured OpenRouter key limit applies"
        ),
        "task_count": len(task_plan),
        "task_plan": [
            {"task_id": task_id, "difficulty_level": difficulty}
            for task_id, difficulty in task_plan
        ],
        "maximum_model_calls": calls_per_task * len(task_plan),
        "provider_max_output_tokens": {
            name: role.max_output_tokens for name, role in config.roles.items()
        },
        "pre_run_cost_upper_bound_usd": None,
        "cost_accounting": "provider-reported usage is logged after every response",
        "models": {name: role.model for name, role in config.roles.items()},
        "ignored_providers": {
            name: list(role.ignored_providers)
            for name, role in config.roles.items()
            if role.ignored_providers
        },
        "target_kubernetes_version": config.target_kubernetes_version,
        "kind_node_image": config.kind_node_image,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config, brief, task_plan = _load(args)
        if args.command == "plan":
            print(json.dumps(_plan(config, task_plan), indent=2))
            return 0

        if args.command == "mock-run":
            client = ReplayClient(args.fixtures)
            pipeline = TaskAuthoringPipeline(
                config, client, args.output_root, task_plan=task_plan
            )
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
        pipeline = TaskAuthoringPipeline(
            config,
            client,
            args.output_root,
            key_budget_context=key_budget_context(key_status),
            task_plan=task_plan,
        )
        summary = pipeline.run(brief)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
