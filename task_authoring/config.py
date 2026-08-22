from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "benchmark" / "pilot_config.json"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
REQUIRED_ROLES = ("spec_generator", "plaintext_writer", "critic")
DIFFICULTY_LEVELS = ("easy", "medium", "hard")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RoleConfig:
    name: str
    model: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    max_output_tokens: int | None
    reasoning_effort: str | None
    reasoning_max_tokens: int | None
    ignored_providers: tuple[str, ...]
    prompt_path: Path


@dataclass(frozen=True)
class DifficultyContract:
    level: str
    minimum_score: int
    maximum_score: int
    minimum_categories: int
    minimum_public_requirements: int
    maximum_public_requirements: int
    minimum_resource_kinds: int
    maximum_resource_kinds: int
    minimum_dependency_edges: int
    maximum_dependency_edges: int | None
    runtime_behaviors: int
    safety_constraints: int
    minimum_interacting_mechanisms: int

    def prompt_view(self) -> dict[str, int | None | str]:
        """Return structural controls without exposing the critic score gate."""
        return {
            "difficulty_level": self.level,
            "minimum_public_requirements": self.minimum_public_requirements,
            "maximum_public_requirements": self.maximum_public_requirements,
            "minimum_resource_kinds": self.minimum_resource_kinds,
            "maximum_resource_kinds": self.maximum_resource_kinds,
            "minimum_dependency_edges": self.minimum_dependency_edges,
            "maximum_dependency_edges": self.maximum_dependency_edges,
            "runtime_behaviors": self.runtime_behaviors,
            "safety_constraints": self.safety_constraints,
            "minimum_interacting_mechanisms": self.minimum_interacting_mechanisms,
        }


@dataclass(frozen=True)
class PilotConfig:
    path: Path
    api_base: str
    budget_usd: Decimal | None
    task_count: int
    max_task_count: int
    max_revision_rounds: int
    target_kubernetes_version: str
    kind_node_image: str
    reservation_safety_multiplier: Decimal
    roles: dict[str, RoleConfig]
    default_difficulty: str
    difficulty_contracts: dict[str, DifficultyContract]


def _decimal(value: Any, label: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as exc:  # pragma: no cover - defensive error detail
        raise ConfigError(f"{label} must be a decimal number") from exc


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> PilotConfig:
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read configuration {config_path}: {exc}") from exc

    raw_budget = raw.get("budget_usd")
    budget = None if raw_budget is None else _decimal(raw_budget, "budget_usd")
    if budget is not None and budget <= 0:
        raise ConfigError("budget_usd must be null or above 0")

    task_count = int(raw.get("task_count", 0))
    max_task_count = int(raw.get("max_task_count", 0))
    if not 1 <= task_count <= max_task_count <= 15:
        raise ConfigError("Require 1 <= task_count <= max_task_count <= 15")

    revision_rounds = int(raw.get("max_revision_rounds", -1))
    if revision_rounds not in (0, 1):
        raise ConfigError("max_revision_rounds must be 0 or 1 for this pilot")

    root = PROJECT_ROOT
    roles: dict[str, RoleConfig] = {}
    model_data = raw.get("models", {})
    for role_name in REQUIRED_ROLES:
        if role_name not in model_data:
            raise ConfigError(f"Missing model configuration for {role_name}")
        item = model_data[role_name]
        prompt_path = (root / str(item["prompt"])).resolve()
        if root not in prompt_path.parents or not prompt_path.is_file():
            raise ConfigError(f"Invalid prompt path for {role_name}: {prompt_path}")
        raw_max_tokens = item.get("max_output_tokens")
        max_tokens = None if raw_max_tokens is None else int(raw_max_tokens)
        if max_tokens is not None and max_tokens < 256:
            raise ConfigError(
                f"max_output_tokens for {role_name} must be null or at least 256"
            )
        reasoning_effort = item.get("reasoning_effort")
        raw_reasoning_max = item.get("reasoning_max_tokens")
        reasoning_max_tokens = (
            None if raw_reasoning_max is None else int(raw_reasoning_max)
        )
        if reasoning_effort not in {None, "low", "medium", "high"}:
            raise ConfigError(
                f"reasoning_effort for {role_name} must be null, low, medium, or high"
            )
        if reasoning_max_tokens is not None and reasoning_max_tokens < 1024:
            raise ConfigError(
                f"reasoning_max_tokens for {role_name} must be at least 1024"
            )
        if (
            reasoning_max_tokens is not None
            and max_tokens is not None
            and reasoning_max_tokens >= max_tokens
        ):
            raise ConfigError(
                f"reasoning_max_tokens for {role_name} must be below max_output_tokens"
            )
        if reasoning_effort is not None and reasoning_max_tokens is not None:
            raise ConfigError(
                f"{role_name} cannot set both reasoning_effort and reasoning_max_tokens"
            )
        raw_ignored_providers = item.get("ignored_providers", [])
        if not isinstance(raw_ignored_providers, list) or any(
            not isinstance(provider, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9/-]*", provider) is None
            for provider in raw_ignored_providers
        ):
            raise ConfigError(
                f"ignored_providers for {role_name} must be provider slug strings"
            )
        ignored_providers = tuple(raw_ignored_providers)
        if len(set(ignored_providers)) != len(ignored_providers):
            raise ConfigError(f"ignored_providers for {role_name} must be unique")
        roles[role_name] = RoleConfig(
            name=role_name,
            model=str(item["model"]),
            input_usd_per_million=_decimal(
                item["input_usd_per_million"], f"{role_name} input price"
            ),
            output_usd_per_million=_decimal(
                item["output_usd_per_million"], f"{role_name} output price"
            ),
            max_output_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_max_tokens=reasoning_max_tokens,
            ignored_providers=ignored_providers,
            prompt_path=prompt_path,
        )

    default_difficulty = str(raw.get("default_difficulty", ""))
    if default_difficulty not in DIFFICULTY_LEVELS:
        raise ConfigError("default_difficulty must be easy, medium, or hard")

    raw_contracts = raw.get("difficulty_contracts", {})
    if set(raw_contracts) != set(DIFFICULTY_LEVELS):
        raise ConfigError("difficulty_contracts must define easy, medium, and hard")
    required_contract_fields = {
        "minimum_score",
        "maximum_score",
        "minimum_categories",
        "minimum_public_requirements",
        "maximum_public_requirements",
        "minimum_resource_kinds",
        "maximum_resource_kinds",
        "minimum_dependency_edges",
        "maximum_dependency_edges",
        "runtime_behaviors",
        "safety_constraints",
        "minimum_interacting_mechanisms",
    }
    contracts: dict[str, DifficultyContract] = {}
    for level in DIFFICULTY_LEVELS:
        item = raw_contracts[level]
        if set(item) != required_contract_fields:
            missing = required_contract_fields - set(item)
            extra = set(item) - required_contract_fields
            raise ConfigError(
                f"Invalid {level} difficulty contract; missing={missing}, extra={extra}"
            )
        maximum_edges = item["maximum_dependency_edges"]
        contract = DifficultyContract(
            level=level,
            minimum_score=int(item["minimum_score"]),
            maximum_score=int(item["maximum_score"]),
            minimum_categories=int(item["minimum_categories"]),
            minimum_public_requirements=int(item["minimum_public_requirements"]),
            maximum_public_requirements=int(item["maximum_public_requirements"]),
            minimum_resource_kinds=int(item["minimum_resource_kinds"]),
            maximum_resource_kinds=int(item["maximum_resource_kinds"]),
            minimum_dependency_edges=int(item["minimum_dependency_edges"]),
            maximum_dependency_edges=(
                None if maximum_edges is None else int(maximum_edges)
            ),
            runtime_behaviors=int(item["runtime_behaviors"]),
            safety_constraints=int(item["safety_constraints"]),
            minimum_interacting_mechanisms=int(
                item["minimum_interacting_mechanisms"]
            ),
        )
        numeric_values = [
            contract.minimum_score,
            contract.maximum_score,
            contract.minimum_categories,
            contract.minimum_public_requirements,
            contract.maximum_public_requirements,
            contract.minimum_resource_kinds,
            contract.maximum_resource_kinds,
            contract.minimum_dependency_edges,
            contract.runtime_behaviors,
            contract.safety_constraints,
            contract.minimum_interacting_mechanisms,
        ]
        if any(value < 1 for value in numeric_values):
            raise ConfigError(f"{level} difficulty contract values must be positive")
        if not 1 <= contract.minimum_score <= contract.maximum_score <= 10:
            raise ConfigError(f"{level} score range must be within 1..10")
        if contract.minimum_public_requirements > contract.maximum_public_requirements:
            raise ConfigError(f"{level} public requirement range is reversed")
        if contract.minimum_resource_kinds > contract.maximum_resource_kinds:
            raise ConfigError(f"{level} resource-kind range is reversed")
        if (
            contract.maximum_dependency_edges is not None
            and contract.minimum_dependency_edges > contract.maximum_dependency_edges
        ):
            raise ConfigError(f"{level} dependency-edge range is reversed")
        contracts[level] = contract

    multiplier = _decimal(
        raw.get("reservation_safety_multiplier"), "reservation_safety_multiplier"
    )
    if multiplier < Decimal("1.0"):
        raise ConfigError("reservation_safety_multiplier cannot be below 1")

    return PilotConfig(
        path=config_path,
        api_base=str(raw["api_base"]).rstrip("/"),
        budget_usd=budget,
        task_count=task_count,
        max_task_count=max_task_count,
        max_revision_rounds=revision_rounds,
        target_kubernetes_version=str(raw["target_kubernetes_version"]),
        kind_node_image=str(raw["kind_node_image"]),
        reservation_safety_multiplier=multiplier,
        roles=roles,
        default_difficulty=default_difficulty,
        difficulty_contracts=contracts,
    )


def load_env_file(path: Path | str = DEFAULT_ENV_PATH) -> None:
    """Load a tiny KEY=VALUE .env file without adding a dependency."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"Invalid .env entry on line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
