from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .config import PilotConfig
from .openrouter import BudgetLedger, ChatResult, CompletionClient
from .schemas import (
    ROLE_SCHEMAS,
    acceptance_errors,
    validate_critic,
    validate_spec,
    validate_writer,
)


class PipelineError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class TaskAuthoringPipeline:
    def __init__(
        self,
        config: PilotConfig,
        client: CompletionClient,
        output_root: Path | str,
        budget_cap_usd: Decimal | None = None,
        key_budget_context: dict[str, Any] | None = None,
    ):
        self.config = config
        self.client = client
        self.output_root = Path(output_root)
        self.ledger = BudgetLedger(budget_cap_usd or config.budget_usd)
        self.key_budget_context = key_budget_context
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self.output_root / self.run_id
        self.public_dir = self.run_dir / "public"
        self.private_dir = self.run_dir / "private"
        self.usage_path = self.private_dir / "usage.jsonl"
        self._event_sequence = 0
        self._unknown_cost_failures = 0
        self._usage_totals = self._new_usage_total()
        self._usage_by_role = {
            role_name: self._new_usage_total() for role_name in self.config.roles
        }

    @staticmethod
    def _new_usage_total() -> dict[str, Any]:
        return {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": Decimal("0"),
        }

    def _record_usage(self, role_name: str, result: ChatResult) -> None:
        for total in (self._usage_totals, self._usage_by_role[role_name]):
            total["requests"] += 1
            total["prompt_tokens"] += result.prompt_tokens
            total["completion_tokens"] += result.completion_tokens
            total["reasoning_tokens"] += result.reasoning_tokens
            total["cached_tokens"] += result.cached_tokens
            total["cost_usd"] += result.cost_usd

    @staticmethod
    def _usage_snapshot(total: dict[str, Any]) -> dict[str, Any]:
        return {
            "requests": total["requests"],
            "prompt_tokens": total["prompt_tokens"],
            "completion_tokens": total["completion_tokens"],
            "total_tokens": total["prompt_tokens"] + total["completion_tokens"],
            "reasoning_tokens": total["reasoning_tokens"],
            "cached_tokens": total["cached_tokens"],
            "cost_usd": str(total["cost_usd"]),
        }

    def _log_event(self, event: dict[str, Any]) -> None:
        self._event_sequence += 1
        enriched = {
            "sequence": self._event_sequence,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            **event,
        }
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    def _call_role(
        self,
        role_name: str,
        task_id: str,
        round_number: int,
        user_data: dict[str, Any],
    ) -> ChatResult:
        role = self.config.roles[role_name]
        system_prompt = role.prompt_path.read_text(encoding="utf-8")
        user_prompt = json.dumps(user_data, ensure_ascii=False, indent=2)
        reservation = self.ledger.estimate_reservation(
            role,
            system_prompt,
            user_prompt,
            self.config.reservation_safety_multiplier,
        )
        if self.client.billable:
            self.ledger.check_reservation(reservation)
        prompt_hash = _sha256_text(system_prompt + "\n" + user_prompt)
        try:
            result = self.client.complete(
                role, system_prompt, user_prompt, ROLE_SCHEMAS[role_name]
            )
            if result.billable:
                self.ledger.charge(result.cost_usd)
        except Exception as exc:
            audit_result = getattr(exc, "audit_result", None)
            raw_content = getattr(exc, "raw_content", None)
            failure_usage: dict[str, Any] = {}
            if isinstance(audit_result, ChatResult):
                if audit_result.billable:
                    self.ledger.charge(audit_result.cost_usd)
                self._record_usage(role_name, audit_result)
                invalid_path = (
                    self.private_dir
                    / task_id
                    / f"round-{round_number}"
                    / f"{role_name}_invalid_response.txt"
                )
                invalid_path.parent.mkdir(parents=True, exist_ok=True)
                invalid_text = (
                    raw_content if isinstance(raw_content, str) else repr(raw_content)
                )
                invalid_path.write_text(invalid_text, encoding="utf-8")
                failure_usage = {
                    "billable": audit_result.billable,
                    "response_model": audit_result.response_model,
                    "provider": audit_result.provider,
                    "request_id": audit_result.request_id,
                    "generation_id": audit_result.generation_id,
                    "prompt_tokens": audit_result.prompt_tokens,
                    "completion_tokens": audit_result.completion_tokens,
                    "reasoning_tokens": audit_result.reasoning_tokens,
                    "cached_tokens": audit_result.cached_tokens,
                    "cost_usd": str(audit_result.cost_usd),
                    "latency_ms": audit_result.latency_ms,
                    "retries": audit_result.retries,
                    "invalid_response_path": str(invalid_path),
                    "invalid_response_sha256": _sha256_text(invalid_text),
                }
            elif self.client.billable:
                self._unknown_cost_failures += 1
            self._log_event(
                {
                    "task_id": task_id,
                    "round": round_number,
                    "role": role_name,
                    "status": "failed",
                    "requested_model": role.model,
                    "prompt_sha256": prompt_hash,
                    "reservation_usd": str(reservation),
                    "spent_so_far_usd": str(self.ledger.spent_usd),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "cost_unknown": self.client.billable and audit_result is None,
                    **failure_usage,
                }
            )
            raise

        self._record_usage(role_name, result)

        self._log_event(
            {
                "task_id": task_id,
                "round": round_number,
                "role": role_name,
                "status": "completed",
                "billable": result.billable,
                "requested_model": result.requested_model,
                "response_model": result.response_model,
                "provider": result.provider,
                "request_id": result.request_id,
                "generation_id": result.generation_id,
                "prompt_sha256": prompt_hash,
                "reservation_usd": str(reservation),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "cached_tokens": result.cached_tokens,
                "cost_usd": str(result.cost_usd),
                "spent_so_far_usd": str(self.ledger.spent_usd),
                "latency_ms": result.latency_ms,
                "retries": result.retries,
                "parse_mode": result.parse_mode,
            }
        )
        return result

    def _provenance(self, brief: str) -> dict[str, Any]:
        roles: dict[str, Any] = {}
        for name, role in self.config.roles.items():
            role_data = asdict(role)
            role_data["input_usd_per_million"] = str(role.input_usd_per_million)
            role_data["output_usd_per_million"] = str(role.output_usd_per_million)
            role_data["prompt_path"] = str(role.prompt_path.relative_to(role.prompt_path.parents[2]))
            role_data["prompt_sha256"] = _sha256_text(
                role.prompt_path.read_text(encoding="utf-8")
            )
            roles[name] = role_data
        return {
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "paid" if self.client.billable else "offline-replay",
            "config_path": str(self.config.path),
            "config_sha256": _sha256_text(self.config.path.read_text(encoding="utf-8")),
            "brief_sha256": _sha256_text(brief),
            "budget_cap_usd": str(self.ledger.cap_usd),
            "target_kubernetes_version": self.config.target_kubernetes_version,
            "kind_node_image": self.config.kind_node_image,
            "models": roles,
            "hardness_gate": self.config.hardness_gate,
            "openrouter_key_budget_context": self.key_budget_context,
        }

    def _generate_task(self, task_index: int, brief: str) -> dict[str, Any]:
        task_id = f"pilot-{task_index:03d}"
        task_dir = self.private_dir / task_id
        spec_revision_context: dict[str, Any] | None = None
        writer_revision_context: dict[str, Any] | None = None
        reusable_spec: dict[str, Any] | None = None
        last_errors: list[str] = []

        for round_number in range(self.config.max_revision_rounds + 1):
            round_dir = task_dir / f"round-{round_number}"
            if reusable_spec is None:
                spec_input = {
                    "task_id": task_id,
                    "authoring_brief": brief,
                    "target_kubernetes_version": self.config.target_kubernetes_version,
                    "kind_node_image": self.config.kind_node_image,
                    "round": round_number,
                    "revision_context": spec_revision_context,
                }
                spec_result = self._call_role(
                    "spec_generator", task_id, round_number, spec_input
                )
                spec = spec_result.content
            else:
                spec = reusable_spec
                reusable_spec = None
            _write_json(round_dir / "spec.json", spec)
            spec_errors = validate_spec(spec, self.config, task_id)
            if spec_errors:
                last_errors = spec_errors
                _write_json(round_dir / "deterministic_errors.json", spec_errors)
                spec_revision_context = {
                    "instruction": "Edit the previous specification in place.",
                    "preserve_public_requirement_count": len(
                        spec.get("public_requirements", [])
                    ),
                    "previous_spec": spec,
                    "deterministic_errors": spec_errors,
                }
                writer_revision_context = None
                continue

            public_projection = {
                "task_id": task_id,
                "title": spec["title"],
                "scenario": spec["scenario"],
                "target_kubernetes_version": spec["target_kubernetes_version"],
                "public_requirements": spec["public_requirements"],
            }
            writer_input = {
                "public_specification": public_projection,
                "round": round_number,
                "revision_context": writer_revision_context,
            }
            writer_result = self._call_role(
                "plaintext_writer", task_id, round_number, writer_input
            )
            writer = writer_result.content
            _write_json(round_dir / "writer.json", writer)
            writer_errors = validate_writer(writer, spec)
            if writer_errors:
                last_errors = writer_errors
                _write_json(round_dir / "deterministic_errors.json", writer_errors)
                reusable_spec = spec
                spec_revision_context = None
                writer_revision_context = {
                    "previous_public_text": writer.get("task_text"),
                    "deterministic_errors": writer_errors,
                    "instruction": (
                        "Revise the public text only; preserve every specification detail."
                    ),
                }
                continue

            critic_input = {
                "private_specification": spec,
                "proposed_public_task_text": writer["task_text"],
                "hardness_gate": self.config.hardness_gate,
                "round": round_number,
            }
            critic_result = self._call_role("critic", task_id, round_number, critic_input)
            critic = critic_result.content
            _write_json(round_dir / "critic.json", critic)
            critic_shape_errors = validate_critic(critic, spec, self.config)
            gate_errors = acceptance_errors(spec, writer, critic, self.config)
            last_errors = sorted(set(critic_shape_errors + gate_errors))
            _write_json(round_dir / "acceptance_errors.json", last_errors)
            if not last_errors:
                self.public_dir.mkdir(parents=True, exist_ok=True)
                public_path = self.public_dir / f"{task_id}.txt"
                public_path.write_text(writer["task_text"].strip() + "\n", encoding="utf-8")
                _write_json(task_dir / "final_spec.json", spec)
                _write_json(task_dir / "final_writer.json", writer)
                _write_json(task_dir / "final_critic.json", critic)
                return {
                    "task_id": task_id,
                    "status": "accepted",
                    "round": round_number,
                    "public_path": str(public_path),
                    "hardness_score": critic["hardness_score"],
                    "categories": [
                        assignment["category"]
                        for assignment in critic["category_assignments"]
                    ],
                }

            spec_revision_context = {
                "instruction": "Edit the previous specification in place.",
                "preserve_public_requirement_count": len(
                    spec.get("public_requirements", [])
                ),
                "previous_spec": spec,
                "critic_revision_instructions": critic.get(
                    "revision_instructions", []
                ),
            }
            writer_revision_context = None

        return {
            "task_id": task_id,
            "status": "rejected",
            "round": self.config.max_revision_rounds,
            "errors": last_errors,
        }

    def run(self, brief: str) -> dict[str, Any]:
        self.private_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.private_dir / "provenance.json", self._provenance(brief))
        task_results: list[dict[str, Any]] = []
        status = "completed"
        fatal_error: dict[str, str] | None = None
        try:
            for task_index in range(1, self.config.task_count + 1):
                task_results.append(self._generate_task(task_index, brief))
        except Exception as exc:
            status = "failed"
            fatal_error = {"type": type(exc).__name__, "message": str(exc)}
        summary = {
            "run_id": self.run_id,
            "status": status,
            "mode": "paid" if self.client.billable else "offline-replay",
            "budget_cap_usd": str(self.ledger.cap_usd),
            "provider_reported_spend_usd": str(self.ledger.spent_usd),
            "provider_cost_complete": self._unknown_cost_failures == 0,
            "unknown_cost_failures": self._unknown_cost_failures,
            "usage_totals": self._usage_snapshot(self._usage_totals),
            "usage_by_role": {
                role_name: self._usage_snapshot(total)
                for role_name, total in self._usage_by_role.items()
            },
            "openrouter_key_budget_context": self.key_budget_context,
            "accepted_tasks": sum(x.get("status") == "accepted" for x in task_results),
            "requested_tasks": self.config.task_count,
            "tasks": task_results,
            "fatal_error": fatal_error,
        }
        _write_json(self.private_dir / "summary.json", summary)
        if fatal_error:
            raise PipelineError(
                f"Pilot failed; private summary retained at {self.private_dir / 'summary.json'}: "
                f"{fatal_error['message']}"
            )
        return summary
