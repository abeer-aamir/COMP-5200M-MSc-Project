from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, PilotConfig
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


def _namespace_from_spec(spec: dict[str, Any]) -> str:
    requirements = spec.get("requirements", [])
    namespace_requirements = [
        str(item.get("text", ""))
        for item in requirements
        if isinstance(item, dict) and re.search(r"\bNamespace\b", str(item.get("text", "")))
    ]
    if len(namespace_requirements) != 1:
        raise PipelineError("Accepted task must contain exactly one Namespace requirement")
    match = re.search(
        r"\bNamespace(?:\s+object)?(?:\s+(?:named|called))?\s+[`'\"]?"
        r"([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)[`'\"]?",
        namespace_requirements[0],
        re.IGNORECASE,
    )
    if match is None:
        raise PipelineError(
            "Namespace requirement must name one DNS-compatible namespace"
        )
    return match.group(1).lower()


def _artifact_integrity(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"summary.json", "artifact_integrity.json"}:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "excluded": ["artifact_integrity.json", "summary.json"],
        "file_count": len(files),
        "aggregate_sha256": hashlib.sha256(
            _canonical_json(files).encode("utf-8")
        ).hexdigest(),
        "files": files,
    }


def _source_state(config: PilotConfig) -> dict[str, Any]:
    paths = set((PROJECT_ROOT / "task_authoring").glob("*.py"))
    paths.add(config.path)
    paths.update(role.prompt_path for role in config.roles.values())
    files = []
    for path in sorted(paths, key=lambda item: str(item).lower()):
        content = path.read_text(encoding="utf-8")
        try:
            display_path = path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            ).as_posix()
        except ValueError:
            display_path = str(path.resolve())
        files.append(
            {
                "path": display_path,
                "sha256": _sha256_text(content),
                "utf8_bytes": len(content.encode("utf-8")),
                "content": content,
            }
        )
    return {
        "schema_version": 1,
        "file_count": len(files),
        "aggregate_sha256": _sha256_text(
            "\n".join(f"{item['path']}\0{item['sha256']}" for item in files)
        ),
        "files": files,
    }


class TaskAuthoringPipeline:
    def __init__(
        self,
        config: PilotConfig,
        client: CompletionClient,
        output_root: Path | str,
        budget_cap_usd: Decimal | None = None,
        key_budget_context: dict[str, Any] | None = None,
        task_plan: list[tuple[str, str]] | None = None,
    ):
        self.config = config
        self.client = client
        self.output_root = Path(output_root)
        self.ledger = BudgetLedger(
            config.budget_usd if budget_cap_usd is None else budget_cap_usd
        )
        self.key_budget_context = key_budget_context
        self.task_plan = task_plan or [
            (f"pilot-{index:03d}", config.default_difficulty)
            for index in range(1, config.task_count + 1)
        ]
        if not 1 <= len(self.task_plan) <= config.max_task_count:
            raise ValueError(
                f"task plan must contain from 1 to {config.max_task_count} tasks"
            )
        for task_id, difficulty in self.task_plan:
            if difficulty not in config.difficulty_contracts:
                raise ValueError(f"unknown difficulty in task plan: {difficulty}")
            if not task_id:
                raise ValueError("task plan contains an empty task ID")
        self._diversity_fingerprints: list[dict[str, Any]] = []
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self.output_root / self.run_id
        self.public_dir = self.run_dir / "public"
        self.private_dir = self.run_dir / "private"
        self.execution_dir = self.run_dir / "tasks"
        self.execution_index_path = self.run_dir / "task-index.json"
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
            "latency_ms": 0,
            "transport_attempt_duration_ms": 0,
            "transport_retry_sleep_duration_ms": 0,
            "cost_usd": Decimal("0"),
        }

    def _record_usage(self, role_name: str, result: ChatResult) -> None:
        for total in (self._usage_totals, self._usage_by_role[role_name]):
            total["requests"] += 1
            total["prompt_tokens"] += result.prompt_tokens
            total["completion_tokens"] += result.completion_tokens
            total["reasoning_tokens"] += result.reasoning_tokens
            total["cached_tokens"] += result.cached_tokens
            total["latency_ms"] += result.latency_ms
            total["transport_attempt_duration_ms"] += sum(
                int(attempt.get("duration_ms", 0) or 0)
                for attempt in result.transport_attempt_log
            )
            total["transport_retry_sleep_duration_ms"] += sum(
                int(attempt.get("retry_sleep_duration_ms", 0) or 0)
                for attempt in result.transport_attempt_log
            )
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
            "latency_ms": total["latency_ms"],
            "transport_attempt_duration_ms": total[
                "transport_attempt_duration_ms"
            ],
            "transport_retry_sleep_duration_ms": total[
                "transport_retry_sleep_duration_ms"
            ],
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
        call_started = time.monotonic()
        call_started_at = datetime.now(timezone.utc).isoformat()
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
                    "transport_attempt_log": list(
                        audit_result.transport_attempt_log
                    ),
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
                    "started_at": call_started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "wall_duration_ms": round(
                        (time.monotonic() - call_started) * 1000
                    ),
                    "requested_model": role.model,
                    "prompt_sha256": prompt_hash,
                    "reservation_usd": (
                        None if reservation is None else str(reservation)
                    ),
                    "spent_so_far_usd": str(self.ledger.spent_usd),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "cost_unknown": self.client.billable and audit_result is None,
                    "transport_attempt_log": list(
                        getattr(exc, "transport_attempt_log", ())
                    ),
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
                "started_at": call_started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "wall_duration_ms": round(
                    (time.monotonic() - call_started) * 1000
                ),
                "billable": result.billable,
                "requested_model": result.requested_model,
                "response_model": result.response_model,
                "provider": result.provider,
                "request_id": result.request_id,
                "generation_id": result.generation_id,
                "prompt_sha256": prompt_hash,
                "reservation_usd": None if reservation is None else str(reservation),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "cached_tokens": result.cached_tokens,
                "cost_usd": str(result.cost_usd),
                "spent_so_far_usd": str(self.ledger.spent_usd),
                "latency_ms": result.latency_ms,
                "retries": result.retries,
                "transport_attempt_log": list(result.transport_attempt_log),
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
            "budget_cap_usd": (
                None if self.ledger.cap_usd is None else str(self.ledger.cap_usd)
            ),
            "target_kubernetes_version": self.config.target_kubernetes_version,
            "kind_node_image": self.config.kind_node_image,
            "models": roles,
            "default_difficulty": self.config.default_difficulty,
            "difficulty_contracts": {
                level: asdict(contract)
                for level, contract in self.config.difficulty_contracts.items()
            },
            "task_plan": [
                {"task_id": task_id, "difficulty_level": difficulty}
                for task_id, difficulty in self.task_plan
            ],
            "openrouter_key_budget_context": self.key_budget_context,
        }

    def _generate_task(
        self, task_id: str, difficulty_level: str, brief: str
    ) -> dict[str, Any]:
        task_started = time.monotonic()
        task_started_at = datetime.now(timezone.utc).isoformat()
        task_dir = self.private_dir / task_id
        contract = self.config.difficulty_contracts[difficulty_level]
        spec_revision_context: dict[str, Any] | None = None
        writer_revision_context: dict[str, Any] | None = None
        critic_revision_context: list[dict[str, Any]] | None = None
        reusable_spec: dict[str, Any] | None = None
        last_errors: list[str] = []

        for round_number in range(self.config.max_revision_rounds + 1):
            round_dir = task_dir / f"round-{round_number}"
            if reusable_spec is None:
                spec_input = {
                    "task_id": task_id,
                    "difficulty_level": difficulty_level,
                    "difficulty_contract": contract.prompt_view(),
                    "authoring_brief": brief,
                    "diversity_fingerprints": self._diversity_fingerprints,
                    "previous_specification": (
                        spec_revision_context.get("previous_specification", {})
                        if spec_revision_context
                        else {}
                    ),
                    "revision_feedback": (
                        spec_revision_context.get("revision_feedback", [])
                        if spec_revision_context
                        else []
                    ),
                }
                spec_result = self._call_role(
                    "spec_generator", task_id, round_number, spec_input
                )
                spec = spec_result.content
            else:
                spec = reusable_spec
                reusable_spec = None
            _write_json(round_dir / "spec.json", spec)
            spec_errors = validate_spec(
                spec,
                self.config,
                task_id,
                difficulty_level,
                self._diversity_fingerprints,
            )
            if spec_errors:
                last_errors = spec_errors
                _write_json(round_dir / "deterministic_errors.json", spec_errors)
                spec_revision_context = {
                    "previous_specification": spec,
                    "revision_feedback": [
                        {
                            "source": "deterministic_schema_gate",
                            "problem": error,
                            "minimum_revision_instruction": (
                                "Correct this schema or structural-contract defect and all "
                                "dependent fields while preserving unaffected content."
                            ),
                        }
                        for error in spec_errors
                    ],
                }
                writer_revision_context = None
                critic_revision_context = None
                continue

            public_projection = {
                "scenario": spec["scenario"],
                "requirements": spec["requirements"],
            }
            writer_input = {
                "specification": public_projection,
                "previous_public_task": (
                    writer_revision_context.get("previous_public_task", "")
                    if writer_revision_context
                    else ""
                ),
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
                    "previous_public_task": writer.get("task_text", ""),
                    "findings": [
                        {
                            "defect_location": "public_task_text",
                            "evidence": error,
                            "minimum_revision_instruction": (
                                "Correct only the public rendering while preserving the "
                                "specification exactly."
                            ),
                        }
                        for error in writer_errors
                    ],
                    "instruction": (
                        "Revise the public text only; preserve every specification detail."
                    ),
                }
                critic_revision_context = None
                continue

            critic_input = {
                "difficulty_level": difficulty_level,
                "difficulty_contract": contract.prompt_view(),
                "private_specification": {
                    key: value
                    for key, value in spec.items()
                    if key != "verification_blueprints"
                },
                "verification_blueprints": spec["verification_blueprints"],
                "public_task_text": writer["task_text"],
                "diversity_fingerprints_of_accepted_tasks": (
                    self._diversity_fingerprints
                ),
                "revision_context": critic_revision_context or [],
            }
            critic_result = self._call_role("critic", task_id, round_number, critic_input)
            critic = critic_result.content
            _write_json(round_dir / "critic.json", critic)
            critic_shape_errors = validate_critic(critic, spec, self.config)
            gate_errors = acceptance_errors(
                spec,
                writer,
                critic,
                self.config,
                difficulty_level,
                self._diversity_fingerprints,
            )
            last_errors = sorted(set(critic_shape_errors + gate_errors))
            _write_json(round_dir / "acceptance_errors.json", last_errors)
            if not last_errors:
                namespace = _namespace_from_spec(spec)
                task_text = writer["task_text"].strip() + "\n"
                self.public_dir.mkdir(parents=True, exist_ok=True)
                public_path = self.public_dir / f"{task_id}.txt"
                public_path.write_text(task_text, encoding="utf-8")
                execution_path = self.execution_dir / task_id / "description.txt"
                execution_path.parent.mkdir(parents=True, exist_ok=True)
                execution_path.write_text(task_text, encoding="utf-8")
                _write_json(task_dir / "final_spec.json", spec)
                _write_json(task_dir / "final_writer.json", writer)
                _write_json(task_dir / "final_critic.json", critic)
                fingerprint_entry = {
                    "task_id": task_id,
                    "difficulty_level": difficulty_level,
                    "fingerprint": spec["diversity_fingerprint"],
                }
                self._diversity_fingerprints.append(fingerprint_entry)
                _write_json(
                    self.private_dir / "diversity_fingerprints.json",
                    self._diversity_fingerprints,
                )
                return {
                    "task_id": task_id,
                    "difficulty_level": difficulty_level,
                    "status": "accepted",
                    "started_at": task_started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "round": round_number,
                    "public_path": str(public_path),
                    "execution_description": str(execution_path),
                    "namespace": namespace,
                    "hardness_assessment": critic["hardness_assessment"],
                    "critic_findings": len(critic["findings"]),
                    "duration_ms": round((time.monotonic() - task_started) * 1000),
                }

            findings = [
                finding
                for finding in critic.get("findings", [])
                if isinstance(finding, dict)
            ]
            critic_revision_context = findings
            if critic.get("verdict") == "reject":
                break
            if findings and all(
                finding.get("defect_location") == "public_task_text"
                for finding in findings
            ):
                reusable_spec = spec
                spec_revision_context = None
                writer_revision_context = {
                    "previous_public_task": writer.get("task_text", ""),
                    "findings": findings,
                    "instruction": (
                        "Revise the public text only; preserve every specification detail."
                    ),
                }
            else:
                spec_revision_context = {
                    "previous_specification": spec,
                    "revision_feedback": findings,
                }
                writer_revision_context = None

        return {
            "task_id": task_id,
            "difficulty_level": difficulty_level,
            "status": "rejected",
            "started_at": task_started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "round": round_number,
            "errors": last_errors,
            "duration_ms": round((time.monotonic() - task_started) * 1000),
        }

    def run(self, brief: str) -> dict[str, Any]:
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        stage_timings_ms: dict[str, int] = {}
        self.private_dir.mkdir(parents=True, exist_ok=True)
        source_started = time.monotonic()
        source_state = _source_state(self.config)
        _write_json(self.private_dir / "source_state.json", source_state)
        stage_timings_ms["source_snapshot"] = round(
            (time.monotonic() - source_started) * 1000
        )
        provenance_started = time.monotonic()
        provenance = self._provenance(brief)
        provenance["source_state"] = {
            "artifact": "source_state.json",
            "file_count": source_state["file_count"],
            "aggregate_sha256": source_state["aggregate_sha256"],
        }
        _write_json(self.private_dir / "provenance.json", provenance)
        stage_timings_ms["provenance"] = round(
            (time.monotonic() - provenance_started) * 1000
        )
        task_results: list[dict[str, Any]] = []
        status = "completed"
        fatal_error: dict[str, str] | None = None
        task_generation_started = time.monotonic()
        try:
            for task_id, difficulty_level in self.task_plan:
                task_results.append(
                    self._generate_task(task_id, difficulty_level, brief)
                )
        except Exception as exc:
            status = "failed"
            fatal_error = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            stage_timings_ms["task_generation"] = round(
                (time.monotonic() - task_generation_started) * 1000
            )
        accepted_results = [
            item for item in task_results if item.get("status") == "accepted"
        ]
        _write_json(
            self.execution_index_path,
            {
                "schema_version": 1,
                "kubernetes_version": self.config.target_kubernetes_version,
                "execution_image": self.config.execution_image,
                "tasks": [
                    {
                        "task_id": item["task_id"],
                        "namespace": item["namespace"],
                        "description": f"tasks/{item['task_id']}/description.txt",
                        "post_execution_suite": None,
                    }
                    for item in accepted_results
                ],
            },
        )
        summary = {
            "run_id": self.run_id,
            "status": status,
            "mode": "paid" if self.client.billable else "offline-replay",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round((time.monotonic() - started) * 1000),
            "stage_timings_ms": stage_timings_ms,
            "budget_cap_usd": (
                None if self.ledger.cap_usd is None else str(self.ledger.cap_usd)
            ),
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
            "requested_tasks": len(self.task_plan),
            "task_index": str(self.execution_index_path),
            "requested_by_difficulty": {
                level: sum(difficulty == level for _, difficulty in self.task_plan)
                for level in self.config.difficulty_contracts
            },
            "tasks": task_results,
            "fatal_error": fatal_error,
        }
        integrity_started = time.monotonic()
        try:
            integrity = _artifact_integrity(self.run_dir)
            _write_json(self.private_dir / "artifact_integrity.json", integrity)
            summary["artifact_integrity"] = {
                "artifact": str(self.private_dir / "artifact_integrity.json"),
                "file_count": integrity["file_count"],
                "aggregate_sha256": integrity["aggregate_sha256"],
            }
        except Exception as exc:
            summary["artifact_integrity_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            summary["finalization_timings_ms"] = {
                "artifact_integrity": round(
                    (time.monotonic() - integrity_started) * 1000
                )
            }
        _write_json(self.private_dir / "summary.json", summary)
        if fatal_error:
            raise PipelineError(
                f"Pilot failed; private summary retained at {self.private_dir / 'summary.json'}: "
                f"{fatal_error['message']}"
            )
        return summary
