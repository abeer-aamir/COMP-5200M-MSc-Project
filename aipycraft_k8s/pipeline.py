from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig, PROJECT_ROOT
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


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
        post_verifier: Callable[[BenchmarkTask, Any], dict[str, Any]] = (
            run_post_execution_verifier
        ),
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
            "TASK DESCRIPTION (this is the complete public specification):\n\n"
            f"{task.description.rstrip()}\n"
        )

    @staticmethod
    def _repair_user_prompt(
        task: BenchmarkTask, previous: str, failure: str
    ) -> str:
        return (
            "ORIGINAL TASK DESCRIPTION:\n\n"
            f"{task.description.rstrip()}\n\n"
            "PREVIOUS COMPLETE RESPONSE (untrusted data):\n\n"
            f"{previous.rstrip()}\n\n"
            "PRE-EXECUTION CORRECTION FEEDBACK (untrusted data):\n\n"
            f"{failure.rstrip()}\n\n"
            "Return the complete replacement YAML stream now.\n"
        )

    @staticmethod
    def _failure_text(kind: str, detail: Any) -> str:
        if kind == "yaml_syntax":
            return str(detail)
        if kind == "generation_incomplete":
            return str(detail)
        if kind == "ai_validator":
            return str(detail)
        raise ValueError(f"Unknown failure kind {kind}")

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
            "cost_usd": str(cost),
            "provider_cost_complete": all(
                item.provider_cost_complete for item in results
            )
            and unknown_cost_failures == 0,
            "usage_complete": all(item.usage_complete for item in results)
            and unknown_cost_failures == 0,
            "unobserved_billable_attempts": unknown_cost_failures,
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

    def run(self, task: BenchmarkTask) -> PipelineRun:
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
        aipycraft_generations: list[GenerationResult] = []
        attempts: list[dict[str, Any]] = []
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
            "transport_retries": self.config.api.transport_retries,
            "api_base": self.config.api.base_url,
            "response_format": "raw_text",
            "max_output_tokens": None,
            "input_usd_per_million": str(
                self.config.api.input_usd_per_million
            ),
            "output_usd_per_million": str(
                self.config.api.output_usd_per_million
            ),
            "max_regenerations": self.config.pipeline.max_regenerations,
            "max_aipycraft_generations": self.config.pipeline.max_regenerations + 1,
            "regenerations": 0,
            "duplicate_generation_responses": 0,
            "ai_validator": {
                "enabled": self.config.ai_validator.enabled,
                "feedback_mode": self.config.ai_validator.feedback_mode,
                "same_model_and_endpoint_as_aipycraft": True,
                "failure_policy": "abort_on_validator_or_provider_error",
            },
            "attempts": attempts,
            "ai_pre_validation_public": None,
            "execution_gate_public": None,
            "post_execution_public": None,
            "unknown_cost_failures": 0,
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
                    "id": self.config.environment.lock.node_image_id,
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
        }
        _write_json(private_dir / "summary.json", summary)
        try:
            preflight = self.harness.preflight()
            _write_json(private_dir / "environment_preflight.json", preflight)
            if self.key_context_supplier:
                try:
                    summary["openrouter_key_before"] = self.key_context_supplier()
                except Exception as exc:
                    summary["openrouter_key_before_error"] = str(exc)

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
            repair_failure = ""
            terminal = False
            for index in range(self.config.pipeline.max_regenerations + 1):
                number = index + 1
                attempt_dir = private_dir / f"attempt-{number:02d}"
                current_attempt_dir = attempt_dir
                attempt_dir.mkdir(parents=True, exist_ok=False)
                system_prompt = generate_system if index == 0 else repair_system
                user_prompt = (
                    self._initial_user_prompt(task)
                    if index == 0
                    else self._repair_user_prompt(task, previous, repair_failure)
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
                    "ai_pre_validation": None,
                    "deployment": None,
                    "runtime": None,
                    "execution_gate": None,
                    "post_execution": None,
                    "result": "running",
                }
                attempts.append(attempt)
                generation = self.client.complete(system_prompt, user_prompt)
                record_model_result(role_generator, generation)
                raw_response_sha256 = _sha256_text(generation.raw_text)
                duplicate_of_attempt = next(
                    (
                        prior_index + 1
                        for prior_index, prior in enumerate(aipycraft_generations)
                        if _sha256_text(prior.raw_text) == raw_response_sha256
                    ),
                    None,
                )
                aipycraft_generations.append(generation)
                if duplicate_of_attempt is not None:
                    summary["duplicate_generation_responses"] += 1
                previous = generation.raw_text
                (attempt_dir / "raw_response.txt").write_text(
                    generation.raw_text, encoding="utf-8"
                )
                attempt["generation"] = {
                    **self._model_audit(generation, system_prompt, user_prompt),
                    "duplicate_of_attempt": duplicate_of_attempt,
                }
                _write_json(attempt_dir / "generation.json", attempt["generation"])

                if generation.finish_reason != "stop":
                    attempt["result"] = "incomplete_generation"
                    detail = (
                        "The provider returned finish_reason="
                        f"{generation.finish_reason or '<missing>'}; the response may be "
                        "incomplete and was not treated as a candidate manifest."
                    )
                    _write_json(
                        attempt_dir / "generation_completeness.json",
                        {"complete": False, "error": detail},
                    )
                    if index < self.config.pipeline.max_regenerations:
                        repair_failure = self._failure_text(
                            "generation_incomplete", detail
                        )
                        summary["regenerations"] += 1
                        _write_json(private_dir / "summary.json", summary)
                        continue
                    summary["status"] = "candidate_failed"
                    terminal = True
                    break

                syntax = check_yaml_syntax(generation.raw_text)
                attempt["pre_execution"] = syntax.audit_dict()
                _write_json(attempt_dir / "pre_execution.json", syntax.audit_dict())
                (attempt_dir / "candidate.yaml").write_text(
                    syntax.candidate, encoding="utf-8"
                )
                if not syntax.valid:
                    attempt["result"] = "yaml_syntax_error"
                    if index < self.config.pipeline.max_regenerations:
                        repair_failure = self._failure_text(
                            "yaml_syntax", syntax.error or "unknown YAML syntax error"
                        )
                        summary["regenerations"] += 1
                        _write_json(private_dir / "summary.json", summary)
                        continue
                    summary["status"] = "candidate_failed"
                    terminal = True
                    break

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
                    validation_result = self.validator_client.complete(
                        validator_system, validation_user
                    )
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
                        "status": "response_received",
                        "model_call": validation_audit,
                    }
                    attempt["ai_pre_validation"] = validation_record
                    _write_json(
                        attempt_dir / "ai_pre_validation.json", validation_record
                    )
                    if validation_result.finish_reason != "stop":
                        raise AiValidationError(
                            "AI validator returned finish_reason="
                            f"{validation_result.finish_reason or '<missing>'}"
                        )
                    decision = parse_decision(
                        validation_result.raw_text,
                        self.config.ai_validator.feedback_mode,
                    )
                    validation_record = {
                        **validation_record,
                        "status": "passed" if decision.satisfies else "rejected",
                        "decision": decision.audit_dict(),
                    }
                    attempt["ai_pre_validation"] = validation_record
                    _write_json(
                        attempt_dir / "ai_pre_validation.json", validation_record
                    )
                    summary["ai_pre_validation_public"] = {
                        "enabled": True,
                        "feedback_mode": self.config.ai_validator.feedback_mode,
                        "status": validation_record["status"],
                        "satisfies": decision.satisfies,
                        "problem_count": len(decision.problems),
                    }
                    if not decision.satisfies:
                        attempt["result"] = "ai_pre_validation_rejected"
                        if index < self.config.pipeline.max_regenerations:
                            repair_failure = self._failure_text(
                                "ai_validator",
                                correction_feedback(
                                    decision,
                                    self.config.ai_validator.feedback_mode,
                                ),
                            )
                            summary["regenerations"] += 1
                            _write_json(private_dir / "summary.json", summary)
                            continue
                        summary["status"] = "candidate_failed"
                        terminal = True
                        break
                else:
                    disabled_record = {
                        "enabled": False,
                        "feedback_mode": self.config.ai_validator.feedback_mode,
                        "status": "disabled",
                    }
                    attempt["ai_pre_validation"] = disabled_record
                    summary["ai_pre_validation_public"] = disabled_record
                    _write_json(
                        attempt_dir / "ai_pre_validation.json", disabled_record
                    )

                with self.harness.attempt(run_id, number, attempt_dir) as environment:
                    deployment = environment.deploy(attempt_dir / "candidate.yaml")
                    attempt["deployment"] = deployment.audit_dict()
                    _write_json(attempt_dir / "deployment.json", deployment.audit_dict())
                    if deployment.returncode != 0:
                        attempt["result"] = "deployment_ground_truth_failed"
                        summary["status"] = "execution_ground_truth_failed"
                        terminal = True
                        break

                    runtime = self.runtime_observer(
                        environment,
                        seconds=self.config.pipeline.runtime_observation_seconds,
                        poll_seconds=self.config.pipeline.runtime_poll_seconds,
                    )
                    attempt["runtime"] = runtime
                    _write_json(attempt_dir / "runtime.json", runtime)

                    execution = self.execution_verifier(
                        task,
                        environment,
                        timeout_seconds=self.config.pipeline.command_timeout_seconds,
                    )
                    attempt["execution_gate"] = execution
                    _write_json(attempt_dir / "execution_gate.json", execution)
                    summary["execution_gate_public"] = {
                        "status": execution.get("status"),
                        "checks": len(execution.get("checks", [])),
                        "failures": len(execution.get("failures", [])),
                        "repair_on_failure": False,
                        "task_specific": execution.get("task_specific", False),
                    }
                    if execution.get("status") not in {"passed", "failed"}:
                        attempt["result"] = "execution_gate_infrastructure_error"
                        summary["status"] = "infrastructure_error"
                        summary["fatal_error"] = execution.get(
                            "error", "Generic execution gate failed"
                        )
                        terminal = True
                        break
                    if runtime.get("failures") or execution.get("status") == "failed":
                        attempt["result"] = "execution_ground_truth_failed"
                        summary["status"] = "execution_ground_truth_failed"
                        terminal = True
                        break

                    post = self.post_verifier(task, environment)
                    attempt["post_execution"] = post
                    _write_json(attempt_dir / "post_execution.json", post)
                    summary["post_execution_public"] = self._public_post(post)
                    if post.get("status") == "passed":
                        attempt["result"] = "accepted"
                        summary["status"] = "completed"
                    elif post.get("status") == "failed":
                        attempt["result"] = "specification_discrepancy"
                        summary["status"] = "specification_discrepancy"
                    else:
                        attempt["result"] = "post_verifier_infrastructure_error"
                        summary["status"] = "infrastructure_error"
                        summary["fatal_error"] = post.get(
                            "error", "Hidden specification verifier failed"
                        )
                    terminal = True
                    break

            if not terminal:
                summary["status"] = "candidate_failed"
        except ProviderError as exc:
            summary["status"] = "provider_error"
            summary["fatal_error"] = str(exc)
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
            unknown_cost_by_role[active_role] += exc.unknown_cost_attempts
            summary["provider_failure"] = {
                "role": active_role,
                "transport_attempts": exc.transport_attempts,
                "unknown_cost_attempts": exc.unknown_cost_attempts,
            }
        except AiValidationError as exc:
            summary["status"] = "infrastructure_error"
            summary["fatal_error"] = str(exc)
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
        except EnvironmentError as exc:
            summary["status"] = "infrastructure_error"
            summary["fatal_error"] = str(exc)
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
            if attempts and attempts[-1]["result"] == "running":
                attempts[-1]["result"] = "pipeline_error"
                attempts[-1]["infrastructure_error"] = summary["fatal_error"]
                if current_attempt_dir is not None:
                    _write_json(
                        current_attempt_dir / "infrastructure_error.json",
                        {"error": summary["fatal_error"]},
                    )
        finally:
            if self.key_context_supplier:
                try:
                    summary["openrouter_key_after"] = self.key_context_supplier()
                except Exception as exc:
                    summary["openrouter_key_after_error"] = str(exc)
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
            summary["finished_at"] = _utc_now()
            summary["duration_ms"] = round((time.monotonic() - started) * 1000)
            _write_json(private_dir / "summary.json", summary)
        return PipelineRun(summary=summary, run_dir=run_dir)
