from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from .config import ApiConfig


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        transport_attempts: int = 0,
        unknown_cost_attempts: int = 0,
        audit_result: GenerationResult | None = None,
        transport_attempt_log: tuple[dict[str, Any], ...] = (),
        provider_unavailable_retries: int = 0,
    ):
        super().__init__(message)
        self.transport_attempts = transport_attempts
        self.unknown_cost_attempts = unknown_cost_attempts
        self.unknown_cost_possible = unknown_cost_attempts > 0
        self.audit_result = audit_result
        self.transport_attempt_log = transport_attempt_log
        self.provider_unavailable_retries = provider_unavailable_retries


@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    requested_model: str
    response_model: str
    provider: str | None
    generation_id: str | None
    request_id: str | None
    system_fingerprint: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cost_usd: Decimal
    cost_source: str
    provider_cost_complete: bool
    usage_complete: bool
    usage_raw: dict[str, Any] | None
    latency_ms: int
    transport_retries: int
    unobserved_billable_attempts: int
    billable: bool
    transport_attempt_log: tuple[dict[str, Any], ...] = ()
    provider_response_attempts: int = 1
    http_post_attempts: int = 1
    provider_response_retry_log: tuple[dict[str, Any], ...] = ()
    provider_unavailable_retries: int = 0
    empty_length_provider_retries: int = 0

    def usage_consistency_issues(self) -> list[str]:
        """Return conservative local checks without replacing provider accounting."""

        if not self.billable:
            return []
        response_bytes = len(self.raw_text.encode("utf-8"))
        issues: list[str] = []
        if response_bytes and self.completion_tokens == 0:
            issues.append("nonempty_response_with_zero_completion_tokens")
        elif (
            self.completion_tokens > 0
            and response_bytes > self.completion_tokens * 64
        ):
            issues.append("response_bytes_implausibly_high_for_completion_tokens")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            issues.append("total_tokens_not_prompt_plus_completion")
        if self.reasoning_tokens > self.completion_tokens:
            issues.append("reasoning_tokens_exceed_completion_tokens")
        if self.cached_tokens > self.prompt_tokens:
            issues.append("cached_tokens_exceed_prompt_tokens")
        return issues

    def audit_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("raw_text")
        value["cost_usd"] = str(self.cost_usd)
        response_bytes = len(self.raw_text.encode("utf-8"))
        issues = self.usage_consistency_issues()
        value["local_response_metrics"] = {
            "characters": len(self.raw_text),
            "utf8_bytes": response_bytes,
            "lines": len(self.raw_text.splitlines()),
        }
        value["provider_usage_consistency"] = {
            "status": (
                "not_applicable"
                if not self.billable
                else "suspicious" if issues else "plausible"
            ),
            "issues": issues,
            "note": "Local checks do not replace provider-reported billing fields.",
        }
        return value


class TextGenerationClient(Protocol):
    billable: bool

    def complete(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        ...


def safe_key_budget_context(key_data: dict[str, Any]) -> dict[str, Any]:
    """Retain only non-secret budget and usage fields from OpenRouter /key."""

    data = key_data.get("data", key_data)
    fields = (
        "limit",
        "limit_remaining",
        "limit_reset",
        "usage",
        "usage_daily",
        "usage_weekly",
        "usage_monthly",
    )
    return {field: data.get(field) for field in fields}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    raise ProviderError("OpenRouter response content was not text")


def _provider_identity(value: str) -> str:
    """Normalize OpenRouter provider slugs and display names for auditing."""

    return "".join(character for character in value.casefold() if character.isalnum())


def _nonnegative_int(value: Any) -> tuple[int, bool]:
    if isinstance(value, bool):
        return 0, False
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0, False
    return (parsed, True) if parsed >= 0 else (0, False)


class OpenRouterTextClient:
    billable = True

    def __init__(
        self,
        api_key: str,
        config: ApiConfig,
        *,
        timeout_seconds: int = 180,
        retry_empty_length_with_alternate_provider: bool = False,
        retry_provider_unavailable: bool = True,
        provider_unavailable_retry_delay_seconds: float = 2.0,
    ):
        if not api_key:
            raise ProviderError("OPENROUTER_API_KEY is empty")
        self.api_key = api_key
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.retry_empty_length_with_alternate_provider = (
            retry_empty_length_with_alternate_provider
        )
        self.retry_provider_unavailable = retry_provider_unavailable
        self.provider_unavailable_retry_delay_seconds = max(
            0.0, provider_unavailable_retry_delay_seconds
        )

    def _request_json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[
        dict[str, Any],
        int,
        int,
        str | None,
        int,
        tuple[dict[str, Any], ...],
    ]:
        url = f"{self.config.base_url}/{path.lstrip('/')}"
        encoded = None if payload is None else json.dumps(payload).encode("utf-8")
        started = time.monotonic()
        last_error: Exception | None = None
        uncertain_attempts = 0
        attempt_log: list[dict[str, Any]] = []
        attempts = self.config.transport_retries + 1
        for attempt in range(attempts):
            attempt_started = time.monotonic()
            attempt_record: dict[str, Any] = {
                "attempt": attempt + 1,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "method": method.upper(),
                "path": path,
            }
            request = urllib.request.Request(
                url,
                data=encoded,
                method=method,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://localhost/aipycraft-dissertation",
                    "X-OpenRouter-Title": "AIPyCraft Kubernetes Baseline",
                },
            )
            retry_after = 1.0
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    decoded = response.read().decode("utf-8")
                    request_id = response.headers.get("x-request-id")
                result = json.loads(decoded)
                if not isinstance(result, dict):
                    raise json.JSONDecodeError("response was not an object", decoded, 0)
                attempt_record.update(
                    {
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "duration_ms": round(
                            (time.monotonic() - attempt_started) * 1000
                        ),
                        "outcome": "completed",
                        "http_status": getattr(response, "status", None),
                        "retry_scheduled": False,
                        "retry_delay_ms": 0,
                    }
                )
                attempt_log.append(attempt_record)
                return (
                    result,
                    round((time.monotonic() - started) * 1000),
                    attempt,
                    request_id,
                    uncertain_attempts,
                    tuple(attempt_log),
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = ProviderError(f"OpenRouter HTTP {exc.code}: {body[:1200]}")
                header = exc.headers.get("Retry-After") if exc.headers else None
                if header:
                    try:
                        retry_after = min(max(float(header), 1.0), 5.0)
                    except ValueError:
                        retry_after = 1.0
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    retry_after = 0
                if method.upper() == "POST" and exc.code == 408:
                    uncertain_attempts += 1
                attempt_record.update(
                    {
                        "outcome": "http_error",
                        "http_status": exc.code,
                        "error_type": type(last_error).__name__,
                    }
                )
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                uncertain_attempts += int(method.upper() == "POST")
                attempt_record.update(
                    {"outcome": "transport_error", "error_type": type(exc).__name__}
                )
            except json.JSONDecodeError as exc:
                last_error = exc
                uncertain_attempts += int(method.upper() == "POST")
                attempt_record.update(
                    {"outcome": "response_parse_error", "error_type": type(exc).__name__}
                )
            retry_scheduled = attempt + 1 < attempts and retry_after > 0
            attempt_record.update(
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(
                        (time.monotonic() - attempt_started) * 1000
                    ),
                    "retry_scheduled": retry_scheduled,
                    "retry_delay_ms": round(retry_after * 1000) if retry_scheduled else 0,
                }
            )
            attempt_log.append(attempt_record)
            if retry_scheduled:
                retry_sleep_started = time.monotonic()
                time.sleep(retry_after)
                attempt_record["retry_sleep_duration_ms"] = round(
                    (time.monotonic() - retry_sleep_started) * 1000
                )
            else:
                break
        raise ProviderError(
            f"OpenRouter request failed after {min(attempt + 1, attempts)} attempt(s): "
            f"{last_error}",
            transport_attempts=min(attempt + 1, attempts),
            unknown_cost_attempts=uncertain_attempts,
            transport_attempt_log=tuple(attempt_log),
        ) from last_error

    def get_key_status(self) -> dict[str, Any]:
        return self._request_json("GET", "/key")[0]

    def request_payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "provider": {
                "only": list(self.config.provider_only),
                "allow_fallbacks": self.config.allow_fallbacks,
                "require_parameters": True,
                "data_collection": "deny",
            },
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        return payload

    def _audit_response(
        self,
        response: dict[str, Any],
        *,
        latency_ms: int,
        retries: int,
        header_request_id: str | None,
        uncertain_attempts: int,
        transport_attempt_log: tuple[dict[str, Any], ...],
    ) -> GenerationResult:
        choices = response.get("choices")
        choice = (
            choices[0]
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else {}
        )
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        try:
            raw_text = _message_text(content)
        except ProviderError:
            raw_text = ""

        usage_value = response.get("usage")
        usage = usage_value if isinstance(usage_value, dict) else {}
        prompt_tokens, prompt_ok = _nonnegative_int(usage.get("prompt_tokens"))
        completion_tokens, completion_ok = _nonnegative_int(
            usage.get("completion_tokens")
        )
        total_tokens, total_ok = _nonnegative_int(usage.get("total_tokens"))
        if not total_ok:
            total_tokens = prompt_tokens + completion_tokens
        completion_details = usage.get("completion_tokens_details")
        if not isinstance(completion_details, dict):
            completion_details = {}
        prompt_details = usage.get("prompt_tokens_details")
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        reasoning_tokens, reasoning_ok = _nonnegative_int(
            completion_details.get("reasoning_tokens", 0)
        )
        cached_tokens, cached_ok = _nonnegative_int(
            prompt_details.get("cached_tokens", 0)
        )
        usage_complete = (
            isinstance(usage_value, dict)
            and prompt_ok
            and completion_ok
            and total_ok
            and reasoning_ok
            and cached_ok
        )

        provider_cost = usage.get("cost")
        provider_cost_ok = False
        if provider_cost is not None:
            try:
                cost = Decimal(str(provider_cost))
                provider_cost_ok = cost.is_finite() and cost >= 0
            except (InvalidOperation, ValueError):
                provider_cost_ok = False
            if not provider_cost_ok:
                cost = Decimal("0")
        else:
            cost = Decimal("0")
        if provider_cost_ok:
            cost_source = "provider_reported"
        else:
            cost = (
                Decimal(prompt_tokens) * self.config.input_usd_per_million
                + Decimal(completion_tokens) * self.config.output_usd_per_million
            ) / Decimal(1_000_000)
            cost_source = "estimated_from_locked_prices"

        generation_id = response.get("id")
        response_model = response.get("model")
        provider = response.get("provider")
        fingerprint = response.get("system_fingerprint")
        sanitized_usage = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost": usage.get("cost"),
            "completion_tokens_details": completion_details,
            "prompt_tokens_details": prompt_details,
        }
        return GenerationResult(
            raw_text=raw_text,
            requested_model=self.config.model,
            response_model=str(response_model) if response_model is not None else "",
            provider=str(provider) if provider is not None else None,
            generation_id=str(generation_id) if generation_id is not None else None,
            request_id=header_request_id
            or (str(generation_id) if generation_id is not None else None),
            system_fingerprint=(
                str(fingerprint) if fingerprint is not None else None
            ),
            finish_reason=(
                str(choice.get("finish_reason"))
                if choice.get("finish_reason") is not None
                else None
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            cost_source=cost_source,
            provider_cost_complete=provider_cost_ok and uncertain_attempts == 0,
            usage_complete=usage_complete,
            usage_raw=sanitized_usage,
            latency_ms=latency_ms,
            transport_retries=retries,
            unobserved_billable_attempts=uncertain_attempts,
            billable=True,
            transport_attempt_log=transport_attempt_log,
            provider_response_attempts=1,
            http_post_attempts=(
                len(transport_attempt_log) if transport_attempt_log else retries + 1
            ),
        )

    @staticmethod
    def _provider_attempt_record(
        result: GenerationResult,
        *,
        attempt: int,
        routing: str,
        trigger: str | None = None,
        excluded_providers: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "attempt": attempt,
            "routing": routing,
            "provider": result.provider,
            "generation_id": result.generation_id,
            "request_id": result.request_id,
            "finish_reason": result.finish_reason,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "total_tokens": result.total_tokens,
            "cost_usd": str(result.cost_usd),
            "latency_ms": result.latency_ms,
            "response_characters": len(result.raw_text),
            "http_post_attempts": result.http_post_attempts,
        }
        if trigger is not None:
            record["trigger"] = trigger
        if excluded_providers:
            record["excluded_providers"] = list(excluded_providers)
        return record

    @classmethod
    def _combine_provider_attempts(
        cls,
        first: GenerationResult,
        second: GenerationResult,
        *,
        trigger: str,
        routing: str,
        excluded_providers: tuple[str, ...] = (),
    ) -> GenerationResult:
        transport_log: list[dict[str, Any]] = []
        for application_attempt, result in ((1, first), (2, second)):
            for item in result.transport_attempt_log:
                annotated = dict(item)
                annotated["provider_response_attempt"] = application_attempt
                annotated["response_provider"] = result.provider
                if application_attempt == 2:
                    annotated["provider_retry_trigger"] = trigger
                    annotated["excluded_providers"] = list(excluded_providers)
                transport_log.append(annotated)
        return GenerationResult(
            raw_text=second.raw_text,
            requested_model=second.requested_model,
            response_model=second.response_model,
            provider=second.provider,
            generation_id=second.generation_id,
            request_id=second.request_id,
            system_fingerprint=second.system_fingerprint,
            finish_reason=second.finish_reason,
            prompt_tokens=first.prompt_tokens + second.prompt_tokens,
            completion_tokens=first.completion_tokens + second.completion_tokens,
            total_tokens=first.total_tokens + second.total_tokens,
            reasoning_tokens=first.reasoning_tokens + second.reasoning_tokens,
            cached_tokens=first.cached_tokens + second.cached_tokens,
            cost_usd=first.cost_usd + second.cost_usd,
            cost_source="aggregated_provider_retry",
            provider_cost_complete=(
                first.provider_cost_complete and second.provider_cost_complete
            ),
            usage_complete=first.usage_complete and second.usage_complete,
            usage_raw={
                "provider_response_attempts": [first.usage_raw, second.usage_raw]
            },
            latency_ms=first.latency_ms + second.latency_ms,
            transport_retries=first.transport_retries + second.transport_retries,
            unobserved_billable_attempts=(
                first.unobserved_billable_attempts
                + second.unobserved_billable_attempts
            ),
            billable=first.billable or second.billable,
            transport_attempt_log=tuple(transport_log),
            provider_response_attempts=(
                first.provider_response_attempts
                + second.provider_response_attempts
            ),
            http_post_attempts=first.http_post_attempts + second.http_post_attempts,
            provider_response_retry_log=(
                first.provider_response_retry_log
                or (
                    cls._provider_attempt_record(
                        first,
                        attempt=1,
                        routing="configured_provider",
                    ),
                )
            )
            + (
                cls._provider_attempt_record(
                    second,
                    attempt=first.provider_response_attempts + 1,
                    routing=routing,
                    trigger=trigger,
                    excluded_providers=excluded_providers,
                ),
            ),
            provider_unavailable_retries=(
                first.provider_unavailable_retries
                + second.provider_unavailable_retries
                + int(trigger == "provider_unavailable_after_backoff")
            ),
            empty_length_provider_retries=(
                first.empty_length_provider_retries
                + second.empty_length_provider_retries
                + int(trigger == "empty_response_with_finish_reason_length")
            ),
        )

    @staticmethod
    def _retryable_error_response(response: dict[str, Any]) -> bool:
        error = response.get("error")
        if not isinstance(error, dict):
            return False

        retryable_codes = {408, 409, 429, 500, 502, 503, 504}

        def values(value: Any) -> list[Any]:
            if isinstance(value, dict):
                result: list[Any] = []
                for child in value.values():
                    result.extend(values(child))
                return result
            if isinstance(value, list):
                result = []
                for child in value:
                    result.extend(values(child))
                return result
            return [value]

        flattened = values(error)
        if any(
            (isinstance(value, int) and value in retryable_codes)
            or (isinstance(value, str) and value.isdigit() and int(value) in retryable_codes)
            for value in flattened
        ):
            return True
        text = json.dumps(error, ensure_ascii=False).casefold()
        return any(
            marker in text
            for marker in (
                "no available provider",
                "provider unavailable",
                "temporarily unavailable",
                "rate limit",
                "no endpoints found",
                "upstream timeout",
            )
        )

    @staticmethod
    def _retryable_provider_exception(exc: ProviderError) -> bool:
        text = str(exc).casefold()
        nonretryable_statuses = ("http 400", "http 401", "http 402", "http 403", "http 404")
        if any(marker in text for marker in nonretryable_statuses):
            return False
        return any(
            marker in text
            for marker in (
                "http 408",
                "http 409",
                "http 429",
                "http 500",
                "http 502",
                "http 503",
                "http 504",
                "timed out",
                "urlerror",
                "temporarily unavailable",
                "connection reset",
                "connection refused",
                "name or service not known",
                "getaddrinfo failed",
            )
        )

    def _provider_unavailable_retry(
        self,
        payload: dict[str, Any],
        *,
        first_audit: GenerationResult | None = None,
        first_error: ProviderError | None = None,
    ) -> tuple[dict[str, Any], GenerationResult]:
        delay_started = time.monotonic()
        time.sleep(self.provider_unavailable_retry_delay_seconds)
        retry_sleep_ms = round((time.monotonic() - delay_started) * 1000)
        try:
            response, audit = self._audit_request(payload)
        except ProviderError as exc:
            prior_posts = (
                first_audit.http_post_attempts
                if first_audit is not None
                else first_error.transport_attempts if first_error is not None else 0
            )
            prior_unknown = (
                first_audit.unobserved_billable_attempts
                if first_audit is not None
                else first_error.unknown_cost_attempts if first_error is not None else 0
            )
            prior_log = (
                first_audit.transport_attempt_log
                if first_audit is not None
                else first_error.transport_attempt_log if first_error is not None else ()
            )
            failed_audit = first_audit
            if failed_audit is not None:
                failed_audit = replace(
                    failed_audit,
                    http_post_attempts=prior_posts + exc.transport_attempts,
                    unobserved_billable_attempts=prior_unknown + exc.unknown_cost_attempts,
                    provider_cost_complete=False,
                    transport_attempt_log=prior_log + exc.transport_attempt_log,
                    provider_unavailable_retries=(
                        failed_audit.provider_unavailable_retries + 1
                    ),
                    provider_response_retry_log=(
                        failed_audit.provider_response_retry_log
                        or (
                            self._provider_attempt_record(
                                failed_audit,
                                attempt=1,
                                routing="configured_provider",
                            ),
                        )
                    )
                    + (
                        {
                            "attempt": failed_audit.provider_response_attempts + 1,
                            "routing": "configured_provider_retry",
                            "trigger": "provider_unavailable_after_backoff",
                            "outcome": "transport_error",
                            "error": str(exc),
                            "retry_sleep_duration_ms": retry_sleep_ms,
                            "http_post_attempts": exc.transport_attempts,
                        },
                    ),
                )
            raise ProviderError(
                "Provider-unavailable retry failed after backoff: " f"{exc}",
                transport_attempts=prior_posts + exc.transport_attempts,
                unknown_cost_attempts=prior_unknown + exc.unknown_cost_attempts,
                audit_result=failed_audit,
                transport_attempt_log=prior_log + exc.transport_attempt_log,
                provider_unavailable_retries=1,
            ) from exc

        if first_audit is not None:
            combined = self._combine_provider_attempts(
                first_audit,
                audit,
                trigger="provider_unavailable_after_backoff",
                routing="configured_provider_retry",
            )
            retry_log = list(combined.provider_response_retry_log)
            retry_log[-1]["retry_sleep_duration_ms"] = retry_sleep_ms
            combined = replace(combined, provider_response_retry_log=tuple(retry_log))
            return response, combined

        if first_error is None:
            raise AssertionError("provider retry requires a prior response or error")
        annotated_log: list[dict[str, Any]] = []
        for item in first_error.transport_attempt_log:
            annotated = dict(item)
            annotated["provider_response_attempt"] = 1
            annotated_log.append(annotated)
        for item in audit.transport_attempt_log:
            annotated = dict(item)
            annotated["provider_response_attempt"] = 2
            annotated["provider_retry_trigger"] = "provider_unavailable_after_backoff"
            annotated_log.append(annotated)
        return response, replace(
            audit,
            provider_cost_complete=(
                audit.provider_cost_complete and first_error.unknown_cost_attempts == 0
            ),
            transport_retries=(
                audit.transport_retries + max(first_error.transport_attempts - 1, 0)
            ),
            unobserved_billable_attempts=(
                audit.unobserved_billable_attempts + first_error.unknown_cost_attempts
            ),
            transport_attempt_log=tuple(annotated_log),
            http_post_attempts=audit.http_post_attempts + first_error.transport_attempts,
            provider_unavailable_retries=audit.provider_unavailable_retries + 1,
            provider_response_retry_log=(
                {
                    "attempt": 1,
                    "routing": "configured_provider",
                    "trigger": "provider_unavailable_after_transport_backoff",
                    "outcome": "transport_error",
                    "error": str(first_error),
                    "http_post_attempts": first_error.transport_attempts,
                },
                {
                    **self._provider_attempt_record(
                        audit,
                        attempt=2,
                        routing="configured_provider_retry",
                        trigger="provider_unavailable_after_backoff",
                    ),
                    "retry_sleep_duration_ms": retry_sleep_ms,
                },
            ),
        )

    def _audit_request(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], GenerationResult]:
        request_result = self._request_json("POST", "/chat/completions", payload)
        response, latency_ms, retries, header_request_id, uncertain_attempts = (
            request_result[:5]
        )
        transport_attempt_log = request_result[5] if len(request_result) > 5 else ()
        return response, self._audit_response(
            response,
            latency_ms=latency_ms,
            retries=retries,
            header_request_id=header_request_id,
            uncertain_attempts=uncertain_attempts,
            transport_attempt_log=transport_attempt_log,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        payload = self.request_payload(system_prompt, user_prompt)
        try:
            response, audit = self._audit_request(payload)
        except ProviderError as exc:
            if not self.retry_provider_unavailable or not self._retryable_provider_exception(exc):
                raise
            response, audit = self._provider_unavailable_retry(
                payload,
                first_error=exc,
            )

        if self.retry_provider_unavailable and self._retryable_error_response(response):
            response, audit = self._provider_unavailable_retry(
                payload,
                first_audit=audit,
            )

        def reject(message: str, result: GenerationResult = audit) -> None:
            raise ProviderError(
                message,
                transport_attempts=result.http_post_attempts,
                unknown_cost_attempts=result.unobserved_billable_attempts,
                audit_result=result,
                transport_attempt_log=result.transport_attempt_log,
                provider_unavailable_retries=result.provider_unavailable_retries,
            )

        if response.get("error"):
            reject(f"OpenRouter error response: {response['error']}")
        choices = response.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            reject("OpenRouter response contained no choices")
        if audit.response_model != self.config.model:
            reject(
                f"Model substitution refused: requested {self.config.model}, "
                f"got {audit.response_model or '<missing>'}"
            )
        if audit.provider is None or _provider_identity(audit.provider) not in {
            _provider_identity(value) for value in self.config.provider_only
        }:
            reject(
                "Provider substitution refused: requested "
                f"{list(self.config.provider_only)}, got {audit.provider or '<missing>'}"
            )
        retry_eligible = (
            self.retry_empty_length_with_alternate_provider
            and audit.finish_reason == "length"
            and not audit.raw_text
        )
        if retry_eligible:
            excluded_providers = tuple(self.config.provider_only)
            retry_payload = self.request_payload(system_prompt, user_prompt)
            retry_payload["provider"] = {
                "ignore": list(excluded_providers),
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
            try:
                retry_response, retry_audit = self._audit_request(retry_payload)
            except ProviderError as exc:
                raise ProviderError(
                    "Alternate-provider retry failed after empty length response: "
                    f"{exc}",
                    transport_attempts=(
                        audit.http_post_attempts + exc.transport_attempts
                    ),
                    unknown_cost_attempts=(
                        audit.unobserved_billable_attempts
                        + exc.unknown_cost_attempts
                    ),
                    audit_result=audit,
                    transport_attempt_log=(
                        audit.transport_attempt_log + exc.transport_attempt_log
                    ),
                ) from exc
            combined = self._combine_provider_attempts(
                audit,
                retry_audit,
                trigger="empty_response_with_finish_reason_length",
                routing="alternate_provider",
                excluded_providers=excluded_providers,
            )
            if retry_response.get("error"):
                reject(
                    "OpenRouter alternate-provider error response: "
                    f"{retry_response['error']}",
                    combined,
                )
            retry_choices = retry_response.get("choices")
            if (
                not isinstance(retry_choices, list)
                or not retry_choices
                or not isinstance(retry_choices[0], dict)
            ):
                reject(
                    "OpenRouter alternate-provider response contained no choices",
                    combined,
                )
            if retry_audit.response_model != self.config.model:
                reject(
                    "Model substitution refused during alternate-provider retry: "
                    f"requested {self.config.model}, got "
                    f"{retry_audit.response_model or '<missing>'}",
                    combined,
                )
            excluded_identities = {
                _provider_identity(value) for value in excluded_providers
            }
            if (
                retry_audit.provider is None
                or _provider_identity(retry_audit.provider) in excluded_identities
            ):
                reject(
                    "Alternate-provider retry did not use a different provider: "
                    f"excluded {list(excluded_providers)}, got "
                    f"{retry_audit.provider or '<missing>'}",
                    combined,
                )
            if retry_audit.finish_reason == "length" and not retry_audit.raw_text:
                reject(
                    "OpenRouter alternate-provider retry also returned "
                    "finish_reason=length with no answer",
                    combined,
                )
            try:
                _message_text(
                    (retry_choices[0].get("message") or {}).get("content")
                )
            except (AttributeError, ProviderError) as exc:
                reject(
                    "OpenRouter alternate-provider response content was not text: "
                    f"{exc}",
                    combined,
                )
            return combined
        choice = choices[0]
        try:
            _message_text((choice.get("message") or {}).get("content"))
        except (AttributeError, ProviderError) as exc:
            reject(f"OpenRouter response content was not text: {exc}")
        return audit


class ReplayTextClient:
    """Deterministic, no-network model replacement backed by ordered text files."""

    billable = False

    def __init__(self, fixture_dir: Path | str, model: str):
        self.paths = sorted(Path(fixture_dir).glob("*.txt"))
        if not self.paths:
            raise ProviderError(f"No replay .txt files found in {fixture_dir}")
        self.model = model
        self.position = 0

    def complete(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        del system_prompt, user_prompt
        if self.position >= len(self.paths):
            raise ProviderError("Replay fixtures exhausted")
        path = self.paths[self.position]
        self.position += 1
        return GenerationResult(
            raw_text=path.read_text(encoding="utf-8"),
            requested_model=self.model,
            response_model=self.model,
            provider="offline-replay",
            generation_id=f"replay-{self.position:03d}",
            request_id=f"replay-{self.position:03d}",
            system_fingerprint=None,
            finish_reason="stop",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            reasoning_tokens=0,
            cached_tokens=0,
            cost_usd=Decimal("0"),
            cost_source="offline_replay",
            provider_cost_complete=True,
            usage_complete=True,
            usage_raw=None,
            latency_ms=0,
            transport_retries=0,
            unobserved_billable_attempts=0,
            billable=False,
        )
