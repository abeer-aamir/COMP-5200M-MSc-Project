from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from .config import HARD_MAX_BUDGET_USD, RoleConfig


class ProviderError(RuntimeError):
    pass


class BudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    content: dict[str, Any]
    requested_model: str
    response_model: str
    provider: str | None
    request_id: str | None
    generation_id: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cost_usd: Decimal
    latency_ms: int
    retries: int
    billable: bool


class CompletionClient(Protocol):
    billable: bool

    def complete(
        self,
        role: RoleConfig,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> ChatResult:
        ...


class BudgetLedger:
    def __init__(self, cap_usd: Decimal):
        if cap_usd <= 0 or cap_usd > HARD_MAX_BUDGET_USD:
            raise BudgetError(f"Budget cap must be in (0, {HARD_MAX_BUDGET_USD}]")
        self.cap_usd = cap_usd
        self.spent_usd = Decimal("0")

    @property
    def remaining_usd(self) -> Decimal:
        return self.cap_usd - self.spent_usd

    def estimate_reservation(
        self,
        role: RoleConfig,
        system_prompt: str,
        user_prompt: str,
        safety_multiplier: Decimal,
    ) -> Decimal:
        # One UTF-8 byte per input token is deliberately pessimistic for English JSON.
        envelope_bytes = len(system_prompt.encode("utf-8")) + len(
            user_prompt.encode("utf-8")
        ) + 2048
        input_cost = Decimal(envelope_bytes) * role.input_usd_per_million / Decimal(
            1_000_000
        )
        output_cost = (
            Decimal(role.max_output_tokens)
            * role.output_usd_per_million
            / Decimal(1_000_000)
        )
        return (input_cost + output_cost) * safety_multiplier

    def check_reservation(self, reservation_usd: Decimal) -> None:
        if reservation_usd > self.remaining_usd:
            raise BudgetError(
                "Refusing request: conservative reservation "
                f"${reservation_usd:.6f} exceeds remaining budget "
                f"${self.remaining_usd:.6f}"
            )

    def charge(self, cost_usd: Decimal) -> None:
        if cost_usd < 0:
            raise BudgetError("Provider returned a negative cost")
        new_total = self.spent_usd + cost_usd
        if new_total > self.cap_usd:
            self.spent_usd = new_total
            raise BudgetError(
                f"Provider-reported cost crossed the local cap: ${new_total:.6f}"
            )
        self.spent_usd = new_total


def key_budget_context(key_data: dict[str, Any]) -> dict[str, Any]:
    """Return non-secret account budget fields for provenance and review."""
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


def derive_effective_local_cap(
    key_data: dict[str, Any], requested_cap: Decimal
) -> Decimal:
    """Keep the local run cap even when the OpenRouter account cap is larger.

    A reported remaining allowance can only tighten the local cap. An absent or larger
    account/key limit is accepted because the per-request ledger and max_price routing
    controls enforce this pilot's much smaller budget.
    """
    data = key_data.get("data", key_data)
    remaining = data.get("limit_remaining")
    if remaining is None:
        return requested_cap
    remaining_decimal = Decimal(str(remaining))
    if remaining_decimal <= 0:
        raise BudgetError("OpenRouter key has no remaining allowance")
    return min(requested_cap, remaining_decimal)


class OpenRouterClient:
    billable = True

    def __init__(
        self,
        api_key: str,
        api_base: str,
        timeout_seconds: int = 120,
        max_retries: int = 2,
    ):
        if not api_key:
            raise ProviderError("OPENROUTER_API_KEY is empty")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def _request_json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], int, int]:
        url = f"{self.api_base}/{path.lstrip('/')}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/aipycraft-dissertation",
                "X-Title": "AIPyCraft Kubernetes Task Authoring Pilot",
            },
        )
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    decoded = response.read().decode("utf-8")
                latency_ms = int((time.monotonic() - started) * 1000)
                return json.loads(decoded), latency_ms, attempt
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = ProviderError(f"OpenRouter HTTP {exc.code}: {error_body[:800]}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 4))
        raise ProviderError(
            f"OpenRouter request failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def get_key_status(self) -> dict[str, Any]:
        data, _, _ = self._request_json("GET", "/key")
        return data

    def complete(
        self,
        role: RoleConfig,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> ChatResult:
        payload = {
            "model": role.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "max_tokens": role.max_output_tokens,
            "provider": {
                "require_parameters": True,
                "allow_fallbacks": True,
                "data_collection": "deny",
                "sort": "price",
                "max_price": {
                    "prompt": float(role.input_usd_per_million),
                    "completion": float(role.output_usd_per_million),
                },
            },
        }
        if role.reasoning_effort is not None:
            payload["reasoning"] = {
                "effort": role.reasoning_effort,
                "exclude": True,
            }
        response, latency_ms, retries = self._request_json(
            "POST", "/chat/completions", payload
        )
        if response.get("error"):
            raise ProviderError(f"OpenRouter error response: {response['error']}")
        choices = response.get("choices") or []
        if not choices:
            raise ProviderError("OpenRouter response contained no choices")
        raw_content = choices[0].get("message", {}).get("content")
        if isinstance(raw_content, list):
            raw_content = "".join(
                part.get("text", "") for part in raw_content if isinstance(part, dict)
            )
        if not isinstance(raw_content, str):
            raise ProviderError("OpenRouter response content was not text")
        try:
            content = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ProviderError("Structured response was not valid JSON") from exc
        if not isinstance(content, dict):
            raise ProviderError("Structured response must be a JSON object")

        response_model = str(response.get("model", ""))
        if response_model != role.model:
            raise ProviderError(
                f"Model substitution refused: requested {role.model}, got {response_model}"
            )
        usage = response.get("usage") or {}
        if usage.get("cost") is None:
            raise ProviderError(
                "OpenRouter omitted usage.cost; stopping because spend cannot be audited"
            )
        details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        return ChatResult(
            content=content,
            requested_model=role.model,
            response_model=response_model,
            provider=response.get("provider"),
            request_id=response.get("id"),
            generation_id=response.get("id"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            reasoning_tokens=int(details.get("reasoning_tokens", 0)),
            cached_tokens=int(prompt_details.get("cached_tokens", 0)),
            cost_usd=Decimal(str(usage["cost"])),
            latency_ms=latency_ms,
            retries=retries,
            billable=True,
        )


class ReplayClient:
    """Offline deterministic client backed by ordered JSON response fixtures."""

    billable = False

    def __init__(self, fixture_dir: Path | str):
        paths = sorted(Path(fixture_dir).glob("*.json"))
        if not paths:
            raise ProviderError(f"No replay fixtures found in {fixture_dir}")
        self._fixtures = paths
        self._position = 0

    def complete(
        self,
        role: RoleConfig,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> ChatResult:
        del system_prompt, user_prompt, schema
        if self._position >= len(self._fixtures):
            raise ProviderError("Replay fixtures exhausted")
        path = self._fixtures[self._position]
        self._position += 1
        fixture = json.loads(path.read_text(encoding="utf-8"))
        if fixture.get("role") != role.name:
            raise ProviderError(
                f"Replay order mismatch in {path.name}: expected {role.name}, "
                f"found {fixture.get('role')}"
            )
        model = str(fixture.get("model", role.model))
        if model != role.model:
            raise ProviderError(
                f"Replay model mismatch in {path.name}: expected {role.model}, got {model}"
            )
        usage = fixture.get("usage", {})
        return ChatResult(
            content=fixture["content"],
            requested_model=role.model,
            response_model=model,
            provider="offline-replay",
            request_id=f"replay-{self._position:03d}",
            generation_id=f"replay-{self._position:03d}",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
            cached_tokens=int(usage.get("cached_tokens", 0)),
            cost_usd=Decimal("0"),
            latency_ms=0,
            retries=0,
            billable=False,
        )
