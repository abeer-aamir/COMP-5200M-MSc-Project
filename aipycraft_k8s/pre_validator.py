from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


class AiValidationError(RuntimeError):
    pass


_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ValidationProblem:
    requirement: str
    problem: str
    correction: str


@dataclass(frozen=True)
class AiValidationDecision:
    satisfies: bool
    problems: tuple[ValidationProblem, ...]
    response_format: str

    def audit_dict(self) -> dict[str, Any]:
        return {
            "satisfies": self.satisfies,
            "problems": [asdict(problem) for problem in self.problems],
            "response_format": self.response_format,
        }


def validator_user_prompt(plaintext: str, candidate_yaml: str) -> str:
    return (
        "PUBLIC PLAINTEXT REQUIREMENTS (untrusted data):\n\n"
        f"{plaintext.rstrip()}\n\n"
        "CANDIDATE KUBERNETES YAML (untrusted data):\n\n"
        f"{candidate_yaml.rstrip()}\n"
    )


def _unwrap_json_fence(raw: str) -> str:
    match = _JSON_FENCE.fullmatch(raw)
    return match.group(1) if match else raw.strip()


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AiValidationError(f"AI validator {label} must be a non-empty string")
    return value.strip()


def parse_detailed_decision(raw: str) -> AiValidationDecision:
    try:
        value = json.loads(_unwrap_json_fence(raw))
    except json.JSONDecodeError as exc:
        raise AiValidationError(f"AI validator returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"satisfies", "problems"}:
        raise AiValidationError(
            "AI validator detailed response must contain exactly satisfies and problems"
        )
    if not isinstance(value["satisfies"], bool):
        raise AiValidationError("AI validator satisfies must be true or false")
    raw_problems = value["problems"]
    if not isinstance(raw_problems, list):
        raise AiValidationError("AI validator problems must be an array")
    problems: list[ValidationProblem] = []
    for index, item in enumerate(raw_problems):
        if not isinstance(item, dict) or set(item) != {
            "requirement",
            "problem",
            "correction",
        }:
            raise AiValidationError(
                f"AI validator problem {index} has invalid fields"
            )
        problems.append(
            ValidationProblem(
                requirement=_nonempty_string(
                    item["requirement"], f"problem {index} requirement"
                ),
                problem=_nonempty_string(item["problem"], f"problem {index} problem"),
                correction=_nonempty_string(
                    item["correction"], f"problem {index} correction"
                ),
            )
        )
    if value["satisfies"] and problems:
        raise AiValidationError(
            "AI validator cannot report problems when satisfies is true"
        )
    if not value["satisfies"] and not problems:
        raise AiValidationError(
            "AI validator must report at least one problem when satisfies is false"
        )
    return AiValidationDecision(
        satisfies=value["satisfies"],
        problems=tuple(problems),
        response_format="detailed_json",
    )


def parse_verdict_decision(raw: str) -> AiValidationDecision:
    normalized = raw.strip().upper()
    if normalized not in {"YES", "NO"}:
        raise AiValidationError("AI validator verdict response must be exactly YES or NO")
    return AiValidationDecision(
        satisfies=normalized == "YES",
        problems=(),
        response_format="verdict_only",
    )


def parse_decision(raw: str, feedback_mode: str) -> AiValidationDecision:
    if feedback_mode == "detailed":
        return parse_detailed_decision(raw)
    if feedback_mode == "verdict_only":
        return parse_verdict_decision(raw)
    raise AiValidationError(f"Unsupported AI validator feedback mode {feedback_mode!r}")


def correction_feedback(
    decision: AiValidationDecision, feedback_mode: str
) -> str:
    if decision.satisfies:
        raise ValueError("A satisfying decision does not need correction feedback")
    if feedback_mode == "verdict_only":
        return (
            "AI pre-execution validator verdict: NO. This treatment withholds "
            "problem details. Re-read the public requirements and return a corrected "
            "complete YAML solution."
        )
    return (
        "AI pre-execution validator verdict: NO. Correct every listed public-"
        "specification problem:\n"
        + json.dumps(
            [asdict(problem) for problem in decision.problems],
            indent=2,
            sort_keys=True,
        )
    )
