from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import yaml


_WHOLE_FENCE = re.compile(
    r"\A\s*```(?:yaml|yml)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class YamlCheck:
    valid: bool
    document_count: int
    normalization: str
    error: str | None
    candidate: str

    def audit_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("candidate")
        return value


def unwrap_whole_response_fence(raw: str) -> tuple[str, str]:
    match = _WHOLE_FENCE.fullmatch(raw)
    if not match:
        return raw, "none"
    return match.group(1), "whole-response-markdown-fence"


def check_yaml_syntax(raw: str) -> YamlCheck:
    """Check YAML grammar only; deliberately perform no Kubernetes validation."""

    candidate, normalization = unwrap_whole_response_fence(raw)
    try:
        # PRE-EXECUTION POLICY: Kubernetes schema, resource-kind, namespace,
        # security, dry-run, and requirement checks remain deliberately
        # disabled here. They are not silently performed by another parser.
        # Reusable operational checks run only after real isolated deployment.
        # compose_all checks the YAML representation graph without constructing
        # Python values. Unknown tags, scalars, empty streams, privileged Pods,
        # and Kubernetes-semantic nonsense therefore remain valid here.
        documents = list(yaml.compose_all(candidate))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = ""
        if mark is not None:
            location = f" at line {mark.line + 1}, column {mark.column + 1}"
        problem = getattr(exc, "problem", None) or str(exc)
        return YamlCheck(
            valid=False,
            document_count=0,
            normalization=normalization,
            error=f"YAML syntax error{location}: {problem}",
            candidate=candidate,
        )
    return YamlCheck(
        valid=True,
        document_count=len(documents),
        normalization=normalization,
        error=None,
        candidate=candidate,
    )
