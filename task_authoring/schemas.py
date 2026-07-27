from __future__ import annotations

import re
from typing import Any

from .config import PilotConfig


CATEGORIES = (
    "multi_resource_dependencies",
    "configuration_wiring",
    "service_networking",
    "readiness_and_health",
    "rbac",
    "pod_security",
    "network_policy",
    "stateful_storage",
    "jobs_and_batch",
    "scheduling",
    "runtime_behavior",
    "availability_and_disruption",
)

ALLOWED_RESOURCE_KINDS = {
    "ConfigMap",
    "Secret",
    "ServiceAccount",
    "Role",
    "RoleBinding",
    "Service",
    "Deployment",
    "StatefulSet",
    "DaemonSet",
    "Job",
    "CronJob",
    "PersistentVolumeClaim",
    "NetworkPolicy",
    "PodDisruptionBudget",
    "ResourceQuota",
    "LimitRange",
}

CHECK_NAMES = (
    "no_easy_shortcut",
    "no_ambiguity",
    "no_contradiction",
    "no_private_leakage",
    "all_requirements_covered",
    "kubernetes_1_35_compatible",
)


SPEC_SCHEMA: dict[str, Any] = {
    "name": "kubernetes_private_spec",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "task_id",
            "title",
            "scenario",
            "target_kubernetes_version",
            "categories",
            "public_requirements",
            "resource_kinds",
            "runtime_behaviors",
            "safety_constraints",
            "hardness_rationale",
            "likely_failure_modes",
        ],
        "properties": {
            "task_id": {"type": "string", "pattern": "^pilot-[0-9]{3}$"},
            "title": {"type": "string", "minLength": 8, "maxLength": 100},
            "scenario": {"type": "string", "minLength": 80, "maxLength": 1000},
            "target_kubernetes_version": {"type": "string", "const": "1.35"},
            "categories": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": list(CATEGORIES)},
            },
            "public_requirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "text", "depends_on", "verification"],
                    "properties": {
                        "id": {"type": "string", "pattern": "^R[0-9]{2}$"},
                        "text": {"type": "string", "minLength": 20, "maxLength": 500},
                        "depends_on": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {"type": "string", "pattern": "^R[0-9]{2}$"},
                        },
                        "verification": {
                            "type": "string",
                            "enum": [
                                "manifest",
                                "runtime",
                                "security",
                                "connectivity",
                                "lifecycle",
                            ],
                        },
                    },
                },
            },
            "resource_kinds": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(ALLOWED_RESOURCE_KINDS)},
            },
            "runtime_behaviors": {
                "type": "array",
                "items": {"type": "string", "minLength": 20, "maxLength": 400},
            },
            "safety_constraints": {
                "type": "array",
                "items": {"type": "string", "minLength": 15, "maxLength": 300},
            },
            "hardness_rationale": {
                "type": "array",
                "items": {"type": "string", "minLength": 20, "maxLength": 400},
            },
            "likely_failure_modes": {
                "type": "array",
                "items": {"type": "string", "minLength": 20, "maxLength": 400},
            },
        },
    },
}


WRITER_SCHEMA: dict[str, Any] = {
    "name": "kubernetes_public_task",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["task_text", "covered_requirement_ids"],
        "properties": {
            "task_text": {"type": "string", "minLength": 600, "maxLength": 7000},
            "covered_requirement_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "pattern": "^R[0-9]{2}$"},
            },
        },
    },
}


CRITIC_SCHEMA: dict[str, Any] = {
    "name": "kubernetes_hardness_review",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "decision",
            "hardness_score",
            "category_assignments",
            "checks",
            "easy_shortcuts",
            "ambiguities",
            "contradictions",
            "leakage_findings",
            "missing_requirement_ids",
            "revision_instructions",
        ],
        "properties": {
            "decision": {"type": "string", "enum": ["accept", "revise", "reject"]},
            "hardness_score": {"type": "integer", "minimum": 1, "maximum": 10},
            "category_assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["category", "evidence_requirement_ids"],
                    "properties": {
                        "category": {"type": "string", "enum": list(CATEGORIES)},
                        "evidence_requirement_ids": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": {"type": "string", "pattern": "^R[0-9]{2}$"},
                        },
                    },
                },
            },
            "checks": {
                "type": "object",
                "additionalProperties": False,
                "required": list(CHECK_NAMES),
                "properties": {name: {"type": "boolean"} for name in CHECK_NAMES},
            },
            "easy_shortcuts": {"type": "array", "items": {"type": "string"}},
            "ambiguities": {"type": "array", "items": {"type": "string"}},
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "leakage_findings": {"type": "array", "items": {"type": "string"}},
            "missing_requirement_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "pattern": "^R[0-9]{2}$"},
            },
            "revision_instructions": {"type": "array", "items": {"type": "string"}},
        },
    },
}


ROLE_SCHEMAS = {
    "spec_generator": SPEC_SCHEMA,
    "plaintext_writer": WRITER_SCHEMA,
    "critic": CRITIC_SCHEMA,
}


def _require_mapping(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return {}
    return value


def validate_spec(spec: Any, config: PilotConfig, task_id: str) -> list[str]:
    errors: list[str] = []
    item = _require_mapping(spec, "spec", errors)
    required = SPEC_SCHEMA["schema"]["required"]
    for field in required:
        if field not in item:
            errors.append(f"spec is missing {field}")
    if errors:
        return errors
    if item["task_id"] != task_id:
        errors.append(f"spec task_id must be {task_id}")
    if item["target_kubernetes_version"] != config.target_kubernetes_version:
        errors.append("spec targets the wrong Kubernetes version")

    gate = config.hardness_gate
    categories = item.get("categories", [])
    if not isinstance(categories, list) or len(set(categories)) < gate["minimum_categories"]:
        errors.append("spec has too few distinct hardness categories")
    if set(categories) - set(CATEGORIES):
        errors.append("spec contains an unknown category")

    requirements = item.get("public_requirements", [])
    if not isinstance(requirements, list) or len(requirements) < gate["minimum_public_requirements"]:
        errors.append("spec has too few public requirements")
        return errors

    ids = [req.get("id") for req in requirements if isinstance(req, dict)]
    if len(ids) != len(requirements) or len(set(ids)) != len(ids):
        errors.append("requirement IDs must be present and unique")
        return errors
    if any(not isinstance(req_id, str) or not re.fullmatch(r"R[0-9]{2}", req_id) for req_id in ids):
        errors.append("requirement IDs must use R00 format")

    edges = 0
    graph: dict[str, list[str]] = {}
    for req in requirements:
        deps = req.get("depends_on", [])
        if not isinstance(deps, list):
            errors.append(f"{req.get('id')} depends_on must be a list")
            deps = []
        graph[str(req.get("id"))] = [str(dep) for dep in deps]
        edges += len(deps)
        for dep in deps:
            if dep not in ids or dep == req.get("id"):
                errors.append(f"{req.get('id')} has an invalid dependency {dep}")
    if edges < gate["minimum_dependency_edges"]:
        errors.append("requirement graph has too few dependency edges")
    if not any(len(deps) >= 2 for deps in graph.values()):
        errors.append("no requirement integrates at least two dependencies")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(dep) for dep in graph.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    if any(visit(req_id) for req_id in ids):
        errors.append("requirement dependency graph contains a cycle")

    kinds = item.get("resource_kinds", [])
    if not isinstance(kinds, list) or len(set(kinds)) < gate["minimum_resource_kinds"]:
        errors.append("spec requires too few distinct resource kinds")
    if set(kinds) - ALLOWED_RESOURCE_KINDS:
        errors.append("spec uses unsupported or cluster-scoped resource kinds")
    if len(item.get("runtime_behaviors", [])) < gate["minimum_runtime_behaviors"]:
        errors.append("spec has too few runtime behaviours")
    if len(item.get("safety_constraints", [])) < gate["minimum_safety_constraints"]:
        errors.append("spec has too few safety constraints")
    if len(item.get("hardness_rationale", [])) < 4:
        errors.append("spec has insufficient private hardness rationale")
    if len(item.get("likely_failure_modes", [])) < 4:
        errors.append("spec has insufficient likely failure modes")
    return errors


def validate_writer(writer: Any, spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    item = _require_mapping(writer, "writer output", errors)
    text = item.get("task_text")
    coverage = item.get("covered_requirement_ids")
    if not isinstance(text, str) or not 600 <= len(text) <= 7000:
        errors.append("public task text must contain 600-7000 characters")
    expected = {req["id"] for req in spec.get("public_requirements", [])}
    if not isinstance(coverage, list) or set(coverage) != expected or len(coverage) != len(expected):
        errors.append("writer coverage must contain every requirement ID exactly once")
    if isinstance(text, str):
        lowered = text.lower()
        forbidden = (
            "hardness score",
            "likely failure mode",
            "hidden test",
            "reference manifest",
            "critic feedback",
            "private requirement",
            "authoring note",
        )
        if any(term in lowered for term in forbidden):
            errors.append("public task text leaks private authoring language")
        if re.search(r"\bR[0-9]{2}\b", text):
            errors.append("public task text leaks private requirement IDs")
    return errors


def validate_critic(
    critic: Any, spec: dict[str, Any], config: PilotConfig
) -> list[str]:
    errors: list[str] = []
    item = _require_mapping(critic, "critic output", errors)
    required_ids = {req["id"] for req in spec.get("public_requirements", [])}
    score = item.get("hardness_score")
    if not isinstance(score, int) or not 1 <= score <= 10:
        errors.append("critic hardness_score must be an integer from 1 to 10")
    assignments = item.get("category_assignments", [])
    if not isinstance(assignments, list):
        errors.append("critic category_assignments must be a list")
        assignments = []
    assigned_categories: list[str] = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            errors.append("critic category assignment must be an object")
            continue
        category = assignment.get("category")
        evidence = assignment.get("evidence_requirement_ids", [])
        assigned_categories.append(str(category))
        if category not in CATEGORIES:
            errors.append(f"critic assigned unknown category {category}")
        if not evidence or not set(evidence) <= required_ids:
            errors.append(f"critic category {category} has invalid evidence")
    if len(set(assigned_categories)) != len(assigned_categories):
        errors.append("critic categories must be unique")
    missing_ids = item.get("missing_requirement_ids", [])
    if not isinstance(missing_ids, list) or not set(missing_ids) <= required_ids:
        errors.append("critic missing_requirement_ids contains invalid IDs")
    checks = item.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECK_NAMES):
        errors.append("critic checks object is incomplete")
    elif any(not isinstance(value, bool) for value in checks.values()):
        errors.append("every critic check must be boolean")
    if item.get("decision") not in {"accept", "revise", "reject"}:
        errors.append("critic decision is invalid")
    return errors


def acceptance_errors(
    spec: dict[str, Any], writer: dict[str, Any], critic: dict[str, Any], config: PilotConfig
) -> list[str]:
    errors = validate_spec(spec, config, str(spec.get("task_id", "")))
    errors.extend(validate_writer(writer, spec))
    errors.extend(validate_critic(critic, spec, config))
    gate = config.hardness_gate
    if critic.get("decision") != "accept":
        errors.append("critic did not accept the task")
    if critic.get("hardness_score", 0) < gate["minimum_score"]:
        errors.append("critic hardness score is below the threshold")
    assignments = critic.get("category_assignments", [])
    category_count = len({x.get("category") for x in assignments if isinstance(x, dict)})
    if category_count < gate["minimum_categories"]:
        errors.append("critic confirmed too few categories")
    checks = critic.get("checks", {})
    for name in CHECK_NAMES:
        if checks.get(name) is not True:
            errors.append(f"critic check failed: {name}")
    for field in (
        "easy_shortcuts",
        "ambiguities",
        "contradictions",
        "leakage_findings",
        "missing_requirement_ids",
    ):
        if critic.get(field):
            errors.append(f"critic reported {field}")
    return sorted(set(errors))
