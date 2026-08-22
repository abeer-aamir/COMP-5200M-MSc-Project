from __future__ import annotations

import json
import re
from typing import Any

from .config import DIFFICULTY_LEVELS, PilotConfig


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

DEPENDENCY_TOPOLOGY_CLASSES = (
    "single_chain",
    "fan_in",
    "fan_out",
    "multi_stage_chain",
    "mixed_dag",
)

RUNTIME_BEHAVIOR_TYPES = (
    "readiness_transition",
    "probe_recovery",
    "controller_replacement",
    "job_completion",
    "cron_scheduling",
    "rollout_propagation",
    "storage_handoff",
    "service_endpoint_transition",
    "policy_connectivity_transition",
)

SAFETY_CONSTRAINT_CATEGORIES = (
    "pod_hardening",
    "rbac_least_privilege",
    "network_policy",
    "resource_enforcement",
)

CRITIC_DEFECT_TYPES = (
    "contract_violation",
    "ambiguity",
    "infeasible",
    "non_deterministic_check",
    "unjustified_hidden_check",
    "leakage",
    "insufficient_diversity",
    "environment_violation",
    "traceability_gap",
    "core_redesign_required",
)

CRITIC_DEFECT_LOCATIONS = (
    "private_specification",
    "verification_blueprint",
    "public_task_text",
)


SPEC_SCHEMA: dict[str, Any] = {
    "name": "kubernetes_private_spec",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "task_id",
            "difficulty_level",
            "scenario",
            "requirements",
            "dependency_edges",
            "resource_kinds",
            "runtime_behaviours",
            "safety_constraints",
            "verification_blueprints",
            "diversity_fingerprint",
            "hardness_rationale",
            "likely_failure_modes",
        ],
        "properties": {
            "task_id": {
                "type": "string",
                "pattern": "^(pilot|easy|medium|hard)-[0-9]{3}$",
            },
            "difficulty_level": {"type": "string", "enum": list(DIFFICULTY_LEVELS)},
            "scenario": {"type": "string"},
            "requirements": {
                "type": "array",
                "minItems": 6,
                "maxItems": 14,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "text"],
                    "properties": {
                        "id": {"type": "string", "pattern": "^R[0-9]{2}$"},
                        "text": {"type": "string"},
                    },
                },
            },
            "dependency_edges": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["from", "to", "reason"],
                    "properties": {
                        "from": {"type": "string", "pattern": "^R[0-9]{2}$"},
                        "to": {"type": "string", "pattern": "^R[0-9]{2}$"},
                        "reason": {"type": "string"},
                    },
                },
            },
            "resource_kinds": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(ALLOWED_RESOURCE_KINDS)},
            },
            "runtime_behaviours": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["description", "related_requirement_ids"],
                    "properties": {
                        "description": {"type": "string"},
                        "related_requirement_ids": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": {"type": "string", "pattern": "^R[0-9]{2}$"},
                        },
                    },
                },
            },
            "safety_constraints": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["category", "description", "related_requirement_ids"],
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": list(SAFETY_CONSTRAINT_CATEGORIES),
                        },
                        "description": {"type": "string"},
                        "related_requirement_ids": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": {"type": "string", "pattern": "^R[0-9]{2}$"},
                        },
                    },
                },
            },
            "verification_blueprints": {
                "type": "array",
                "minItems": 6,
                "maxItems": 14,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "requirement_id",
                        "observable_state",
                        "precondition",
                        "assertion",
                        "timeout_seconds",
                        "candidate_failure_signals",
                        "infrastructure_failure_signals",
                    ],
                    "properties": {
                        "requirement_id": {
                            "type": "string",
                            "pattern": "^R[0-9]{2}$",
                        },
                        "observable_state": {"type": "string"},
                        "precondition": {"type": "string"},
                        "assertion": {"type": "string"},
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 300,
                        },
                        "candidate_failure_signals": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "infrastructure_failure_signals": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "diversity_fingerprint": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "resource_kind_set",
                    "dependency_topology_class",
                    "runtime_behaviour_types",
                    "safety_constraint_categories",
                ],
                "properties": {
                    "resource_kind_set": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": sorted(ALLOWED_RESOURCE_KINDS),
                        },
                    },
                    "dependency_topology_class": {
                        "type": "string",
                        "enum": list(DEPENDENCY_TOPOLOGY_CLASSES),
                    },
                    "runtime_behaviour_types": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": list(RUNTIME_BEHAVIOR_TYPES),
                        },
                    },
                    "safety_constraint_categories": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": list(SAFETY_CONSTRAINT_CATEGORIES),
                        },
                    },
                },
            },
            "hardness_rationale": {"type": "string"},
            "likely_failure_modes": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
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
            "task_text": {"type": "string"},
            "covered_requirement_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "pattern": "^R[0-9]{2}$"},
            },
        },
    },
}


CRITIC_SCHEMA: dict[str, Any] = {
    "name": "kubernetes_authoring_review",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "verdict",
            "hardness_assessment",
            "hardness_justification",
            "findings",
        ],
        "properties": {
            "verdict": {"type": "string", "enum": ["accept", "revise", "reject"]},
            "hardness_assessment": {
                "type": "string",
                "enum": [
                    "matches_contract",
                    "easier_than_contract",
                    "harder_than_contract",
                ],
            },
            "hardness_justification": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "requirement_id",
                        "defect_type",
                        "defect_location",
                        "evidence",
                        "minimum_revision_instruction",
                    ],
                    "properties": {
                        "requirement_id": {
                            "type": "string",
                            "pattern": "^(R[0-9]{2})?$",
                        },
                        "defect_type": {
                            "type": "string",
                            "enum": list(CRITIC_DEFECT_TYPES),
                        },
                        "defect_location": {
                            "type": "string",
                            "enum": list(CRITIC_DEFECT_LOCATIONS),
                        },
                        "evidence": {"type": "string"},
                        "minimum_revision_instruction": {"type": "string"},
                    },
                },
            },
        },
    },
}


ROLE_SCHEMAS = {
    "spec_generator": SPEC_SCHEMA,
    "plaintext_writer": WRITER_SCHEMA,
    "critic": CRITIC_SCHEMA,
}


# Anthropic's structured-output grammar does not accept every validation keyword
# that is useful to us locally. OpenRouter can route the same request through
# several Anthropic providers, so send the common structural subset and retain the
# complete schema for deterministic validation after the response is received.
_PROVIDER_DESCRIPTION_CONSTRAINTS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
}


def _constraint_description(keyword: str, value: Any) -> str:
    descriptions = {
        "minimum": f"Value must be at least {value}.",
        "maximum": f"Value must be at most {value}.",
        "exclusiveMinimum": f"Value must be greater than {value}.",
        "exclusiveMaximum": f"Value must be less than {value}.",
        "multipleOf": f"Value must be a multiple of {value}.",
        "pattern": f"Value must match this regular expression: {value}.",
        "format": f"Value must use the {value} format.",
        "minItems": f"Array must contain at least {value} items.",
        "maxItems": f"Array must contain at most {value} items.",
        "uniqueItems": "Array items must be unique.",
        "minProperties": f"Object must contain at least {value} properties.",
        "maxProperties": f"Object must contain at most {value} properties.",
    }
    return descriptions[keyword]


def _provider_schema_node(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    transformed: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in node.items():
        if key in _PROVIDER_DESCRIPTION_CONSTRAINTS:
            if value is not False and value is not None:
                notes.append(_constraint_description(key, value))
            continue
        if key == "properties":
            transformed[key] = {
                name: _provider_schema_node(child) for name, child in value.items()
            }
        elif key in {"items", "additionalProperties"} and isinstance(value, dict):
            transformed[key] = _provider_schema_node(value)
        elif key in {"anyOf", "allOf", "oneOf", "prefixItems"}:
            transformed[key] = [_provider_schema_node(child) for child in value]
        else:
            transformed[key] = value
    if notes:
        existing = transformed.get("description", "").strip()
        transformed["description"] = " ".join(filter(None, [existing, *notes]))
    return transformed


def provider_compatible_schema(schema_wrapper: dict[str, Any]) -> dict[str, Any]:
    """Return an Anthropic/provider-compatible copy of an OpenRouter schema.

    The original schema is never changed and remains the source of truth for local
    response validation.
    """
    transformed = dict(schema_wrapper)
    transformed["schema"] = _provider_schema_node(schema_wrapper["schema"])
    return transformed


def _json_schema_errors(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    type_checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected_type in type_checks and not type_checks[expected_type](value):
        return [f"{path} must be of type {expected_type}"]

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path} must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} is not an allowed value")

    if isinstance(value, dict):
        required = schema.get("required", [])
        errors.extend(f"{path} is missing {key}" for key in required if key not in value)
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errors.extend(
                f"{path} contains unexpected property {key}"
                for key in value
                if key not in properties
            )
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(
                    _json_schema_errors(value[key], child_schema, f"{path}.{key}")
                )

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path} must contain at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path} must contain at most {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            fingerprints = [
                json.dumps(item, ensure_ascii=False, sort_keys=True) for item in value
            ]
            if len(fingerprints) != len(set(fingerprints)):
                errors.append(f"{path} items must be unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(
                    _json_schema_errors(item, item_schema, f"{path}[{index}]")
                )

    if (
        isinstance(value, str)
        and "pattern" in schema
        and re.fullmatch(schema["pattern"], value) is None
    ):
        errors.append(f"{path} does not match the required pattern")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be at most {schema['maximum']}")
    return errors


def validate_schema_instance(
    value: Any, schema_wrapper: dict[str, Any], label: str
) -> list[str]:
    return _json_schema_errors(value, schema_wrapper["schema"], label)


def _require_mapping(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _fingerprint_duplicate(
    fingerprint: dict[str, Any], previous: list[dict[str, Any]]
) -> str | None:
    kinds = set(fingerprint.get("resource_kind_set", []))
    topology = fingerprint.get("dependency_topology_class")
    for entry in previous:
        prior = entry.get("fingerprint", entry)
        if (
            kinds == set(prior.get("resource_kind_set", []))
            and topology == prior.get("dependency_topology_class")
        ):
            return str(entry.get("task_id", "previous task"))
    return None


def validate_spec(
    spec: Any,
    config: PilotConfig,
    task_id: str,
    difficulty_level: str | None = None,
    diversity_fingerprints: list[dict[str, Any]] | None = None,
) -> list[str]:
    errors = validate_schema_instance(spec, SPEC_SCHEMA, "spec")
    if errors:
        return errors
    item = _require_mapping(spec, "spec", errors)
    required = SPEC_SCHEMA["schema"]["required"]
    for field in required:
        if field not in item:
            errors.append(f"spec is missing {field}")
    if errors:
        return errors
    if item["task_id"] != task_id:
        errors.append(f"spec task_id must be {task_id}")

    requested_level = difficulty_level or str(item.get("difficulty_level", ""))
    if requested_level not in config.difficulty_contracts:
        errors.append("spec has an unknown difficulty level")
        return errors
    if item.get("difficulty_level") != requested_level:
        errors.append(f"spec difficulty_level must be {requested_level}")
    contract = config.difficulty_contracts[requested_level]
    if not str(item.get("scenario", "")).strip():
        errors.append("spec scenario must be non-empty")

    requirements = item.get("requirements", [])
    if not isinstance(requirements, list) or not (
        contract.minimum_public_requirements
        <= len(requirements)
        <= contract.maximum_public_requirements
    ):
        errors.append("spec public requirement count is outside its difficulty contract")
        return errors

    ids = [req.get("id") for req in requirements if isinstance(req, dict)]
    if len(ids) != len(requirements) or len(set(ids)) != len(ids):
        errors.append("requirement IDs must be present and unique")
        return errors
    if any(not isinstance(req_id, str) or not re.fullmatch(r"R[0-9]{2}", req_id) for req_id in ids):
        errors.append("requirement IDs must use R00 format")
    expected_ids = [f"R{index:02d}" for index in range(1, len(requirements) + 1)]
    if ids != expected_ids:
        errors.append("requirement IDs must be sequential and in source order")
    if any(not str(req.get("text", "")).strip() for req in requirements):
        errors.append("every requirement text must be non-empty")
    namespace_requirements = [
        req
        for req in requirements
        if isinstance(req, dict)
        and re.search(r"\bNamespace\b", str(req.get("text", "")))
    ]
    if len(namespace_requirements) != 1:
        errors.append("spec must contain exactly one explicit Namespace requirement")

    dependency_edges = item.get("dependency_edges", [])
    edges = len(dependency_edges) if isinstance(dependency_edges, list) else 0
    graph: dict[str, list[str]] = {str(req_id): [] for req_id in ids}
    edge_pairs: set[tuple[str, str]] = set()
    for edge in dependency_edges if isinstance(dependency_edges, list) else []:
        if not isinstance(edge, dict):
            continue
        prerequisite = edge.get("from")
        dependent = edge.get("to")
        if prerequisite not in ids or dependent not in ids or prerequisite == dependent:
            errors.append(
                f"dependency edge {prerequisite!r}->{dependent!r} is invalid"
            )
            continue
        if not str(edge.get("reason", "")).strip():
            errors.append(f"dependency edge {prerequisite}->{dependent} needs a reason")
        pair = (str(prerequisite), str(dependent))
        if pair in edge_pairs:
            errors.append(f"dependency edge {prerequisite}->{dependent} is duplicated")
            continue
        edge_pairs.add(pair)
        graph[str(prerequisite)].append(str(dependent))
    if edges < contract.minimum_dependency_edges or (
        contract.maximum_dependency_edges is not None
        and edges > contract.maximum_dependency_edges
    ):
        errors.append("requirement dependency count is outside its difficulty contract")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(dependent) for dependent in graph.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    if any(visit(req_id) for req_id in ids):
        errors.append("requirement dependency graph contains a cycle")

    kinds = item.get("resource_kinds", [])
    if not isinstance(kinds, list) or not (
        contract.minimum_resource_kinds
        <= len(set(kinds))
        <= contract.maximum_resource_kinds
    ):
        errors.append("spec resource-kind count is outside its difficulty contract")
    if set(kinds) - ALLOWED_RESOURCE_KINDS:
        errors.append("spec uses unsupported or cluster-scoped resource kinds")
    if len(item.get("runtime_behaviours", [])) != contract.runtime_behaviours:
        errors.append("spec runtime-behaviour count violates its difficulty contract")
    if len(item.get("safety_constraints", [])) != contract.safety_constraints:
        errors.append("spec safety-constraint count violates its difficulty contract")

    for field in ("runtime_behaviours", "safety_constraints"):
        for entry in item.get(field, []):
            if not isinstance(entry, dict):
                continue
            related = entry.get("related_requirement_ids", [])
            if not related or not set(related) <= set(ids):
                errors.append(f"{field} contains invalid related requirement IDs")
            if not str(entry.get("description", "")).strip():
                errors.append(f"{field} contains an empty description")

    blueprints = item.get("verification_blueprints", [])
    blueprint_ids = [
        blueprint.get("requirement_id")
        for blueprint in blueprints
        if isinstance(blueprint, dict)
    ]
    if len(blueprint_ids) != len(ids) or sorted(blueprint_ids) != sorted(ids):
        errors.append("verification blueprints must cover every requirement exactly once")
    for blueprint in blueprints:
        if not isinstance(blueprint, dict):
            continue
        for field in ("observable_state", "assertion"):
            if not str(blueprint.get(field, "")).strip():
                errors.append(f"verification blueprint {field} must be non-empty")
        for field in ("candidate_failure_signals", "infrastructure_failure_signals"):
            if any(not str(signal).strip() for signal in blueprint.get(field, [])):
                errors.append(f"verification blueprint {field} cannot contain empty signals")

    fingerprint = item.get("diversity_fingerprint", {})
    if isinstance(fingerprint, dict):
        fingerprint_kinds = fingerprint.get("resource_kind_set", [])
        if set(fingerprint_kinds) != set(kinds):
            errors.append("diversity fingerprint resource kinds do not match the spec")
        if fingerprint_kinds != sorted(set(fingerprint_kinds)):
            errors.append("diversity fingerprint resource kinds must be canonical and sorted")
        actual_safety_categories = {
            constraint.get("category")
            for constraint in item.get("safety_constraints", [])
            if isinstance(constraint, dict)
        }
        if set(fingerprint.get("safety_constraint_categories", [])) != actual_safety_categories:
            errors.append("diversity fingerprint safety categories do not match the spec")
        duplicate = _fingerprint_duplicate(
            fingerprint, diversity_fingerprints or []
        )
        if duplicate is not None:
            errors.append(f"diversity fingerprint duplicates {duplicate}")
    if not str(item.get("hardness_rationale", "")).strip():
        errors.append("spec has no private hardness rationale")
    if not item.get("likely_failure_modes") or any(
        not str(mode).strip() for mode in item.get("likely_failure_modes", [])
    ):
        errors.append("spec has no likely failure modes")
    return errors


def validate_writer(writer: Any, spec: dict[str, Any]) -> list[str]:
    errors = validate_schema_instance(writer, WRITER_SCHEMA, "writer output")
    if errors:
        return errors
    item = _require_mapping(writer, "writer output", errors)
    text = item.get("task_text")
    coverage = item.get("covered_requirement_ids")
    expected = [req["id"] for req in spec.get("requirements", [])]
    if coverage != expected:
        errors.append("writer coverage must contain every requirement ID exactly once and in order")
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
        bullets = [line for line in text.splitlines() if line.startswith("- ")]
        if len(bullets) != len(expected):
            errors.append("public task text must contain exactly one bullet per requirement")
    return errors


def validate_critic(
    critic: Any, spec: dict[str, Any], config: PilotConfig
) -> list[str]:
    errors = validate_schema_instance(critic, CRITIC_SCHEMA, "critic output")
    if errors:
        return errors
    item = _require_mapping(critic, "critic output", errors)
    required_ids = {req["id"] for req in spec.get("requirements", [])}
    if not str(item.get("hardness_justification", "")).strip():
        errors.append("critic hardness justification must be non-empty")
    findings = item.get("findings", [])
    if not isinstance(findings, list):
        errors.append("critic findings must be a list")
        findings = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        requirement_id = finding.get("requirement_id")
        if requirement_id and requirement_id not in required_ids:
            errors.append("critic finding references an invalid requirement ID")
        if not str(finding.get("evidence", "")).strip():
            errors.append("critic finding evidence must be non-empty")
        defect_type = finding.get("defect_type")
        instruction = finding.get("minimum_revision_instruction")
        if not str(instruction).strip() and not (
            defect_type == "core_redesign_required" and item.get("verdict") == "reject"
        ):
            errors.append("critic finding is missing a minimum revision instruction")
        if defect_type == "core_redesign_required" and item.get("verdict") != "reject":
            errors.append("core_redesign_required is valid only with a reject verdict")

    verdict = item.get("verdict")
    assessment = item.get("hardness_assessment")
    if verdict == "accept":
        if findings:
            errors.append("accepted critic output cannot contain findings")
        if assessment != "matches_contract":
            errors.append("accepted critic output must match its difficulty contract")
    elif verdict in {"revise", "reject"}:
        if not findings:
            errors.append("non-accepted critic output must contain findings")
        if verdict == "reject" and not any(
            finding.get("defect_type") == "core_redesign_required"
            for finding in findings
            if isinstance(finding, dict)
        ):
            errors.append("reject verdict must identify a core redesign requirement")
    if assessment != "matches_contract" and not any(
        finding.get("defect_type") == "contract_violation"
        for finding in findings
        if isinstance(finding, dict)
    ):
        errors.append("hardness mismatch must have a contract_violation finding")
    return errors


def acceptance_errors(
    spec: dict[str, Any],
    writer: dict[str, Any],
    critic: dict[str, Any],
    config: PilotConfig,
    difficulty_level: str | None = None,
    diversity_fingerprints: list[dict[str, Any]] | None = None,
) -> list[str]:
    level = difficulty_level or str(spec.get("difficulty_level", ""))
    errors = validate_spec(
        spec,
        config,
        str(spec.get("task_id", "")),
        level,
        diversity_fingerprints,
    )
    errors.extend(validate_writer(writer, spec))
    errors.extend(validate_critic(critic, spec, config))
    if critic.get("verdict") != "accept":
        errors.append("critic did not accept the task")
    if critic.get("hardness_assessment") != "matches_contract":
        errors.append("critic hardness assessment does not match the difficulty contract")
    if critic.get("findings"):
        errors.append("critic reported material findings")
    return sorted(set(errors))
