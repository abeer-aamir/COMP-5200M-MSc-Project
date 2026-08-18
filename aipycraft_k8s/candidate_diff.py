from __future__ import annotations

from typing import Any

import yaml


def _identity(document: dict[str, Any]) -> str:
    metadata = document.get("metadata", {}) or {}
    namespace = metadata.get("namespace", "<cluster-or-default>")
    return "/".join(
        str(value)
        for value in (
            document.get("apiVersion", "<missing>"),
            document.get("kind", "<missing>"),
            namespace,
            metadata.get("name", "<unnamed>"),
        )
    )


def _resources(text: str) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    for document in yaml.safe_load_all(text):
        if isinstance(document, dict):
            if document.get("kind") == "List" and isinstance(
                document.get("items"), list
            ):
                for item in document["items"]:
                    if isinstance(item, dict):
                        resources[_identity(item)] = item
            else:
                resources[_identity(document)] = document
    return resources


def _changed_paths(before: Any, after: Any, prefix: str = "") -> list[str]:
    if type(before) is not type(after):
        return [prefix or "$type"]
    if isinstance(before, dict):
        paths: list[str] = []
        for key in sorted(set(before) | set(after)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(child)
            else:
                paths.extend(_changed_paths(before[key], after[key], child))
        return paths
    if isinstance(before, list):
        if before == after:
            return []
        return [prefix]
    return [] if before == after else [prefix]


def candidate_semantic_diff(previous: str | None, current: str) -> dict[str, Any]:
    current_resources = _resources(current)
    if previous is None:
        return {
            "schema_version": 1,
            "basis": "initial_candidate",
            "resource_count": len(current_resources),
            "added_resources": sorted(current_resources),
            "removed_resources": [],
            "modified_resources": [],
            "changed_field_path_count": 0,
            "changed_field_paths": [],
        }
    previous_resources = _resources(previous)
    previous_ids = set(previous_resources)
    current_ids = set(current_resources)
    common = sorted(previous_ids & current_ids)
    modified = [
        identity
        for identity in common
        if previous_resources[identity] != current_resources[identity]
    ]
    paths = [
        f"{identity}:{path}"
        for identity in modified
        for path in _changed_paths(
            previous_resources[identity], current_resources[identity]
        )
    ]
    return {
        "schema_version": 1,
        "basis": "previous_yaml_valid_candidate",
        "previous_resource_count": len(previous_resources),
        "resource_count": len(current_resources),
        "added_resources": sorted(current_ids - previous_ids),
        "removed_resources": sorted(previous_ids - current_ids),
        "modified_resources": modified,
        "changed_field_path_count": len(paths),
        "changed_field_paths": paths[:500],
        "changed_field_paths_truncated": max(len(paths) - 500, 0),
    }
