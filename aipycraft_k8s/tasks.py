from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


TASK_INDEX = PROJECT_ROOT / "benchmark" / "tasks" / "index.json"

# Keep the task-to-namespace contract in code as a fail-fast guard.  The JSON
# index remains the user-facing catalogue, but a typo or an unregistered suite
# must be rejected before a paid generation call is made.
CANONICAL_TASK_NAMESPACES = {
    "pilot-001": "order-system",
    "pilot-002": "pipeline-ns",
    "pilot-003": "status-page",
    "pilot-004": "service-chain",
    "easy-001": "easy-status-ns",
    "easy-002": "easy-calc-ns",
    "easy-003": "easy-rbac-ns",
    "easy-004": "easy-restart-ns",
    "easy-005": "easy-diagnostics-ns",
    "medium-001": "medium-portal-ns",
    "medium-002": "medium-batch-ns",
    "medium-003": "medium-stateful-ns",
    "medium-004": "medium-check-ns",
    "medium-005": "medium-ledger-ns",
    "hard-001": "hard-report-ns",
    "hard-002": "hard-queue-ns",
    "hard-003": "hard-chain-ns",
    "hard-004": "hard-audit-ns",
    "hard-005": "hard-monitor-ns",
    "very-hard-001": "very-hard-release-ns",
    "very-hard-002": "very-hard-ledger-ns",
    "very-hard-003": "very-hard-chain-ns",
    "very-hard-004": "very-hard-rbac-ns",
    "very-hard-005": "very-hard-maintenance-ns",
}


class TaskError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    namespace: str
    description_path: Path
    description: str
    post_execution_suite: str | None
    kubernetes_version: str
    execution_image: str


def _resolve_description(index_path: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise TaskError("Task description path must be a string")
    relative = Path(value)
    if relative.is_absolute():
        raise TaskError("Task description path must be relative")
    canonical_index = index_path.resolve() == TASK_INDEX.resolve()
    base = PROJECT_ROOT if canonical_index else index_path.resolve().parent
    path = (base / relative).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise TaskError("Task description escapes its task-set root") from exc
    if not path.is_file():
        raise TaskError(f"Task description was not found: {path}")
    return path


def load_tasks(index_path: Path = TASK_INDEX) -> dict[str, BenchmarkTask]:
    index_path = Path(index_path).resolve()
    canonical_index = index_path == TASK_INDEX.resolve()
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskError(f"Could not load task index: {exc}") from exc
    expected_top_level = {
        "schema_version",
        "kubernetes_version",
        "execution_image",
        "tasks",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != expected_top_level
        or raw.get("schema_version") != 1
        or not isinstance(raw.get("tasks"), list)
        or not isinstance(raw.get("kubernetes_version"), str)
        or not isinstance(raw.get("execution_image"), str)
    ):
        raise TaskError("Unsupported or malformed task index")
    tasks: dict[str, BenchmarkTask] = {}
    for item in raw["tasks"]:
        required = {"task_id", "namespace", "description", "post_execution_suite"}
        if not isinstance(item, dict) or set(item) != required:
            raise TaskError("Task index entry has unexpected fields")
        description_path = _resolve_description(index_path, item["description"])
        if not isinstance(item["task_id"], str) or not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", item["task_id"]
        ):
            raise TaskError(f"Invalid task id: {item['task_id']!r}")
        task_id = item["task_id"]
        if task_id in tasks:
            raise TaskError(f"Duplicate task id: {task_id}")
        namespace = item["namespace"]
        if not isinstance(namespace, str) or not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", namespace
        ):
            raise TaskError(f"Invalid namespace for {task_id}: {namespace!r}")
        post_execution_suite = item["post_execution_suite"]
        if canonical_index:
            expected_namespace = CANONICAL_TASK_NAMESPACES.get(task_id)
            if expected_namespace is None:
                raise TaskError(
                    f"Task {task_id} has no registered private evaluator contract"
                )
            if namespace != expected_namespace:
                raise TaskError(
                    f"Task {task_id} must use namespace {expected_namespace!r}, "
                    f"not {namespace!r}"
                )
            if post_execution_suite != task_id:
                raise TaskError(
                    f"Task {task_id} must use its matching private suite"
                )
            expected_description = (
                PROJECT_ROOT / "benchmark" / "tasks" / task_id / "description.txt"
            ).resolve()
            if description_path != expected_description:
                raise TaskError(
                    f"Task {task_id} description must be {expected_description}"
                )
        elif post_execution_suite is not None:
            if not isinstance(post_execution_suite, str):
                raise TaskError(
                    f"Invalid private suite for {task_id}: {post_execution_suite!r}"
                )
            expected_namespace = CANONICAL_TASK_NAMESPACES.get(task_id)
            if post_execution_suite != task_id or namespace != expected_namespace:
                raise TaskError(
                    f"External task {task_id} may use a private suite only when its "
                    "registered task ID and namespace match"
                )
        description = description_path.read_text(encoding="utf-8")
        if not description.strip():
            raise TaskError(f"Task {task_id} description is empty")
        tasks[task_id] = BenchmarkTask(
            task_id=task_id,
            namespace=namespace,
            description_path=description_path,
            description=description,
            post_execution_suite=post_execution_suite,
            kubernetes_version=str(raw["kubernetes_version"]),
            execution_image=str(raw["execution_image"]),
        )
    return tasks


def load_task(task_id: str, index_path: Path = TASK_INDEX) -> BenchmarkTask:
    tasks = load_tasks(index_path)
    try:
        return tasks[task_id]
    except KeyError as exc:
        raise TaskError(f"Unknown task {task_id!r}; choose from {sorted(tasks)}") from exc
