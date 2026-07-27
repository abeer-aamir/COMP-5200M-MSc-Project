from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


TASK_INDEX = PROJECT_ROOT / "benchmark" / "tasks" / "index.json"


class TaskError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    namespace: str
    description_path: Path
    description: str
    post_execution_suite: str
    kubernetes_version: str
    execution_image: str


def _resolve_project_file(value: Any) -> Path:
    if not isinstance(value, str):
        raise TaskError("Task description path must be a string")
    path = (PROJECT_ROOT / value).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise TaskError("Task description escapes the project root") from exc
    if not path.is_file():
        raise TaskError(f"Task description was not found: {path}")
    return path


def load_tasks(index_path: Path = TASK_INDEX) -> dict[str, BenchmarkTask]:
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskError(f"Could not load task index: {exc}") from exc
    if raw.get("schema_version") != 1 or not isinstance(raw.get("tasks"), list):
        raise TaskError("Unsupported or malformed task index")
    tasks: dict[str, BenchmarkTask] = {}
    for item in raw["tasks"]:
        required = {"task_id", "namespace", "description", "post_execution_suite"}
        if not isinstance(item, dict) or set(item) != required:
            raise TaskError("Task index entry has unexpected fields")
        description_path = _resolve_project_file(item["description"])
        task_id = str(item["task_id"])
        if task_id in tasks:
            raise TaskError(f"Duplicate task id: {task_id}")
        tasks[task_id] = BenchmarkTask(
            task_id=task_id,
            namespace=str(item["namespace"]),
            description_path=description_path,
            description=description_path.read_text(encoding="utf-8"),
            post_execution_suite=str(item["post_execution_suite"]),
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
