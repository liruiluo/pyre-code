#!/usr/bin/env python3
"""Export problem metadata from torch_judge task definitions for the frontend."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "web" / "src" / "lib" / "problems.json"

# Make torch_judge importable when the script is run directly.
sys.path.insert(0, str(ROOT))

from torch_judge.tasks import TASKS, list_tasks

REQUIRED_TASK_KEYS = (
    "title",
    "title_zh",
    "difficulty",
    "function_name",
    "hint",
    "hint_zh",
    "description_en",
    "description_zh",
    "tests",
)

REQUIRED_CHOICE_KEYS = (
    "title",
    "title_zh",
    "difficulty",
    "question_en",
    "question_zh",
    "options",
    "answer",
    "explanation_en",
    "explanation_zh",
)


def _validate_task(task_id: str, task: dict[str, Any]) -> None:
    required = REQUIRED_CHOICE_KEYS if task.get("type") == "choice" else REQUIRED_TASK_KEYS
    missing = [key for key in required if key not in task]
    if missing:
        raise ValueError(f"Task '{task_id}' is missing required keys: {', '.join(missing)}")
    if task.get("type") == "choice":
        options = task["options"]
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"Choice task '{task_id}' needs at least 2 options")
        if not (0 <= task["answer"] < len(options)):
            raise ValueError(f"Choice task '{task_id}' answer index out of range")


def _problem_entry(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    _validate_task(task_id, task)
    if task.get("type") == "choice":
        return {
            "id": task_id,
            "type": "choice",
            "title": task["title"],
            "titleZh": task["title_zh"],
            "difficulty": task["difficulty"],
            "functionName": "",
            "hint": "",
            "hintZh": "",
            "descriptionEn": task["question_en"],
            "descriptionZh": task["question_zh"],
            "options": task["options"],
            "explanationEn": task["explanation_en"],
            "explanationZh": task["explanation_zh"],
            "tests": [],
        }
    entry = {
        "id": task_id,
        "type": "code",
        "title": task["title"],
        "titleZh": task["title_zh"],
        "difficulty": task["difficulty"],
        "functionName": task["function_name"],
        "hint": task["hint"],
        "hintZh": task["hint_zh"],
        "descriptionEn": task["description_en"],
        "descriptionZh": task["description_zh"],
        "tests": task["tests"],
    }
    return entry


def _load_existing_order(output_path: Path) -> list[str]:
    if not output_path.exists():
        return []
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        return [problem["id"] for problem in data.get("problems", [])]
    except (json.JSONDecodeError, KeyError):
        return []


def _ordered_task_ids(existing_order: list[str]) -> list[str]:
    ordered = [task_id for task_id in existing_order if task_id in TASKS]
    seen = set(ordered)
    ordered.extend(task_id for task_id, _ in list_tasks() if task_id not in seen)
    return ordered


def build_problem_catalog(output_path: Path = OUTPUT) -> dict[str, list[dict[str, Any]]]:
    task_ids = _ordered_task_ids(_load_existing_order(output_path))
    return {"problems": [_problem_entry(task_id, TASKS[task_id]) for task_id in task_ids]}


def export_problem_catalog(output_path: Path = OUTPUT) -> dict[str, list[dict[str, Any]]]:
    data = build_problem_catalog(output_path)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return data


def main() -> None:
    data = export_problem_catalog()
    print(f"Written {len(data['problems'])} problems to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
