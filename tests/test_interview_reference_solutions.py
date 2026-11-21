"""Executable quality gate for the 49 selected ML/post-training interview exercises.

Use: python -m pytest -q tests/test_interview_reference_solutions.py
These tests only execute trusted repository reference solutions and test definitions.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_solutions import build_solution_entry
from export_problems import build_problem_catalog
from torch_judge.tasks import TASKS

CATALOG_PATH = ROOT / "web" / "src" / "lib" / "problems.json"
SOLUTIONS_PATH = ROOT / "web" / "src" / "lib" / "solutions.json"
PATHS_PATH = ROOT / "web" / "src" / "lib" / "paths.json"
ALL_PATHS = json.loads(PATHS_PATH.read_text(encoding="utf-8"))["paths"]
INTERVIEW_PATH = next(path for path in ALL_PATHS if path["id"] == "post-training-interview-essentials")
IDS = INTERVIEW_PATH["problems"]


def test_interview_path_contains_49_unique_registered_code_tasks():
    assert len(IDS) == len(set(IDS)) == 49
    assert all(task_id in TASKS and TASKS[task_id].get("type") != "choice" for task_id in IDS)
    assert "kl_estimators_k1_k2_k3" in IDS


def test_problem_catalog_matches_python_task_definitions():
    actual = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert actual == build_problem_catalog()


def test_reference_solutions_match_python_task_definitions():
    actual = json.loads(SOLUTIONS_PATH.read_text(encoding="utf-8"))
    for task_id in IDS:
        assert actual[task_id] == build_solution_entry(TASKS[task_id]), task_id


@pytest.mark.parametrize("task_id", IDS)
def test_reference_solution_against_all_visible_cases(task_id):
    task = TASKS[task_id]
    namespace = {"torch": torch, "Tensor": torch.Tensor, "nn": nn, "F": F, "math": math, "np": np}
    exec(task["solution"], namespace)
    function_name = task["function_name"]
    assert function_name in namespace
    assert task["tests"], f"{task_id} has no tests"
    for case in task["tests"]:
        torch.manual_seed(0)
        local = namespace.copy()
        code = case["code"].replace("{fn}", function_name)
        try:
            exec(compile(code, f"<{task_id}/{case['name']}>", "exec"), local)
        except Exception as exc:
            pytest.fail(f"{task_id} - {case['name']}: {type(exc).__name__}: {exc}", pytrace=True)
