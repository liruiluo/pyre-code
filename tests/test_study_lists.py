"""Regression tests for persistent, per-session personal study lists."""

import pytest
from fastapi import HTTPException

from grading_service import main as service


@pytest.fixture()
def learner(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "_DB_PATH", str(tmp_path / "study-lists.db"))
    token = "learner-1"
    user_id = service.get_or_create_user(service.UserRequest(sessionToken=token))["userId"]
    return token, user_id


def change(token, action, *, list_id=None, name=None, task_id=None):
    return service.update_study_lists(
        service.StudyListAction(
            sessionToken=token, action=action, listId=list_id, name=name, taskId=task_id
        )
    )["lists"]


def test_defaults_created_once_and_persist(learner):
    token, user_id = learner
    initial = service.get_study_lists(user_id)["lists"]
    assert [item["name"] for item in initial] == ["不会的题", "已会的题", "细节易错"]
    assert service.get_study_lists(user_id)["lists"] == initial

    for item in initial:
        change(token, "delete", list_id=item["id"])

    # Deliberately deleting all three defaults must not recreate them.
    assert service.get_study_lists(user_id)["lists"] == []


def test_multiple_lists_and_repeated_add_are_idempotent(learner):
    token, user_id = learner
    a, b, _ = service.get_study_lists(user_id)["lists"]

    change(token, "add", list_id=a["id"], task_id="mha")
    change(token, "add", list_id=a["id"], task_id="mha")
    changed = change(token, "add", list_id=b["id"], task_id="mha")
    assert changed[0]["problemIds"] == ["mha"]
    assert changed[1]["problemIds"] == ["mha"]

    changed = change(token, "remove", list_id=a["id"], task_id="mha")
    assert changed[0]["problemIds"] == []
    assert changed[1]["problemIds"] == ["mha"]
    assert service.get_study_lists(user_id)["lists"] == changed


def test_create_rename_duplicate_and_invalid_problem(learner):
    token, _ = learner
    created = change(token, "create", name="  高频复习  ")
    item = created[-1]
    assert item["name"] == "高频复习"

    renamed = change(token, "rename", list_id=item["id"], name="面试再练")
    assert renamed[-1]["name"] == "面试再练"

    with pytest.raises(HTTPException) as conflict:
        change(token, "create", name="面试再练")
    assert conflict.value.status_code == 409

    with pytest.raises(HTTPException) as invalid:
        change(token, "add", list_id=item["id"], task_id="unknown_problem")
    assert invalid.value.status_code == 404

    with pytest.raises(HTTPException) as empty:
        change(token, "rename", list_id=item["id"], name=" ")
    assert empty.value.status_code == 400


def test_session_ownership(learner):
    token, user_id = learner
    first = service.get_study_lists(user_id)["lists"][0]
    other_token = "learner-2"
    other_id = service.get_or_create_user(
        service.UserRequest(sessionToken=other_token)
    )["userId"]
    other_lists = service.get_study_lists(other_id)["lists"]
    assert other_lists[0]["id"] != first["id"]

    with pytest.raises(HTTPException) as denied:
        change(other_token, "add", list_id=first["id"], task_id="mha")
    assert denied.value.status_code == 404
    assert service.get_study_lists(user_id)["lists"][0]["problemIds"] == []
