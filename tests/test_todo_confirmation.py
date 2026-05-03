import pytest
from im_copilot.memory.todo_store import TodoStore, TodoStatus

@pytest.fixture
def store(tmp_path, monkeypatch):
    db = str(tmp_path / "test.sqlite")
    monkeypatch.setenv("TODO_DB", db)
    return TodoStore()

def test_create_awaiting_confirmation(store):
    record = store.create(
        chat_id="chat1",
        message_id="msg1",
        source_open_id="user1",
        assignee_open_id="user2",
        title="测试待办",
        action="完成报告",
        due_at="2026-05-10T18:00",
        remind_at="2026-05-10T17:00",
        source_text="赵磊周五前给报告",
        status="awaiting_confirmation",
    )
    assert record is not None
    assert record.status == "awaiting_confirmation"

def test_get_by_id(store):
    record = store.create(
        chat_id="chat1", message_id="msg2", source_open_id="u1",
        assignee_open_id="u2", title="T", action="A",
        due_at="2026-05-10T18:00", remind_at="2026-05-10T17:00",
        source_text="src", status="awaiting_confirmation",
    )
    fetched = store.get_by_id(record.id)
    assert fetched is not None
    assert fetched.id == record.id

def test_update_status_to_pending(store):
    record = store.create(
        chat_id="chat1", message_id="msg3", source_open_id="u1",
        assignee_open_id="u2", title="T2", action="A2",
        due_at="2026-05-10T18:00", remind_at="2026-05-10T17:00",
        source_text="src", status="awaiting_confirmation",
    )
    ok = store.update_status(record.id, "pending")
    assert ok is True
    updated = store.get_by_id(record.id)
    assert updated.status == "pending"

def test_update_status_to_deleted(store):
    record = store.create(
        chat_id="chat1", message_id="msg4", source_open_id="u1",
        assignee_open_id="u2", title="T3", action="A3",
        due_at="2026-05-10T18:00", remind_at="2026-05-10T17:00",
        source_text="src", status="awaiting_confirmation",
    )
    ok = store.update_status(record.id, "deleted")
    assert ok is True
    updated = store.get_by_id(record.id)
    assert updated.status == "deleted"

from unittest.mock import patch
from im_copilot.memory.todo_extractor import extract_and_store_todos

def test_needs_confirmation_stored_as_awaiting(tmp_path, monkeypatch):
    db = str(tmp_path / "test2.sqlite")
    monkeypatch.setenv("TODO_DB", db)

    from im_copilot.memory.todo_extractor import TodoDraft
    from datetime import datetime, timezone

    draft = TodoDraft(
        assignee_open_id="ou_abc",
        title="给数据口径",
        action="提交数据口径文档",
        due_at=datetime(2026, 5, 8, 18, 0, tzinfo=timezone.utc),
        remind_at=datetime(2026, 5, 8, 17, 0, tzinfo=timezone.utc),
        source_text="赵磊周五18:00前给数据口径",
        confidence=0.5,
        needs_confirmation=True,
    )

    with patch(
        "im_copilot.memory.todo_extractor.extract_todos_from_message",
        return_value=[draft],
    ):
        records = extract_and_store_todos(
            "赵磊周五18:00前给数据口径",
            chat_id="chat1",
            message_id="msg_confirm",
            source_open_id="ou_sender",
        )

    awaiting = [r for r in records if r.status == "awaiting_confirmation"]
    assert len(awaiting) == 1
    assert awaiting[0].assignee_open_id == "ou_abc"
