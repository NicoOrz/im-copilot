from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from im_copilot.deep_agent.events import record_event
from im_copilot.memory.group_board_extractor import (
    BoardCandidate,
    BoardExtractionOutput,
    extract_and_store_group_board_items,
)
from im_copilot.memory.group_board_store import GroupBoardStore
from im_copilot.memory.todo_store import TodoStore

_TZ = ZoneInfo("Asia/Hong_Kong")
_NOW = datetime(2026, 5, 4, 10, 0, tzinfo=_TZ)


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    db = str(tmp_path / "board.sqlite")
    monkeypatch.setenv("AGENT_EVENTS_DB", db)
    monkeypatch.setenv("GROUP_BOARD_DB", db)
    monkeypatch.setenv("TODO_DB", db)
    return db


def _candidate(**kwargs) -> BoardCandidate:
    data = {
        "item_type": "assignment",
        "title": "整理演示材料",
        "status": "open",
        "owner_open_id": "ou_liran",
        "owner_name": "李然",
        "due_at": "2026-05-05T10:00+08:00",
        "start_at": "",
        "end_at": "",
        "recipients": [],
        "confidence": 0.9,
        "reason": "触发消息形成团队公开事项",
        "metadata": {"public_scope": "team"},
        "links_to_existing_id": "",
    }
    data.update(kwargs)
    return BoardCandidate(**data)


def _output(*items: BoardCandidate) -> BoardExtractionOutput:
    return BoardExtractionOutput(items=list(items))


def _record(chat_id: str, message_id: str, text: str, open_id: str, mentions: list[dict] | None = None) -> int:
    return record_event(
        chat_id,
        "feishu",
        "user_message",
        {
            "text": text,
            "chat_id": chat_id,
            "message_id": message_id,
            "source_open_id": open_id,
            "user_id": open_id,
            "mentions": mentions or [],
        },
    )


def test_creates_assignment_meeting_and_decision(db_env, monkeypatch):
    chat_id = "oc_board_create"
    mentions = [{"open_id": "ou_liran", "name": "李然", "key": "@_user_1"}]
    _record(chat_id, "m1", "@李然 明天 10 点前整理演示材料；下午 3 点开评审会；采用 A 方案", "ou_sender", mentions)

    monkeypatch.setattr(
        "im_copilot.memory.group_board_extractor.invoke_structured",
        lambda *args, **kwargs: _output(
            _candidate(),
            _candidate(
                item_type="meeting",
                title="评审会",
                start_at="2026-05-04T15:00+08:00",
                due_at="",
                recipients=["ou_sender", "ou_liran"],
                metadata={"public_scope": "team", "location": "3F-01", "topic": "评审会"},
            ),
            _candidate(
                item_type="decision",
                title="采用 A 方案",
                due_at="",
                metadata={"public_scope": "team"},
            ),
        ),
    )

    result = extract_and_store_group_board_items(
        "@李然 明天 10 点前整理演示材料；下午 3 点开评审会；采用 A 方案",
        chat_id=chat_id,
        message_id="m1",
        source_open_id="ou_sender",
        mentions=mentions,
        now=_NOW,
    )

    assert [item.item_type for item in result.items] == ["assignment", "meeting", "decision"]
    meeting = result.items[1]
    assert meeting.status == "pending_confirmation"
    assert result.confirmation_recipients == ["ou_sender", "ou_liran"]


def test_links_to_existing_id_updates_meeting_assignment_and_decision(db_env, monkeypatch):
    chat_id = "oc_board_update"
    store = GroupBoardStore()
    meeting = store.create(
        chat_id=chat_id,
        message_id="old1",
        item_type="meeting",
        title="评审会",
        owner_open_id="ou_liran",
        owner_name="李然",
        due_at="2026-05-04T15:00+08:00",
        status="pending_confirmation",
        source_open_id="ou_sender",
        source_text="下午 3 点开评审会",
        metadata_json=json.dumps({"topic": "评审会", "location": "3F-01"}, ensure_ascii=False),
    )
    assignment = store.create(
        chat_id=chat_id,
        message_id="old2",
        item_type="assignment",
        title="整理演示材料",
        owner_open_id="ou_liran",
        owner_name="李然",
        due_at="2026-05-05T10:00+08:00",
        status="open",
        source_open_id="ou_sender",
        source_text="整理演示材料",
    )
    decision = store.create(
        chat_id=chat_id,
        message_id="old3",
        item_type="decision",
        title="采用 A 方案",
        owner_open_id="ou_liran",
        owner_name="李然",
        status="open",
        source_open_id="ou_sender",
        source_text="采用 A 方案",
    )
    assert meeting and assignment and decision
    mentions = [{"open_id": "ou_liran", "name": "李然", "key": "@_user_1"}]
    _record(chat_id, "m2", "@李然 评审会改到 4F-02，材料周三下班前，A 方案描述改为灰度发布", "ou_sender", mentions)

    monkeypatch.setattr(
        "im_copilot.memory.group_board_extractor.invoke_structured",
        lambda *args, **kwargs: _output(
            _candidate(
                item_type="meeting",
                title="评审会",
                start_at="2026-05-04T15:00+08:00",
                due_at="",
                metadata={"public_scope": "team", "location": "4F-02", "topic": "评审会"},
                links_to_existing_id=str(meeting.id),
            ),
            _candidate(
                title="整理演示材料",
                due_at="2026-05-06T18:00+08:00",
                links_to_existing_id=str(assignment.id),
            ),
            _candidate(
                item_type="decision",
                title="采用灰度发布方案",
                due_at="",
                metadata={"public_scope": "team", "wording": "采用灰度发布方案"},
                links_to_existing_id=str(decision.id),
            ),
        ),
    )

    result = extract_and_store_group_board_items(
        "@李然 评审会改到 4F-02，材料周三下班前，A 方案描述改为灰度发布",
        chat_id=chat_id,
        message_id="m2",
        source_open_id="ou_sender",
        mentions=mentions,
        now=_NOW,
    )

    assert [item.id for item in result.items] == [meeting.id, assignment.id, decision.id]
    updated_meeting = store.get(meeting.id)
    updated_assignment = store.get(assignment.id)
    updated_decision = store.get(decision.id)
    assert updated_meeting is not None
    assert json.loads(updated_meeting.metadata_json)["location"] == "4F-02"
    assert updated_assignment is not None
    assert updated_assignment.due_at == "2026-05-06T18:00+08:00"
    assert updated_decision is not None
    assert updated_decision.title == "采用灰度发布方案"


def test_unknown_owner_open_id_is_dropped(db_env, monkeypatch):
    chat_id = "oc_board_unknown_owner"
    _record(chat_id, "m1", "明天整理演示材料", "ou_sender")
    monkeypatch.setattr(
        "im_copilot.memory.group_board_extractor.invoke_structured",
        lambda *args, **kwargs: _output(_candidate(owner_open_id="ou_unknown")),
    )

    result = extract_and_store_group_board_items(
        "明天整理演示材料",
        chat_id=chat_id,
        message_id="m1",
        source_open_id="ou_sender",
        now=_NOW,
    )

    assert result.items == []
    assert GroupBoardStore().list(chat_id=chat_id) == []


def test_unknown_owner_name_is_cleared(db_env, monkeypatch):
    chat_id = "oc_board_unknown_name"
    _record(chat_id, "m1", "我明天整理演示材料", "ou_liran")
    monkeypatch.setattr(
        "im_copilot.memory.group_board_extractor.invoke_structured",
        lambda *args, **kwargs: _output(_candidate(owner_name="张伟")),
    )

    result = extract_and_store_group_board_items(
        "我明天整理演示材料",
        chat_id=chat_id,
        message_id="m1",
        source_open_id="ou_liran",
        now=_NOW,
    )

    assert len(result.items) == 1
    assert result.items[0].owner_name == ""


def test_unknown_existing_link_is_created_and_warned(db_env, monkeypatch, caplog):
    chat_id = "oc_board_unknown_link"
    _record(chat_id, "m1", "我明天整理演示材料", "ou_liran")
    monkeypatch.setattr(
        "im_copilot.memory.group_board_extractor.invoke_structured",
        lambda *args, **kwargs: _output(_candidate(links_to_existing_id="404")),
    )

    result = extract_and_store_group_board_items(
        "我明天整理演示材料",
        chat_id=chat_id,
        message_id="m1",
        source_open_id="ou_liran",
        now=_NOW,
    )

    assert len(result.items) == 1
    assert result.items[0].message_id == "m1"
    assert "unknown existing id" in caplog.text


def test_low_confidence_new_item_is_dropped(db_env, monkeypatch):
    chat_id = "oc_board_low_confidence"
    _record(chat_id, "m1", "我明天整理演示材料", "ou_liran")
    monkeypatch.setattr(
        "im_copilot.memory.group_board_extractor.invoke_structured",
        lambda *args, **kwargs: _output(_candidate(confidence=0.2)),
    )

    result = extract_and_store_group_board_items(
        "我明天整理演示材料",
        chat_id=chat_id,
        message_id="m1",
        source_open_id="ou_liran",
        now=_NOW,
    )

    assert result.items == []
    assert GroupBoardStore().list(chat_id=chat_id) == []


def test_empty_llm_output_has_no_side_effects(db_env, monkeypatch):
    chat_id = "oc_board_empty"
    _record(chat_id, "m1", "大家早", "ou_sender")
    monkeypatch.setattr(
        "im_copilot.memory.group_board_extractor.invoke_structured",
        lambda *args, **kwargs: _output(),
    )

    result = extract_and_store_group_board_items(
        "大家早",
        chat_id=chat_id,
        message_id="m1",
        source_open_id="ou_sender",
        now=_NOW,
    )

    assert result.items == []
    assert GroupBoardStore().list(chat_id=chat_id) == []


def test_group_board_path_does_not_create_personal_todo(db_env, monkeypatch):
    chat_id = "oc_board_no_personal_todo"
    _record(chat_id, "m1", "我明天 10 点前整理演示材料", "ou_liran")
    todo_calls: list[object] = []

    def fail_create(self, **kwargs):
        todo_calls.append(kwargs)
        raise AssertionError("group board must not create personal todo")

    monkeypatch.setattr(TodoStore, "create", fail_create)
    monkeypatch.setattr(
        "im_copilot.memory.group_board_extractor.invoke_structured",
        lambda *args, **kwargs: _output(_candidate()),
    )

    result = extract_and_store_group_board_items(
        "我明天 10 点前整理演示材料",
        chat_id=chat_id,
        message_id="m1",
        source_open_id="ou_liran",
        now=_NOW,
    )

    assert len(result.items) == 1
    assert todo_calls == []


def test_prompt_uses_trigger_window_and_existing_items(db_env, monkeypatch):
    chat_id = "oc_board_prompt"
    for index in range(55):
        _record(chat_id, f"m{index}", f"历史消息 {index}", "ou_sender")
    _record(chat_id, "m55", "我明天整理演示材料", "ou_liran")
    existing = GroupBoardStore().create(
        chat_id=chat_id,
        message_id="old",
        item_type="assignment",
        title="整理演示材料",
        owner_open_id="ou_liran",
        owner_name="李然",
        status="open",
        source_open_id="ou_sender",
        source_text="旧事项",
    )
    assert existing
    captured: dict[str, str] = {}

    def fake_invoke(*args, **kwargs):
        captured["prompt"] = args[2]
        return _output(_candidate())

    monkeypatch.setattr("im_copilot.memory.group_board_extractor.invoke_structured", fake_invoke)

    extract_and_store_group_board_items(
        "我明天整理演示材料",
        chat_id=chat_id,
        message_id="m55",
        source_open_id="ou_liran",
        now=_NOW,
    )

    prompt = captured["prompt"]
    assert "[1]" in prompt
    assert "← TRIGGER" in prompt
    assert "id=" in prompt
    assert f"id={existing.id}" in prompt
    assert "links_to_existing_id" in prompt
