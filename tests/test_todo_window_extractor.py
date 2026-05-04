from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from im_copilot.deep_agent.events import record_event
from im_copilot.memory.todo_extractor import (
    ExistingTodoBrief,
    TodoExtractionItem,
    TodoExtractionOutput,
    TodoDraft,
    TodoUpdate,
    WindowMessage,
    assemble_window,
    extract_and_store_todos_from_window,
    extract_todos_from_window,
    load_open_todos_brief,
)
from im_copilot.memory.todo_store import TodoStore

_TZ = ZoneInfo("Asia/Hong_Kong")
_NOW = datetime(2026, 5, 4, 10, 0, tzinfo=_TZ)


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    db = str(tmp_path / "todos.sqlite")
    monkeypatch.setenv("TODO_DB", db)
    monkeypatch.setenv("AGENT_EVENTS_DB", db)
    return db


def _output(*items: TodoExtractionItem) -> TodoExtractionOutput:
    return TodoExtractionOutput(items=list(items))


def _item(**kwargs) -> TodoExtractionItem:
    data = {
        "is_todo": True,
        "links_to_existing_id": "",
        "assignee_open_id": "ou_liran",
        "title": "整理技术限制说明",
        "action_phrase": "整理技术限制说明",
        "due_at": "2026-05-05T10:00+08:00",
        "remind_at": "",
        "confidence": 0.9,
        "needs_confirmation": False,
        "scope": "team",
        "reasoning": "触发消息确认了前文承诺的截止时间",
    }
    data.update(kwargs)
    return TodoExtractionItem(**data)


def test_window_extraction_returns_create_update_and_filters_invalid_items(monkeypatch):
    window = [
        WindowMessage("m1", "ou_liran", "李然", "我明早整理技术限制说明", _NOW.timestamp(), False),
        WindowMessage("m2", "ou_wang", "王敏", "那就明早 10 点前", _NOW.timestamp(), True),
    ]
    existing = [
        ExistingTodoBrief(
            id=7,
            assignee_open_id="ou_liran",
            title="整理技术限制说明",
            action_phrase="整理技术限制说明",
            due_at="2026-05-04T18:00+08:00",
            status="pending",
        )
    ]
    monkeypatch.setattr(
        "im_copilot.memory.todo_extractor.invoke_structured",
        lambda *args, **kwargs: _output(
            _item(links_to_existing_id="", title="整理评审材料", action_phrase="整理评审材料"),
            _item(links_to_existing_id="7", confidence=0.2),
            _item(is_todo=False),
            _item(assignee_open_id="ou_other"),
            _item(links_to_existing_id="", confidence=0.2),
        ),
    )

    results = extract_todos_from_window(window, existing_open_todos=existing, now=_NOW)

    assert [type(item) for item in results] == [TodoDraft, TodoUpdate]
    assert results[0].title == "整理评审材料"
    assert results[1].existing_id == 7
    assert results[1].due_at.isoformat(timespec="minutes") == "2026-05-05T10:00+08:00"
    assert "我明早整理技术限制说明" in results[1].source_text
    assert "那就明早 10 点前" in results[1].source_text


def test_unknown_existing_link_is_created_as_new(monkeypatch):
    window = [
        WindowMessage("m1", "ou_liran", "李然", "我明早整理技术限制说明", _NOW.timestamp(), True),
    ]
    monkeypatch.setattr(
        "im_copilot.memory.todo_extractor.invoke_structured",
        lambda *args, **kwargs: _output(_item(links_to_existing_id="404")),
    )

    results = extract_todos_from_window(window, existing_open_todos=[], now=_NOW)

    assert len(results) == 1
    assert isinstance(results[0], TodoDraft)
    assert results[0].assignee_open_id == "ou_liran"


def test_window_prompt_includes_mention_open_id_mapping(monkeypatch):
    captured = {}
    window = [
        WindowMessage(
            "m1",
            "ou_wang",
            "",
            "请她明天下午给演示风格参考",
            _NOW.timestamp(),
            True,
            mentions=({"open_id": "ou_chen", "name": "陈悦", "key": "@_user_1"},),
        )
    ]

    def fake_invoke(*args, **kwargs):
        captured["prompt"] = args[2]
        return _output()

    monkeypatch.setattr("im_copilot.memory.todo_extractor.invoke_structured", fake_invoke)

    extract_todos_from_window(window, existing_open_todos=[], now=_NOW)

    assert "窗口 mentions" in captured["prompt"]
    assert "ou_chen" in captured["prompt"]
    assert "陈悦" in captured["prompt"]


def test_tmp_md_window_extraction_creates_and_updates_todos(db_env, monkeypatch):
    chat_id = "oc_tmp"
    name_to_open_id = {
        "王敏": "ou_wang",
        "周琪": "ou_zhou",
        "李然": "ou_liran",
        "陈悦": "ou_chen",
        "赵磊": "ou_zhao",
    }
    responses = []
    for _ in range(21):
        responses.append(_output())
    responses[6] = _output(
        _item(
            assignee_open_id="ou_zhao",
            title="提交数据口径",
            action_phrase="提交数据口径",
            due_at="2026-05-08T18:00+08:00",
        )
    )
    responses[8] = _output(
        _item(
            assignee_open_id="ou_chen",
            title="整理演示风格参考",
            action_phrase="整理演示风格参考",
            due_at="2026-05-05T15:00+08:00",
        )
    )
    responses[17] = _output(
        _item(
            assignee_open_id="ou_liran",
            title="整理技术限制说明",
            action_phrase="整理技术限制说明",
            due_at="2026-05-04T18:00+08:00",
        )
    )
    responses[18] = _output(
        _item(
            links_to_existing_id="3",
            assignee_open_id="ou_liran",
            title="整理技术限制说明",
            action_phrase="整理技术限制说明",
            due_at="2026-05-05T12:00+08:00",
        )
    )
    responses[19] = _output(
        _item(
            links_to_existing_id="3",
            assignee_open_id="ou_liran",
            title="整理技术限制说明",
            action_phrase="整理技术限制说明",
            due_at="2026-05-05T10:00+08:00",
        )
    )
    iterator = iter(responses)
    monkeypatch.setattr(
        "im_copilot.memory.todo_extractor.invoke_structured",
        lambda *args, **kwargs: next(iterator),
    )

    for index, line in enumerate(Path("tmp.md").read_text(encoding="utf-8").splitlines()):
        clock, rest = line.split("] ", 1)
        name, text = rest.split("：", 1)
        message_id = f"m{index + 1}"
        event_id = record_event(
            chat_id,
            "feishu",
            "user_message",
            {
                "text": text,
                "chat_id": chat_id,
                "message_id": message_id,
                "source_open_id": name_to_open_id[name],
                "user_id": name_to_open_id[name],
                "mentions": [],
                "clock": clock.strip("["),
            },
        )
        records = extract_and_store_todos_from_window(
            chat_id=chat_id,
            message_id=message_id,
            source_open_id=name_to_open_id[name],
            source="feishu",
            window=assemble_window(chat_id, event_id, now=_NOW),
            existing_open_todos=load_open_todos_brief(chat_id),
            now=_NOW,
        )
        assert all(record.status in {"pending", "awaiting_confirmation"} for record in records)

    todos = TodoStore().list(chat_id=chat_id, status="")

    tech = [todo for todo in todos if "技术限制说明" in todo.title]
    assert len(tech) == 1
    assert tech[0].assignee_open_id == "ou_liran"
    assert tech[0].due_at == "2026-05-05T10:00+08:00"

    style = [todo for todo in todos if "演示风格参考" in todo.title]
    assert len(style) == 1
    assert style[0].assignee_open_id == "ou_chen"
    assert style[0].due_at == "2026-05-05T15:00+08:00"

    data = [todo for todo in todos if "数据口径" in todo.title]
    assert len(data) == 1
    assert data[0].assignee_open_id == "ou_zhao"
    assert data[0].due_at == "2026-05-08T18:00+08:00"

    wang_deadline = [
        todo for todo in todos
        if "明早 10 点前" in todo.title or todo.assignee_open_id == "ou_wang"
    ]
    assert wang_deadline == []
    assert 2 <= len(todos) <= 4


def test_mention_assignment_false_output_is_ignored_and_true_output_is_stored(db_env, monkeypatch):
    chat_id = "oc_mention"
    window = [
        WindowMessage(
            "m1",
            "ou_wang",
            "王敏",
            "@何添 今天下班前整理技术限制说明",
            _NOW.timestamp(),
            True,
            mentions=({"open_id": "ou_hetian", "name": "何添", "key": "@_user_1"},),
        )
    ]
    responses = iter([
        _output(_item(is_todo=False, assignee_open_id="ou_hetian")),
        _output(
            _item(
                assignee_open_id="ou_hetian",
                title="整理技术限制说明",
                action_phrase="整理技术限制说明",
                due_at="2026-05-04T18:00+08:00",
            )
        ),
    ])
    monkeypatch.setattr(
        "im_copilot.memory.todo_extractor.invoke_structured",
        lambda *args, **kwargs: next(responses),
    )

    ignored = extract_todos_from_window(window, existing_open_todos=[], now=_NOW)
    stored = extract_and_store_todos_from_window(
        chat_id=chat_id,
        message_id="m1",
        source_open_id="ou_wang",
        window=window,
        existing_open_todos=[],
        now=_NOW,
    )

    assert ignored == []
    assert len(stored) == 1
    assert stored[0].assignee_open_id == "ou_hetian"
    assert stored[0].action_phrase == "整理技术限制说明"


def test_linked_update_keeps_correct_assignee(db_env, monkeypatch):
    store = TodoStore()
    existing = store.create(
        chat_id="oc_link",
        message_id="m0",
        source_open_id="ou_hetian",
        assignee_open_id="ou_hetian",
        title="整理技术限制说明",
        action_phrase="整理技术限制说明",
        due_at="2026-05-04T18:00+08:00",
        remind_at="2026-05-04T17:50+08:00",
        source_text="我今天下班前整理技术限制说明",
    )
    assert existing is not None
    window = [
        WindowMessage("m0", "ou_hetian", "何添", "我今天下班前整理技术限制说明", _NOW.timestamp(), False),
        WindowMessage(
            "m1",
            "ou_wang",
            "王敏",
            "@何添 那就明早 10 点前。",
            _NOW.timestamp(),
            True,
            mentions=({"open_id": "ou_hetian", "name": "何添", "key": "@_user_1"},),
        ),
    ]
    monkeypatch.setattr(
        "im_copilot.memory.todo_extractor.invoke_structured",
        lambda *args, **kwargs: _output(
            _item(
                links_to_existing_id=str(existing.id),
                assignee_open_id="ou_hetian",
                title="整理技术限制说明",
                action_phrase="整理技术限制说明",
                due_at="2026-05-05T10:00+08:00",
            )
        ),
    )

    records = extract_and_store_todos_from_window(
        chat_id="oc_link",
        message_id="m1",
        source_open_id="ou_wang",
        window=window,
        existing_open_todos=load_open_todos_brief("oc_link"),
        now=_NOW,
    )

    assert len(records) == 1
    assert records[0].id == existing.id
    assert records[0].assignee_open_id == "ou_hetian"
    assert records[0].due_at == "2026-05-05T10:00+08:00"
    assert len(store.list(chat_id="oc_link", status="")) == 1


def test_action_phrase_is_stored_and_can_be_empty(db_env, monkeypatch):
    window = [
        WindowMessage("m1", "ou_wang", "王敏", "两项待办", _NOW.timestamp(), True),
    ]
    monkeypatch.setattr(
        "im_copilot.memory.todo_extractor.invoke_structured",
        lambda *args, **kwargs: _output(
            _item(
                assignee_open_id="ou_wang",
                title="提交数据口径",
                action_phrase="提交数据口径",
                due_at="2026-05-08T18:00+08:00",
            ),
            _item(
                assignee_open_id="ou_wang",
                title="确认隐私权限边界",
                action_phrase="",
                due_at="2026-05-09T18:00+08:00",
            ),
        ),
    )

    records = extract_and_store_todos_from_window(
        chat_id="oc_phrase",
        message_id="m1",
        source_open_id="ou_wang",
        window=window,
        existing_open_todos=[],
        now=_NOW,
    )

    assert [record.action_phrase for record in records] == ["提交数据口径", ""]


def test_links_to_existing_id_updates_and_empty_link_creates(db_env, monkeypatch):
    store = TodoStore()
    existing = store.create(
        chat_id="oc_mix",
        message_id="m0",
        source_open_id="ou_liran",
        assignee_open_id="ou_liran",
        title="整理技术限制说明",
        action_phrase="整理技术限制说明",
        due_at="2026-05-04T18:00+08:00",
        remind_at="2026-05-04T17:50+08:00",
        source_text="整理技术限制说明",
    )
    assert existing is not None
    window = [
        WindowMessage("m1", "ou_liran", "李然", "更新技术说明，并新增评审材料", _NOW.timestamp(), True),
    ]
    monkeypatch.setattr(
        "im_copilot.memory.todo_extractor.invoke_structured",
        lambda *args, **kwargs: _output(
            _item(
                links_to_existing_id=str(existing.id),
                assignee_open_id="ou_liran",
                title="整理技术限制说明",
                action_phrase="整理技术限制说明",
                due_at="2026-05-05T10:00+08:00",
            ),
            _item(
                links_to_existing_id="",
                assignee_open_id="ou_liran",
                title="整理评审材料",
                action_phrase="整理评审材料",
                due_at="2026-05-06T10:00+08:00",
            ),
        ),
    )

    records = extract_and_store_todos_from_window(
        chat_id="oc_mix",
        message_id="m1",
        source_open_id="ou_liran",
        window=window,
        existing_open_todos=load_open_todos_brief("oc_mix"),
        now=_NOW,
    )

    assert [record.id for record in records] == [existing.id, existing.id + 1]
    assert records[0].due_at == "2026-05-05T10:00+08:00"
    assert records[1].title == "整理评审材料"
