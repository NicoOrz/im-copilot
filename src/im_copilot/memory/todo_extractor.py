from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from im_copilot.deep_agent.events import record_event
from im_copilot.llm import invoke_structured
from im_copilot.memory.todo_store import TodoRecord, todo_store

_TZ = ZoneInfo("Asia/Hong_Kong")
_MIN_CONFIDENCE = 0.65


class TodoExtractionOutput(BaseModel):
    is_todo: bool = False
    assignee_open_id: str = ""
    title: str = ""
    action: str = ""
    due_at: str = ""
    remind_at: str = ""
    confidence: float = 0.0
    needs_confirmation: bool = False
    scope: Literal["personal", "team", "none"] = "none"


@dataclass
class TodoDraft:
    assignee_open_id: str
    title: str
    action: str
    due_at: datetime
    remind_at: datetime
    source_text: str
    confidence: float = 1.0
    needs_confirmation: bool = False


def extract_todos_from_message(
    text: str,
    *,
    source_open_id: str,
    mentions: list[dict] | None = None,
    now: datetime | None = None,
) -> list[TodoDraft]:
    clean = _clean_text(text)
    if not clean or not source_open_id:
        return []
    current_time = now or datetime.now(_TZ)
    try:
        output = _extract_with_llm(
            clean,
            source_open_id=source_open_id,
            mentions=mentions or [],
            now=current_time,
        )
    except Exception:
        return []
    if not output.is_todo or output.scope == "none" or output.confidence < _MIN_CONFIDENCE:
        return []
    due_at = _parse_iso_time(output.due_at)
    remind_at = _parse_iso_time(output.remind_at) or _default_remind_at(due_at)
    if due_at is None or remind_at is None:
        return []
    assignee = output.assignee_open_id.strip()
    if not assignee:
        return []
    return [
        TodoDraft(
            assignee_open_id=assignee,
            title=(output.title.strip() or _title(clean))[:80],
            action=(output.action.strip() or output.title.strip() or _title(clean))[:80],
            due_at=due_at,
            remind_at=remind_at,
            source_text=clean,
            confidence=output.confidence,
            needs_confirmation=output.needs_confirmation,
        )
    ]


def extract_and_store_todos(
    text: str,
    *,
    chat_id: str,
    message_id: str,
    source_open_id: str,
    source: str = "feishu",
    mentions: list[dict] | None = None,
    now: datetime | None = None,
) -> list[TodoRecord]:
    records: list[TodoRecord] = []
    drafts = extract_todos_from_message(
        text,
        source_open_id=source_open_id,
        mentions=mentions,
        now=now,
    )
    for draft in drafts:
        if draft.needs_confirmation:
            continue
        record = store_todo_draft(
            draft,
            chat_id=chat_id,
            message_id=message_id,
            source_open_id=source_open_id,
            source=source,
        )
        if record is None:
            continue
        records.append(record)
    return records


def store_todo_draft(
    draft: TodoDraft,
    *,
    chat_id: str,
    message_id: str,
    source_open_id: str,
    source: str = "feishu",
) -> TodoRecord | None:
    record = todo_store.create(
        chat_id=chat_id,
        message_id=message_id,
        source_open_id=source_open_id,
        assignee_open_id=draft.assignee_open_id,
        title=draft.title,
        action=draft.action,
        due_at=draft.due_at.isoformat(timespec="minutes"),
        remind_at=draft.remind_at.isoformat(timespec="minutes"),
        source_text=draft.source_text,
    )
    if record is None:
        return None
    record_event(
        chat_id,
        source,
        "todo_detected",
        {
            "id": record.id,
            "chat_id": chat_id,
            "message_id": message_id,
            "source_open_id": source_open_id,
            "assignee_open_id": record.assignee_open_id,
            "title": record.title,
            "action": record.action,
            "due_at": record.due_at,
            "remind_at": record.remind_at,
            "status": record.status,
            "source_text": record.source_text,
            "confidence": draft.confidence,
            "needs_confirmation": draft.needs_confirmation,
        },
    )
    return record


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_with_llm(
    text: str,
    *,
    source_open_id: str,
    mentions: list[dict],
    now: datetime,
) -> TodoExtractionOutput:
    mention_lines = [
        {
            "key": str(item.get("key") or ""),
            "name": str(item.get("name") or ""),
            "open_id": str(item.get("open_id") or ""),
        }
        for item in mentions
    ]
    prompt = (
        "你是个人待办结构化识别器。只根据语义判断是否应创建个人待办，不要使用关键词规则。\n"
        "输出字段含义：is_todo 表示是否创建待办；scope 为 personal/team/none；"
        "assignee_open_id 必须来自 source_open_id 或 mentions；due_at/remind_at 必须为 ISO 8601，无法确定则留空。\n"
        "个人自我提醒可以输出 personal；团队公开事项如果明确要求某人行动，可输出 team；闲聊和无行动事项输出 none。\n"
        "confidence 低于明确可执行程度时设为较低值；信息缺失需要确认时 needs_confirmation=true。\n\n"
        f"当前时间：{now.isoformat(timespec='minutes')}\n"
        f"source_open_id：{source_open_id}\n"
        f"mentions：{mention_lines}\n"
        f"消息：{text}"
    )
    return invoke_structured("todo_extractor", TodoExtractionOutput, prompt, timeout=30, max_retries=1)


def _parse_iso_time(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_TZ)
    return parsed.astimezone(_TZ)


def _default_remind_at(due_at: datetime | None) -> datetime | None:
    if due_at is None:
        return None
    return due_at - timedelta(minutes=10)


def _title(text: str) -> str:
    return text.strip(" ，,。")[:80] or "待办"
