from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from im_copilot.deep_agent.events import record_event
from im_copilot.memory.todo_store import TodoRecord, todo_store

_TZ = ZoneInfo("Asia/Hong_Kong")
_TIME_RE = re.compile(r"(?:(明天|明日|今天|今日|后天)|(\d{1,2})月(\d{1,2})日)(早上|上午|中午|下午|晚上|晚间)?(?:(\d{1,2})点)?")
_ACTION_RE = re.compile(r"(把|将|负责|完成|提交|交付|发送|整理|评审|确认|更新|发给|交到)")


@dataclass
class TodoDraft:
    assignee_open_id: str
    title: str
    action: str
    due_at: datetime
    remind_at: datetime
    source_text: str


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
    if not _ACTION_RE.search(clean):
        return []
    due_at, remind_at = _parse_time(clean, now or datetime.now(_TZ))
    if due_at is None or remind_at is None:
        return []
    assignee = _assignee_open_id(clean, source_open_id, mentions or [])
    if not assignee:
        return []
    title = _title(clean)
    return [
        TodoDraft(
            assignee_open_id=assignee,
            title=title,
            action=title,
            due_at=due_at,
            remind_at=remind_at,
            source_text=clean,
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
            continue
        records.append(record)
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
            },
        )
    return records


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _assignee_open_id(text: str, source_open_id: str, mentions: list[dict]) -> str:
    if "我" in text:
        return source_open_id
    for mention in mentions:
        key = str(mention.get("key") or mention.get("name") or "")
        open_id = str(mention.get("open_id") or "")
        if key and key in text and open_id:
            return open_id
    return ""


def _parse_time(text: str, now: datetime) -> tuple[datetime | None, datetime | None]:
    match = _TIME_RE.search(text)
    if not match:
        return None, None
    relative, month, day, period, hour_text = match.groups()
    base = now
    if relative in {"明天", "明日"}:
        base = now + timedelta(days=1)
    elif relative == "后天":
        base = now + timedelta(days=2)
    elif month and day:
        year = now.year
        base = now.replace(year=year, month=int(month), day=int(day))
        if base.date() < now.date():
            base = base.replace(year=year + 1)

    has_time = bool(period or hour_text)
    hour = _hour(period, hour_text, has_time)
    due_at = base.replace(hour=hour, minute=0, second=0, microsecond=0)
    if has_time:
        remind_at = due_at - timedelta(minutes=10)
    else:
        remind_at = due_at.replace(hour=8, minute=50)
    return due_at, remind_at


def _hour(period: str | None, hour_text: str | None, has_time: bool) -> int:
    if hour_text:
        hour = int(hour_text)
        if period in {"下午", "晚上", "晚间"} and hour < 12:
            hour += 12
        return hour
    if period in {"早上", "上午"}:
        return 9
    if period == "中午":
        return 12
    if period == "下午":
        return 15
    if period in {"晚上", "晚间"}:
        return 20
    return 18 if not has_time else 9


def _title(text: str) -> str:
    stripped = re.sub(r"^(明天|明日|今天|今日|后天)(早上|上午|中午|下午|晚上|晚间)?", "", text)
    stripped = re.sub(r"^\d{1,2}月\d{1,2}日(早上|上午|中午|下午|晚上|晚间)?", "", stripped)
    stripped = stripped.strip(" ，,。")
    return stripped[:80] or text[:80]
