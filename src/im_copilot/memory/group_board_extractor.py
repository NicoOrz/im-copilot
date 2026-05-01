from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from im_copilot.deep_agent.events import record_event
from im_copilot.memory.group_board_store import GroupBoardItem, group_board_store
from im_copilot.memory.todo_extractor import TodoDraft, store_todo_draft

_TZ = ZoneInfo("Asia/Hong_Kong")
_ASSIGN_RE = re.compile(r"(负责|跟进|完成|整理|提交|发送|更新|确认|推进|处理)")
_MEETING_RE = re.compile(r"(会议|评审|同步|讨论|约个会|开会|会审)")
_TIME_RE = re.compile(r"(?:(明天|明日|今天|今日|后天)|(\d{1,2})月(\d{1,2})日)(早上|上午|中午|下午|晚上|晚间)?(?:(\d{1,2})点)?")


@dataclass
class BoardExtractionResult:
    items: list[GroupBoardItem]
    confirmation_recipients: list[str]


def extract_and_store_group_board_items(
    text: str,
    *,
    chat_id: str,
    message_id: str,
    source_open_id: str,
    source: str = "feishu",
    mentions: list[dict] | None = None,
    now: datetime | None = None,
) -> BoardExtractionResult:
    clean = _clean_text(text)
    if not clean or not chat_id or not message_id:
        return BoardExtractionResult([], [])

    mentions = mentions or []
    created: list[GroupBoardItem] = []
    confirmation_recipients: list[str] = []
    current_time = now or datetime.now(_TZ)

    assignment = _extract_assignment(clean, mentions)
    if assignment is not None:
        owner_open_id, owner_name = assignment
        due_at, remind_at = _parse_time(clean, current_time)
        item = group_board_store.create(
            chat_id=chat_id,
            message_id=message_id,
            item_type="assignment",
            title=_title(clean),
            owner_open_id=owner_open_id,
            owner_name=owner_name,
            due_at=due_at.isoformat(timespec="minutes") if due_at else "",
            status="open",
            source_open_id=source_open_id,
            source_text=clean,
        )
        if item:
            created.append(item)
            record_event(chat_id, source, "todo_detected", _event_payload(item, "group_assignment"))
        if owner_open_id and due_at and remind_at:
            store_todo_draft(
                TodoDraft(
                    assignee_open_id=owner_open_id,
                    title=_title(clean),
                    action=_title(clean),
                    due_at=due_at,
                    remind_at=remind_at,
                    source_text=clean,
                    confidence=0.9,
                    needs_confirmation=False,
                ),
                chat_id=chat_id,
                message_id=message_id,
                source_open_id=source_open_id,
                source=source,
            )

    if _MEETING_RE.search(clean):
        due_at, _ = _parse_time(clean, current_time)
        recipients = _mention_open_ids(mentions)
        item = group_board_store.create(
            chat_id=chat_id,
            message_id=message_id,
            item_type="meeting",
            title=_title(clean),
            owner_open_id="",
            owner_name="",
            due_at=due_at.isoformat(timespec="minutes") if due_at else "",
            status="pending_confirmation",
            source_open_id=source_open_id,
            source_text=clean,
            metadata_json=json.dumps({"recipients": recipients}, ensure_ascii=False),
        )
        if item:
            created.append(item)
            confirmation_recipients.extend(recipients)
            record_event(chat_id, source, "todo_detected", _event_payload(item, "meeting_candidate"))

    return BoardExtractionResult(created, _dedupe(confirmation_recipients))


def _extract_assignment(text: str, mentions: list[dict]) -> tuple[str, str] | None:
    if not _ASSIGN_RE.search(text):
        return None
    for mention in mentions:
        key = str(mention.get("key") or mention.get("name") or "").strip()
        open_id = str(mention.get("open_id") or "").strip()
        name = str(mention.get("name") or key).strip()
        if key and key in text:
            return open_id, name
    explicit = re.search(r"([\u4e00-\u9fffA-Za-z0-9_]{2,12})(负责|跟进|完成|整理|提交|发送|更新|确认|推进|处理)", text)
    if explicit:
        return "", explicit.group(1)
    return None


def _mention_open_ids(mentions: list[dict]) -> list[str]:
    return _dedupe(str(item.get("open_id") or "") for item in mentions)


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
        base = now.replace(year=now.year, month=int(month), day=int(day))
        if base.date() < now.date():
            base = base.replace(year=now.year + 1)

    has_time = bool(period or hour_text)
    hour = _hour(period, hour_text, has_time)
    due_at = base.replace(hour=hour, minute=0, second=0, microsecond=0)
    remind_at = due_at - timedelta(minutes=10) if has_time else due_at.replace(hour=8, minute=50)
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


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _title(text: str) -> str:
    return text.strip(" ，,。")[:80] or "群聊事项"


def _dedupe(values) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result


def _event_payload(item: GroupBoardItem, kind: str) -> dict:
    return {
        "kind": kind,
        "id": item.id,
        "chat_id": item.chat_id,
        "message_id": item.message_id,
        "item_type": item.item_type,
        "title": item.title,
        "owner_open_id": item.owner_open_id,
        "owner_name": item.owner_name,
        "due_at": item.due_at,
        "status": item.status,
        "source_open_id": item.source_open_id,
        "source_text": item.source_text,
    }
