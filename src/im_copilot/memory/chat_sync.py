from __future__ import annotations

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from im_copilot.memory.events import iter_user_messages_for_chat
from im_copilot.memory.todo_extractor import extract_and_store_todos

_TZ = ZoneInfo("Asia/Hong_Kong")


def sync_today(chat_id: str) -> int:
    start = datetime.combine(datetime.now(_TZ).date(), dt_time.min, tzinfo=_TZ).timestamp()
    count = 0
    for event in iter_user_messages_for_chat(chat_id, start):
        payload = event.get("payload", {})
        records = extract_and_store_todos(
            str(payload.get("text") or ""),
            chat_id=chat_id,
            message_id=str(payload.get("message_id") or event.get("id")),
            source_open_id=str(payload.get("source_open_id") or payload.get("user_id") or ""),
            source=str(event.get("source") or "feishu"),
        )
        count += len(records)
    return count
