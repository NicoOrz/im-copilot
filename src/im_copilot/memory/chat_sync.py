from __future__ import annotations

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from im_copilot.deep_agent.events import iter_user_messages_for_chat
from im_copilot.memory.group_board_extractor import extract_and_store_group_board_items
from im_copilot.memory.todo_extractor import (
    assemble_window_from_events,
    extract_and_store_todos_from_window,
    load_open_todos_brief,
)

_TZ = ZoneInfo("Asia/Hong_Kong")


def sync_today(chat_id: str) -> int:
    start = datetime.combine(datetime.now(_TZ).date(), dt_time.min, tzinfo=_TZ).timestamp()
    count = 0
    events = iter_user_messages_for_chat(chat_id, start)
    for index, event in enumerate(events):
        payload = event.get("payload", {})
        mentions = list(payload.get("mentions") or [])
        event_time = datetime.fromtimestamp(float(event.get("created_at") or 0.0), _TZ)
        records = extract_and_store_todos_from_window(
            chat_id=chat_id,
            message_id=str(payload.get("message_id") or event.get("id")),
            source_open_id=str(payload.get("source_open_id") or payload.get("user_id") or ""),
            source=str(event.get("source") or "feishu"),
            window=assemble_window_from_events(events[:index + 1], int(event.get("id") or 0)),
            existing_open_todos=load_open_todos_brief(chat_id),
            is_bot_request=bool(payload.get("is_bot_request")),
            now=event_time,
        )
        board = extract_and_store_group_board_items(
            str(payload.get("text") or ""),
            chat_id=chat_id,
            message_id=str(payload.get("message_id") or event.get("id")),
            source_open_id=str(payload.get("source_open_id") or payload.get("user_id") or ""),
            source=str(event.get("source") or "feishu"),
            mentions=mentions,
        )
        count += len(records) + len(board.items)
    return count
