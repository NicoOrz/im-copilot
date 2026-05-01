from __future__ import annotations

from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo

from im_copilot.memory.events import iter_user_messages_for_chat
from im_copilot.memory.todo_store import todo_store

_TZ = ZoneInfo("Asia/Hong_Kong")


def summary_today(chat_id: str) -> str:
    today = datetime.now(_TZ).date()
    start = datetime.combine(today, dt_time.min, tzinfo=_TZ).timestamp()
    end = datetime.combine(today + timedelta(days=1), dt_time.min, tzinfo=_TZ).timestamp()
    todos = todo_store.created_between(chat_id, start, end)
    messages = iter_user_messages_for_chat(chat_id, start)
    if not todos and not messages:
        return "今天暂无群聊任务。"
    lines = ["今天群聊摘要："]
    if messages:
        lines.append("消息：")
        for event in messages[-10:]:
            text = str(event.get("payload", {}).get("text") or "").strip()
            if text:
                lines.append(f"- {text[:120]}")
    if todos:
        lines.append("待办：")
    for todo in todos:
        lines.append(f"{todo.id}. {todo.title}｜{todo.status}｜{todo.due_at}")
    return "\n".join(lines)
