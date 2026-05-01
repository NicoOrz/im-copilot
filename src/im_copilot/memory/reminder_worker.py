from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from im_copilot.memory.events import record_event
from im_copilot.memory.todo_store import TodoRecord, todo_store

logger = logging.getLogger(__name__)
_TZ = ZoneInfo("Asia/Hong_Kong")


def send_due_reminders(lark_bot) -> int:
    sent = 0
    for todo in todo_store.due_for_reminder(datetime.now(_TZ)):
        if _send_one(lark_bot, todo):
            todo_store.mark_reminded(todo.id)
            record_event(
                todo.chat_id,
                "feishu",
                "todo_reminded",
                {"id": todo.id, "assignee_open_id": todo.assignee_open_id, "title": todo.title},
            )
            sent += 1
    return sent


def start_reminder_loop(lark_bot, interval_seconds: int = 60) -> threading.Thread:
    def _loop() -> None:
        while True:
            try:
                send_due_reminders(lark_bot)
            except Exception as exc:
                logger.exception("todo reminder loop failed")
                record_event("todo-reminder", "feishu", "error", {"error": str(exc)})
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, daemon=True, name="todo-reminder")
    thread.start()
    return thread


def _send_one(lark_bot, todo: TodoRecord) -> bool:
    text = f"任务提醒：{todo.title}\n截止：{todo.due_at}\n来源：{todo.source_text}"
    resp = lark_bot.send_text_to_open_id(todo.assignee_open_id, text)
    if resp.get("code") == 0:
        return True
    record_event(
        todo.chat_id,
        "feishu",
        "error",
        {"error": "todo reminder send failed", "todo_id": todo.id, "response": resp},
    )
    return False
