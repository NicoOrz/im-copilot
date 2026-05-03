from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from im_copilot.deep_agent.events import record_event
from im_copilot.llm import get_llm_for_node
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
    try:
        text = _generate_reminder_text(todo)
    except Exception as exc:
        record_event(
            todo.chat_id,
            "feishu",
            "error",
            {"error": "todo reminder LLM failed", "todo_id": todo.id, "detail": str(exc)},
        )
        return False
    if not text:
        record_event(
            todo.chat_id,
            "feishu",
            "error",
            {"error": "todo reminder LLM returned empty text", "todo_id": todo.id},
        )
        return False
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


def _generate_reminder_text(todo: TodoRecord) -> str:
    prompt = (
        "请为以下待办事项生成一条飞书单聊提醒，发送给任务负责人本人。\n"
        "要求：第二人称（你），中文，简短自然（不超过 50 字）；"
        "必须包含任务内容和截止时间；"
        "以 title 和 action 为准，source_text 仅供参考语境；"
        "不要在正文中出现字段名（title、action、due_at 等词）；"
        "不编造 source_text 之外的信息；只输出提醒正文。\n\n"
        f"title: {todo.title}\n"
        f"action: {todo.action}\n"
        f"due_at: {todo.due_at}\n"
        f"source_text: {todo.source_text}"
    )
    content = get_llm_for_node("todo_reminder", timeout=20, max_retries=1).invoke(prompt).content
    return str(content or "").strip()
