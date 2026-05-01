from __future__ import annotations

from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo

from im_copilot.deep_agent.events import iter_user_messages_for_chat
from im_copilot.memory.group_board_store import group_board_store

_TZ = ZoneInfo("Asia/Hong_Kong")


def summary_today(chat_id: str) -> str:
    today = datetime.now(_TZ).date()
    start = datetime.combine(today, dt_time.min, tzinfo=_TZ).timestamp()
    end = datetime.combine(today + timedelta(days=1), dt_time.min, tzinfo=_TZ).timestamp()
    board_items = group_board_store.created_between(chat_id, start, end)
    messages = iter_user_messages_for_chat(chat_id, start)
    if not board_items and not messages:
        return "今天暂无群聊看板内容。"
    lines = ["今天群聊摘要："]
    if messages:
        lines.append("消息：")
        for event in messages[-10:]:
            text = str(event.get("payload", {}).get("text") or "").strip()
            if text:
                lines.append(f"- {text[:120]}")
    if board_items:
        lines.append("群看板：")
    for item in board_items:
        owner = item.owner_name or item.owner_open_id or "未指定"
        due = f"｜时间 {item.due_at}" if item.due_at else ""
        lines.append(f"{item.id}. [{item.item_type}] {item.title}｜负责人 {owner}｜{item.status}{due}")
    return "\n".join(lines)
