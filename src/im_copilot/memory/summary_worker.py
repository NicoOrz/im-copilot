from __future__ import annotations

import logging
import re
from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo

from im_copilot.deep_agent.events import iter_user_messages_for_chat
from im_copilot.llm import get_llm_for_node
from im_copilot.memory.group_board_store import group_board_store

logger = logging.getLogger(__name__)
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
        lines.append("聊天概要：")
        lines.append(_summarize_messages(messages))
    if board_items:
        lines.append("群看板：")
    for item in board_items:
        owner = item.owner_name or item.owner_open_id or "未指定"
        due = f"｜时间 {item.due_at}" if item.due_at else ""
        lines.append(f"{item.id}. [{item.item_type}] {_clean_message_text(item.title)}｜负责人 {owner}｜{item.status}{due}")
    return "\n".join(lines)


def _summarize_messages(messages: list[dict]) -> str:
    texts = []
    for event in messages[-80:]:
        text = _clean_message_text(str(event.get("payload", {}).get("text") or ""))
        if text:
            texts.append(text[:300])
    if not texts:
        return "暂无可总结的文本消息。"
    prompt = (
        "请基于以下群聊消息生成今日聊天概要。\n"
        "要求：只总结事实，不扩写；忽略寒暄和无意义数字；输出 3-6 条短句；"
        "若包含任务、会议、地点、时间，请明确写出。\n\n"
        + "\n".join(f"- {text}" for text in texts)
    )
    try:
        content = get_llm_for_node("group_summary", timeout=30, max_retries=1).invoke(prompt).content
    except Exception as exc:
        logger.warning("group summary LLM failed: %s", exc)
        return "智能摘要生成失败，请稍后重试。"
    summary = str(content or "").strip()
    return summary or "暂无可总结的有效内容。"


def _clean_message_text(text: str) -> str:
    cleaned = re.sub(r"@_user_\d+", "@某成员", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()
