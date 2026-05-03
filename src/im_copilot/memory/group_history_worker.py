from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

from im_copilot.deep_agent.events import record_event
from im_copilot.lark_bot import LarkBot
from im_copilot.memory.group_board_extractor import extract_and_store_group_board_items
from im_copilot.memory.group_chat_store import group_chat_store
from im_copilot.memory.todo_extractor import extract_and_store_todos

logger = logging.getLogger(__name__)


def start_group_history_loop(
    lark_bot: LarkBot,
    *,
    interval_seconds: int = 60,
    lookback_seconds: int = 300,
) -> threading.Thread:
    thread = threading.Thread(
        target=_loop,
        args=(lark_bot, max(interval_seconds, 10), max(lookback_seconds, 60)),
        daemon=True,
        name="group-history",
    )
    thread.start()
    return thread


def record_bot_joined_group(chat_id: str, *, name: str = "", external: bool = False, tenant_key: str = "") -> None:
    cursor_ms = int((time.time() - _lookback_seconds()) * 1000)
    group_chat_store.upsert(
        chat_id=chat_id,
        name=name,
        external=external,
        tenant_key=tenant_key,
        source="bot_added_event",
        initial_cursor_ms=cursor_ms,
    )
    logger.info("group_history bot_added recorded: chat_id=%s name=%s", chat_id, name)


def _loop(lark_bot: LarkBot, interval_seconds: int, lookback_seconds: int) -> None:
    logger.info(
        "group_history loop start: interval_seconds=%s lookback_seconds=%s",
        interval_seconds,
        lookback_seconds,
    )
    while True:
        try:
            discover_bot_chats(lark_bot, lookback_seconds=lookback_seconds)
            poll_known_chats(lark_bot, lookback_seconds=lookback_seconds)
        except Exception:
            logger.exception("group_history loop error")
        time.sleep(interval_seconds)


def discover_bot_chats(lark_bot: LarkBot, *, lookback_seconds: int) -> int:
    cursor_ms = int((time.time() - lookback_seconds) * 1000)
    count = 0
    for chat in lark_bot.list_bot_chats():
        chat_id = str(chat.get("chat_id") or "")
        if not chat_id:
            continue
        status = str(chat.get("status") or "normal")
        group_chat_store.upsert(
            chat_id=chat_id,
            name=str(chat.get("name") or ""),
            status=status,
            external=bool(chat.get("external")),
            tenant_key=str(chat.get("tenant_key") or ""),
            source="startup_discovery",
            initial_cursor_ms=cursor_ms,
        )
        if status == "normal":
            count += 1
    logger.info("group_history discovered bot chats: count=%s", count)
    return count


def poll_known_chats(lark_bot: LarkBot, *, lookback_seconds: int) -> int:
    processed = 0
    now_sec = int(time.time())
    default_start_ms = int((now_sec - lookback_seconds) * 1000)
    for chat in group_chat_store.active():
        start_ms = chat.last_message_create_time_ms or default_start_ms
        start_sec = max(0, int(start_ms / 1000) - 1)
        messages = lark_bot.list_chat_messages(
            chat.chat_id,
            start_time=start_sec,
            end_time=now_sec,
        )
        if _chat_unavailable(getattr(lark_bot, "last_list_chat_messages_error", None)):
            group_chat_store.mark_inactive(chat.chat_id)
            logger.info("group_history chat marked inactive: chat_id=%s", chat.chat_id)
            continue
        max_seen_ms = start_ms
        for message in messages:
            create_time = int(message.get("create_time") or 0)
            max_seen_ms = max(max_seen_ms, create_time)
            if create_time <= start_ms:
                continue
            if _process_history_message(lark_bot, message):
                processed += 1
        if max_seen_ms > start_ms:
            group_chat_store.update_cursor(chat.chat_id, max_seen_ms)
    if processed:
        logger.info("group_history processed messages: count=%s", processed)
    return processed


def _chat_unavailable(error: Any) -> bool:
    if not isinstance(error, dict):
        return False
    code = error.get("code")
    msg = str(error.get("msg") or "")
    return code == 230002 or "out of the chat" in msg or "NOT be out of the chat" in msg


def _process_history_message(lark_bot: LarkBot, message: dict[str, Any]) -> bool:
    message_id = str(message.get("message_id") or "")
    chat_id = str(message.get("chat_id") or "")
    sender_type = str(message.get("sender_type") or "")
    source_open_id = str(message.get("sender_id") or "")
    if not message_id or not chat_id or sender_type != "user" or not source_open_id:
        return False
    if bool(message.get("deleted")):
        return False
    if str(message.get("msg_type") or "") not in {"text", "post"}:
        return False
    mentions = list(message.get("mentions") or [])
    if _mentions_bot(mentions):
        return False

    text = _extract_text_content(str(message.get("content") or ""))
    text = _replace_mentions(text, mentions)
    if not text.strip():
        return False
    if text.strip().startswith("/"):
        return False

    from im_copilot.lark_handlers import _mark_message_processing

    if not _mark_message_processing(message_id):
        return False

    record_event(
        chat_id,
        "feishu_history",
        "user_message",
        {
            "text": text,
            "chat_id": chat_id,
            "message_id": message_id,
            "source_open_id": source_open_id,
            "user_id": source_open_id,
            "mentions": mentions,
        },
    )
    extract_and_store_todos(
        text,
        chat_id=chat_id,
        message_id=message_id,
        source_open_id=source_open_id,
        source="feishu_history",
        mentions=mentions,
    )
    board_result = extract_and_store_group_board_items(
        text,
        chat_id=chat_id,
        message_id=message_id,
        source_open_id=source_open_id,
        source="feishu_history",
        mentions=mentions,
    )
    if board_result.confirmation_recipients:
        from im_copilot.lark_handlers import _send_meeting_confirmation_cards

        _send_meeting_confirmation_cards(lark_bot, board_result.items, board_result.confirmation_recipients)
    logger.info("group_history message processed: chat_id=%s message_id=%s", chat_id, message_id)
    return True


def _mentions_bot(mentions: list[Any]) -> bool:
    bot_open_id = os.getenv("LARK_BOT_OPEN_ID") or os.getenv("FEISHU_BOT_OPEN_ID") or ""
    if not bot_open_id:
        return False
    for mention in mentions:
        if isinstance(mention, dict) and mention.get("open_id") == bot_open_id:
            return True
    return False


def _replace_mentions(text: str, mentions: list[Any]) -> str:
    result = text
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        key = str(mention.get("key") or "").strip()
        name = str(mention.get("name") or "").strip()
        if key and name:
            result = result.replace(key, f"@{name}")
    return result


def _extract_text_content(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(parsed, dict):
        text = parsed.get("text")
        if isinstance(text, str) and text.strip():
            return text
        return _extract_rich_text(parsed.get("content")).strip()
    return ""


def _extract_rich_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_extract_rich_text(item) for item in value)
    if isinstance(value, dict):
        parts = []
        for key in ("text", "name", "href", "content"):
            item = value.get(key)
            if isinstance(item, (str, list, dict)):
                parts.append(_extract_rich_text(item))
        return "".join(parts)
    return ""


def _lookback_seconds() -> int:
    raw = os.getenv("LARK_GROUP_HISTORY_LOOKBACK_SECONDS", "300")
    if re.fullmatch(r"\d+", raw or ""):
        return max(int(raw), 60)
    return 300
