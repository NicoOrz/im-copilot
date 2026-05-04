"""Event handlers for Feishu (Lark) bot WebSocket events.

This module wires incoming messages and card actions into the LangGraph
pipeline, manages interactive cards, and handles HITL interrupts via the
session manager.

Interaction design:
- All intents use a streaming card for LLM output (typing effect)
- Non-chat intents show progress steps then approval cards
- Chat intents stream the reply directly in the card
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from functools import partial
from typing import Any

import lark_oapi as lark
from pydantic import BaseModel
from lark_oapi.api.im.v1.model.p2_im_chat_member_bot_added_v1 import (
    P2ImChatMemberBotAddedV1,
)
from lark_oapi.api.im.v1.model.p2_im_message_receive_v1 import P2ImMessageReceiveV1
from lark_oapi.api.im.v1.model.p2_im_message_message_read_v1 import (
    P2ImMessageMessageReadV1,
)
from lark_oapi.api.im.v1.model.p2_im_message_reaction_created_v1 import (
    P2ImMessageReactionCreatedV1,
)
from lark_oapi.api.im.v1.model.p2_im_message_reaction_deleted_v1 import (
    P2ImMessageReactionDeletedV1,
)
from lark_oapi.event.callback.model.p2_card_action_trigger import (
    CallBackToast,
    P2CardActionTrigger,
    P2CardActionTriggerResponse,
)

from im_copilot.deep_agent.events import iter_user_messages_for_chat, record_event
from im_copilot.deep_agent.service import run_agent
from im_copilot.llm import get_llm_for_node, invoke_structured
from im_copilot.lark_bot import LarkBot
from im_copilot.lark_card import (
    create_approval_card,
    create_clarification_card,
    create_command_response_card,
    create_meeting_confirmation_card,
    create_progress_card,
    create_result_card,
    create_streaming_card,
    create_todo_confirm_card,
)
from im_copilot.oauth_scopes import user_oauth_scope_string
from im_copilot.session_manager import session_manager
from im_copilot.user_session_store import session_store as user_session_store
from im_copilot.user_token_store import token_store

logger = logging.getLogger(__name__)

_PROCESSED_MESSAGE_TTL_SECONDS = 300
_PERSISTED_MESSAGE_TTL_SECONDS = 7 * 24 * 60 * 60
_ACK_REACTION = "Get"
_DONE_REACTION = "DONE"
_ERROR_REACTION = "CrossMark"
_FEISHU_ARTIFACT_URL_RE = re.compile(
    r"https?://[^\s)\]>\"']*feishu\.cn/"
    r"(?:docx|doc|slides|sheets?|base|wiki|mindnotes|file|minutes)/[^\s)\]>\"']+"
)
_processed_messages: dict[str, float] = {}
_processed_messages_lock = threading.Lock()
_runtime_bot_open_id = ""


def _processed_messages_db_path() -> str:
    return os.getenv("LARK_PROCESSED_DB", ".copilot_dedup.sqlite")


def _mark_message_persisted(message_id: str, now: float) -> bool:
    db_path = _processed_messages_db_path()
    logger.debug("Dedup DB mark start: message_id=%s db_path=%s", message_id, db_path)
    conn = sqlite3.connect(db_path, timeout=1)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS lark_processed_messages "
            "(message_id TEXT PRIMARY KEY, created_at REAL NOT NULL)"
        )
        conn.execute(
            "DELETE FROM lark_processed_messages WHERE created_at < ?",
            (now - _PERSISTED_MESSAGE_TTL_SECONDS,),
        )
        cursor = conn.execute(
            "INSERT OR IGNORE INTO lark_processed_messages (message_id, created_at) VALUES (?, ?)",
            (message_id, now),
        )
        conn.commit()
        marked = cursor.rowcount == 1
        logger.debug("Dedup DB mark result: message_id=%s marked=%s", message_id, marked)
        return marked
    finally:
        conn.close()


def _mark_message_processing(message_id: str) -> bool:
    logger.debug("Dedup memory check start: message_id=%s", message_id)
    if not message_id:
        logger.debug("Dedup bypass: empty message_id")
        return True

    now = time.time()
    with _processed_messages_lock:
        expired = [
            key for key, ts in _processed_messages.items()
            if now - ts > _PROCESSED_MESSAGE_TTL_SECONDS
        ]
        for key in expired:
            _processed_messages.pop(key, None)

        if message_id in _processed_messages:
            logger.debug("Dedup memory hit: message_id=%s", message_id)
            return False

        if not _mark_message_persisted(message_id, now):
            logger.debug("Dedup persisted hit: message_id=%s", message_id)
            return False

        _processed_messages[message_id] = now
        logger.debug("Dedup accepted: message_id=%s", message_id)
        return True


def _extract_text_content(content: str) -> str:
    """Parse the JSON content string from a Feishu text or rich-text message."""
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


def _replace_mentions(text: str, mentions: list[dict[str, Any]]) -> str:
    result = text
    for mention in mentions:
        key = str(mention.get("key") or "").strip()
        name = str(mention.get("name") or "").strip()
        if key and name:
            result = result.replace(key, f"@{name}")
    return result


def _make_thread_id(chat_id: str, message_id: str | None = None) -> str:
    with _chat_generation_lock:
        gen = _chat_generation.get(chat_id)
    if gen is None:
        gen = _load_generation(chat_id)
        with _chat_generation_lock:
            _chat_generation[chat_id] = gen
    return f"{chat_id}:gen{gen}"


def _reset_chat_thread(chat_id: str) -> str:
    with _chat_generation_lock:
        gen = _chat_generation.get(chat_id, 0) + 1
        _chat_generation[chat_id] = gen
    _persist_generation(chat_id, gen)
    return f"{chat_id}:gen{gen}"


def _generation_db_path() -> str:
    return _processed_messages_db_path()


def _load_generation(chat_id: str) -> int:
    db_path = _generation_db_path()
    conn = sqlite3.connect(db_path, timeout=1)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_thread_generation "
            "(chat_id TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT 0)"
        )
        row = conn.execute(
            "SELECT generation FROM chat_thread_generation WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def _persist_generation(chat_id: str, gen: int) -> None:
    db_path = _generation_db_path()
    conn = sqlite3.connect(db_path, timeout=1)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_thread_generation "
            "(chat_id TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "INSERT INTO chat_thread_generation (chat_id, generation) VALUES (?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET generation = excluded.generation",
            (chat_id, gen),
        )
        conn.commit()
    finally:
        conn.close()


_chat_generation: dict[str, int] = {}
_chat_generation_lock = threading.Lock()

_AT_MENTION_RE = re.compile(r"^@\S+\s*")
_reminder_started = False
_reminder_lock = threading.Lock()


class BoardQueryIntent(BaseModel):
    is_board_query: bool = False
    reason: str = ""


def _is_interrupt_update(step: dict[str, Any]) -> bool:
    """Check whether a stream step represents an ``__interrupt__`` update."""
    return "__interrupt__" in step


def _get_interrupt_info(step: dict[str, Any]) -> dict[str, Any] | None:
    """Extract interrupt payload from a stream step or state dict, if present."""
    interrupt_data = step.get("__interrupt__")
    if interrupt_data is None:
        return None
    if isinstance(interrupt_data, (list, tuple)) and interrupt_data:
        interrupt_data = interrupt_data[0]
    if hasattr(interrupt_data, "value"):
        return interrupt_data.value  # type: ignore[union-attr]
    return interrupt_data  # type: ignore[return-value]


def _mentions_bot(text: str, mentions: list[dict[str, Any]]) -> bool:
    bot_open_ids = {
        value.strip()
        for value in (
            _runtime_bot_open_id,
            os.getenv("LARK_BOT_OPEN_ID"),
            os.getenv("FEISHU_BOT_OPEN_ID"),
        )
        if value and value.strip()
    }
    if bot_open_ids:
        return any(str(m.get("open_id") or "").strip() in bot_open_ids for m in mentions)
    bot_names = {
        value.strip().lstrip("@").lower()
        for value in (
            os.getenv("LARK_BOT_NAME"),
            os.getenv("FEISHU_BOT_NAME"),
            os.getenv("BOT_NAME"),
        )
        if value and value.strip()
    }
    if bot_names and any(str(m.get("name") or "").strip().lstrip("@").lower() in bot_names for m in mentions):
        return True
    return any(str(m.get("mentioned_type") or "").strip().lower() == "bot" for m in mentions)


def _mention_dicts(message: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for mention in getattr(message, "mentions", None) or []:
        user_id = getattr(mention, "id", None)
        result.append({
            "key": getattr(mention, "key", "") or "",
            "name": getattr(mention, "name", "") or "",
            "open_id": getattr(user_id, "open_id", "") if user_id else "",
            "user_id": getattr(user_id, "user_id", "") if user_id else "",
            "mentioned_type": getattr(mention, "mentioned_type", "") or "",
        })
    return result


def _send_streaming_card(lark_bot: LarkBot, chat_id: str) -> tuple[str | None, str | None]:
    """Create and send a streaming card, returning (card_entity_id, message_id)."""
    logger.debug("Streaming card create start: chat_id=%s", chat_id)
    card_json = create_streaming_card(title="生成中...")
    resp = lark_bot.create_card_entity(card_json)
    if resp.get("code") != 0:
        logger.error("Failed to create card entity: %s", resp.get("msg"))
        return None, None

    card_entity_id = resp.get("data", {}).get("card_id")
    logger.debug("Streaming card entity created: chat_id=%s card_entity_id=%s", chat_id, card_entity_id)
    if not card_entity_id:
        logger.error("No card_id in create_card_entity response")
        return None, None

    send_resp = lark_bot.send_card_entity(chat_id, card_entity_id)
    if send_resp.get("code") != 0:
        logger.error("Failed to send card entity: %s", send_resp.get("msg"))
        return card_entity_id, None

    message_id = send_resp.get("data", {}).get("message_id")
    logger.debug("Streaming card sent: chat_id=%s card_entity_id=%s message_id=%s", chat_id, card_entity_id, message_id)
    return card_entity_id, message_id


def _update_card_for_interrupt(
    lark_bot: LarkBot,
    session: dict[str, Any],
    interrupt_info: dict[str, Any],
    thread_id: str | None = None,
) -> None:
    """Replace the progress card with an approval or clarification card."""
    gate = interrupt_info.get("gate", "")
    thread_id = thread_id or session["thread_id"]
    chat_id = session.get("chat_id", thread_id)
    card_message_id = session.get("card_message_id") or session.get("card_id")

    logger.debug("Interrupt card update start: thread_id=%s gate=%s card_message_id=%s", thread_id, gate, card_message_id)

    if gate == "plan_approval":
        plan = interrupt_info.get("plan", [])
        intent_type = interrupt_info.get("intent_type", "")
        intent_params = interrupt_info.get("intent_params", {})
        card = create_approval_card(
            plan=plan,
            intent_type=intent_type,
            intent_params=intent_params,
            thread_id=thread_id,
        )
    elif gate == "clarification":
        questions = interrupt_info.get("questions", [])
        card = create_clarification_card(questions=questions, thread_id=thread_id)
    else:
        logger.warning("Unknown interrupt gate: %s", gate)
        return

    if card_message_id:
        logger.debug("Interrupt patch existing card: message_id=%s gate=%s", card_message_id, gate)
        lark_bot.patch_message(card_message_id, card)
    else:
        logger.debug("Interrupt send new card: chat_id=%s gate=%s", chat_id, gate)
        resp = lark_bot.send_card(chat_id, card)
        message_id = resp.get("data", {}).get("message_id")
        if message_id:
            session_manager.update_session(
                thread_id,
                card_id=message_id,
                card_message_id=message_id,
            )


def _update_progress_card(
    lark_bot: LarkBot,
    session: dict[str, Any],
    node_name: str,
    detail: str,
) -> None:
    """Update the progress card with the current node status."""
    card_message_id = session.get("card_message_id") or session.get("card_id")
    if not card_message_id:
        logger.debug("Progress card skipped: no card message id thread_id=%s node=%s", session.get("thread_id"), node_name)
        return

    logger.debug("Progress card update: message_id=%s node=%s detail=%s", card_message_id, node_name, detail)
    card = create_progress_card(title=detail)
    card["header"]["title"]["content"] = f"步骤: {node_name}"
    card["body"]["elements"][0]["content"] = f"**{detail}**"
    card["body"]["elements"][1]["text"]["content"] = f"当前节点: {node_name}"

    lark_bot.patch_message(card_message_id, card)


def _finalize_card(
    lark_bot: LarkBot,
    session: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Send the final result card when the pipeline finishes."""
    chat_id = session.get("chat_id", session["thread_id"])
    card_message_id = session.get("card_message_id") or session.get("card_id")
    summary = state.get("summary", "处理完成")
    artifacts = state.get("artifacts", {})
    logger.debug(
        "Finalize card start: thread_id=%s message_id=%s summary_len=%s artifact_count=%s",
        chat_id,
        card_message_id,
        len(summary),
        len(artifacts),
    )

    artifacts_plain: dict[str, str] = {}
    doc_links: list[dict] = []
    for k, v in artifacts.items():
        if isinstance(v, dict):
            artifacts_plain[k] = f"{v.get('title', k)} — {v.get('status', 'unknown')}"
            if v.get("url"):
                doc_links.append({"title": v.get("title", k), "url": v["url"]})
        else:
            artifacts_plain[k] = str(v)

    card = create_result_card(
        summary=summary,
        artifacts=artifacts_plain,
        doc_links=doc_links or None,
    )
    if card_message_id:
        logger.debug("Finalize patch existing card: message_id=%s", card_message_id)
        lark_bot.patch_message(card_message_id, card)
    else:
        logger.debug("Finalize send new card: chat_id=%s", chat_id)
        lark_bot.send_card(chat_id, card)


def _send_oauth_prompt(lark_bot: LarkBot, chat_id: str, open_id: str) -> None:
    """Send an OAuth authorization link to the user."""
    app_id = os.environ.get("FEISHU_APP_ID") or os.environ.get("LARK_APP_ID", "")
    callback_url = os.environ.get("OAUTH_CALLBACK_URL", "")
    import urllib.parse
    auth_url = (
        "https://accounts.feishu.cn/open-apis/authen/v1/authorize"
        f"?app_id={urllib.parse.quote(app_id)}"
        f"&redirect_uri={urllib.parse.quote(callback_url)}"
        f"&response_type=code"
        f"&scope={urllib.parse.quote(user_oauth_scope_string())}"
        f"&state={urllib.parse.quote(open_id)}"
    )
    lark_bot.send_text(
        chat_id,
        f"首次使用需要授权，请点击以下链接完成授权后重新发送消息：\n{auth_url}",
    )


def _make_card_response(toast: str | None = None) -> P2CardActionTriggerResponse:
    """Build a card action response with an optional toast message."""
    resp = P2CardActionTriggerResponse()
    if toast:
        t = CallBackToast()
        t.type = "info"
        t.content = toast
        resp.toast = t
    return resp


def _ws_broadcast(open_id: str, data: dict) -> None:
    try:
        from im_copilot.web.ws import ws_manager

        ws_manager.broadcast_to_user_threadsafe(open_id, data)
    except Exception:
        logger.debug("WS broadcast skipped (no event loop or ws_manager unavailable)")


def _ack_message(lark_bot: LarkBot, message_id: str) -> None:
    _react_message(lark_bot, message_id, _ACK_REACTION)


def _complete_message(lark_bot: LarkBot, message_id: str) -> None:
    _react_message(lark_bot, message_id, _DONE_REACTION)


def _fail_message(lark_bot: LarkBot, message_id: str) -> None:
    _react_message(lark_bot, message_id, _ERROR_REACTION)


def _react_message(lark_bot: LarkBot, message_id: str, emoji_type: str) -> None:
    if not message_id:
        return
    lark_bot.add_reaction(message_id, emoji_type)


def _reply_result(lark_bot: LarkBot, message_id: str, result: Any) -> None:
    text = _result_reply_text(result)
    lark_bot.reply_text(message_id, text)


def _result_reply_text(result: Any) -> str:
    artifact_lines = _artifact_link_lines(getattr(result, "artifacts", {}) or {})
    if artifact_lines:
        return "好的，已生成，以下是产物链接：\n" + "\n".join(artifact_lines)
    return str(getattr(result, "summary", "") or "处理完成").strip()


def _unverified_artifact_link_lines(result: Any, source_text: str = "") -> list[str]:
    summary = str(getattr(result, "summary", "") or "")
    artifact_urls = {
        _normalize_url(str(artifact.get("url") or ""))
        for artifact in (getattr(result, "artifacts", {}) or {}).values()
        if isinstance(artifact, dict) and str(artifact.get("url") or "").strip()
    }
    source_urls = {
        _normalize_url(match.group(0))
        for match in _FEISHU_ARTIFACT_URL_RE.finditer(source_text or "")
    }
    lines: list[str] = []
    for match in _FEISHU_ARTIFACT_URL_RE.finditer(summary):
        url = _normalize_url(match.group(0))
        if url in artifact_urls or url in source_urls:
            continue
        lines.append(f"- {url}")
    return lines


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("。；;，,.)）]")


def _artifact_link_lines(artifacts: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("doc", "whiteboard", "slide"):
        artifact = artifacts.get(key)
        if not isinstance(artifact, dict):
            continue
        url = str(artifact.get("url") or "").strip()
        if not url:
            continue
        title = str(artifact.get("title") or artifact.get("kind") or key).strip()
        lines.append(f"- {title}：{url}")
    return lines


def _missing_artifact_link_lines(artifacts: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("doc", "whiteboard", "slide"):
        artifact = artifacts.get(key)
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("url") or "").strip():
            continue
        title = str(artifact.get("title") or artifact.get("kind") or key).strip()
        status = str(artifact.get("status") or "unknown").strip()
        lines.append(f"- {title}：{status}")
    return lines


def _send_private_group_command_response(
    lark_bot: LarkBot,
    *,
    chat_id: str,
    open_id: str,
    text: str,
    command: str,
    title: str = "命令结果",
) -> bool:
    if not open_id:
        record_event(
            chat_id,
            "feishu",
            "error",
            {"error": "missing command sender open_id", "command": command},
        )
        return False
    resp = lark_bot.send_ephemeral_card(
        chat_id,
        open_id,
        create_command_response_card(text, title=title),
    )
    if resp.get("code") == 0:
        return True
    record_event(
        chat_id,
        "feishu",
        "error",
        {
            "error": "command ephemeral card send failed",
            "command": command,
            "open_id": open_id,
            "response": resp,
        },
    )
    return False


def _send_meeting_confirmation_cards(lark_bot: LarkBot, items: list[Any], recipients: list[str]) -> None:
    if not items:
        return
    meeting_items = [item for item in items if getattr(item, "item_type", "") == "meeting"]
    if not meeting_items:
        return
    item = meeting_items[-1]
    metadata = _json_dict(getattr(item, "metadata_json", ""))
    recipients = _json_list(metadata.get("recipients")) or _json_list(recipients)
    if not recipients:
        return
    start = str(metadata.get("start") or getattr(item, "due_at", "") or "")
    end = str(metadata.get("end") or "")
    source_summary = _meeting_card_source_summary(
        title=str(item.title or "会议"),
        start=start,
        end=end,
        source_text=str(item.source_text or ""),
    )
    if not source_summary:
        record_event(
            str(item.chat_id),
            "feishu",
            "error",
            {
                "error": "meeting card source summary failed",
                "board_item_id": int(item.id),
            },
        )
        return
    card = create_meeting_confirmation_card(
        board_item_id=int(item.id),
        title=str(item.title or "会议"),
        start=start,
        end=end,
        source_text=source_summary,
        attendee_ids=recipients,
    )
    for open_id in recipients:
        resp = lark_bot.send_ephemeral_card(str(item.chat_id), open_id, card)
        if resp.get("code") != 0:
            record_event(
                str(item.chat_id),
                "feishu",
                "error",
                {
                    "error": "meeting ephemeral card send failed",
                    "board_item_id": int(item.id),
                    "open_id": open_id,
                    "response": resp,
                },
            )


def _meeting_card_source_summary(
    *,
    title: str,
    start: str,
    end: str,
    source_text: str,
) -> str:
    prompt = (
        "请为飞书会议确认卡片生成“依据”字段。\n"
        "要求：中文，1 句话，40 字以内；只说明为什么这是一个会议候选；"
        "保留关键时间、地点或对象；不要输出完整原文；不要添加原文没有的信息；不要 Markdown。\n\n"
        f"会议事项：{title}\n"
        f"时间：{start} ~ {end}\n"
        f"群聊原文：{source_text}"
    )
    try:
        content = get_llm_for_node("meeting_card_summary", timeout=20, max_retries=1).invoke(prompt).content
    except Exception as exc:
        logger.warning("meeting card source summary LLM failed: %s", exc)
        return ""
    text = str(content or "").strip()
    text = re.sub(r"\s+", " ", text).strip("` \n")
    if not text:
        return ""
    return text[:60]


def _json_dict(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _command_response_title(command: str, args: str) -> str:
    if command == "todo" and args.strip().lower() == "board":
        return "今日群看板"
    return "命令结果"


def _is_board_query(text: str) -> bool:
    clean = _AT_MENTION_RE.sub("", text or "").strip()
    if not clean:
        return False
    prompt = (
        "判断用户是否在群聊中向机器人查询今日群看板、群任务、今日待办或群聊总结。\n"
        "只判断查询意图，不处理创建任务、安排会议、普通聊天或业务执行请求。\n"
        "输出结构化结果。\n\n"
        f"用户消息：{clean}"
    )
    try:
        result = invoke_structured("group_board_query_intent", BoardQueryIntent, prompt, timeout=15, max_retries=1)
    except Exception as exc:
        logger.warning("board query intent LLM failed: %s", exc)
        return False
    return bool(result.is_board_query)


def _process_audio_message(
    file_key: str,
    message_id: str,
    chat_id: str,
    lark_bot: LarkBot,
    open_id: str = "",
    *,
    chat_type: str = "",
    mentions: list[dict[str, Any]] | None = None,
    sender_type: str = "",
) -> None:
    """Download and transcribe an audio message, then route as text."""
    logger.info(
        "audio_message start: message_id=%s chat_id=%s open_id=%s",
        message_id,
        chat_id,
        open_id,
    )
    _ack_message(lark_bot, message_id)

    audio_bytes = lark_bot.download_message_resource(message_id, file_key)
    if not audio_bytes:
        logger.error("audio_message download failed: message_id=%s", message_id)
        _fail_message(lark_bot, message_id)
        lark_bot.reply_text(message_id, "语音下载失败，请重试。")
        return

    text = lark_bot.recognize_speech(audio_bytes)
    if not text:
        logger.error("audio_message asr failed: message_id=%s", message_id)
        _fail_message(lark_bot, message_id)
        lark_bot.reply_text(message_id, "语音识别失败，请重试或改用文字。")
        return

    logger.info(
        "audio_message asr success: message_id=%s text_len=%s text_preview=%r",
        message_id,
        len(text),
        text[:80],
    )
    _process_message(
        text,
        chat_id,
        message_id,
        lark_bot,
        open_id,
        chat_type=chat_type,
        mentions=mentions or [],
        sender_type=sender_type,
    )


def _send_todo_confirmation_cards(
    lark_bot: LarkBot,
    records: list,
    *,
    source_open_id: str = "",
) -> None:
    from im_copilot.memory.todo_store import TodoRecord
    for record in records:
        if not isinstance(record, TodoRecord):
            continue
        if record.status != "awaiting_confirmation":
            continue
        recipient = record.assignee_open_id or source_open_id
        if not recipient:
            logger.warning(
                "todo_confirm: no recipient for todo_id=%s title=%r",
                record.id,
                record.title,
            )
            continue
        card = create_todo_confirm_card(
            todo_id=record.id,
            title=record.title,
            action_phrase=record.action_phrase,
            due_at=record.due_at,
            source_text=record.source_text,
            source_open_id=source_open_id,
        )
        try:
            lark_bot.send_card_to_open_id(recipient, card)
            logger.info(
                "todo_confirm card sent: todo_id=%s recipient=%s",
                record.id,
                recipient,
            )
        except Exception:
            logger.exception(
                "todo_confirm card send failed: todo_id=%s recipient=%s",
                record.id,
                recipient,
            )


def _process_message(
    text: str,
    chat_id: str,
    message_id: str,
    lark_bot: LarkBot,
    open_id: str = "",
    *,
    chat_type: str = "",
    mentions: list[dict[str, Any]] | None = None,
    sender_type: str = "",
) -> None:
    from im_copilot.commands import parse_command, execute_command
    from im_copilot.memory.group_board_extractor import extract_and_store_group_board_items
    from im_copilot.memory.todo_extractor import (
        assemble_window,
        extract_and_store_todos_from_window,
        load_open_todos_brief,
    )

    clean_text = _AT_MENTION_RE.sub("", text.strip())
    mentions = mentions or []
    is_group = (chat_type or "").lower() not in {"", "p2p", "private"}
    is_command = parse_command(clean_text) is not None
    is_mentioned = _mentions_bot(text, mentions)
    is_bot_message = sender_type.lower() == "bot"
    logger.info(
        "lark_message route_check chat_id=%s message_id=%s chat_type=%s text_len=%s clean_len=%s is_group=%s is_command=%s is_mentioned=%s is_bot_message=%s open_id=%s",
        chat_id,
        message_id,
        chat_type,
        len(text),
        len(clean_text),
        is_group,
        is_command,
        is_mentioned,
        is_bot_message,
        open_id,
    )

    if is_group and not is_command and not is_mentioned:
        if not is_bot_message:
            logger.info(
                "lark_message group_memory_only chat_id=%s message_id=%s open_id=%s",
                chat_id,
                message_id,
                open_id,
            )
            event_id = record_event(
                chat_id,
                "feishu",
                "user_message",
                {
                    "text": text,
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "source_open_id": open_id,
                    "user_id": open_id,
                    "mentions": mentions,
                },
            )
            todo_records = extract_and_store_todos_from_window(
                chat_id=chat_id,
                message_id=message_id,
                source_open_id=open_id,
                window=assemble_window(chat_id, event_id),
                existing_open_todos=load_open_todos_brief(chat_id),
                is_bot_request=False,
            )
            _send_todo_confirmation_cards(lark_bot, todo_records, source_open_id=open_id)
            board_result = extract_and_store_group_board_items(
                text,
                chat_id=chat_id,
                message_id=message_id,
                source_open_id=open_id,
                mentions=mentions,
            )
            if board_result.confirmation_recipients:
                _send_meeting_confirmation_cards(lark_bot, board_result.items, board_result.confirmation_recipients)
        return

    parsed = parse_command(clean_text)
    if parsed is not None:
        cmd_name, cmd_args = parsed
        thread_id = _make_thread_id(chat_id)
        logger.info(
            "lark_message command chat_id=%s message_id=%s thread_id=%s command=%s args_len=%s",
            chat_id,
            message_id,
            thread_id,
            cmd_name,
            len(cmd_args),
        )
        try:
            command_source = "feishu_group" if is_group else "feishu"
            cmd_result = execute_command(cmd_name, cmd_args, chat_id, thread_id, source=command_source, user_id=open_id)
            if cmd_result.metadata.get("action") == "reset_thread":
                _reset_chat_thread(chat_id)
            if is_group:
                _send_private_group_command_response(
                    lark_bot,
                    chat_id=chat_id,
                    open_id=open_id,
                    text=cmd_result.response_text,
                    command=cmd_name,
                    title=_command_response_title(cmd_name, cmd_args),
                )
            else:
                lark_bot.reply_text(message_id, cmd_result.response_text)
            _complete_message(lark_bot, message_id)
        except Exception as exc:
            logger.exception("Command error for thread_id=%s", thread_id)
            _fail_message(lark_bot, message_id)
            error_text = f"处理出错：{exc}"
            if is_group:
                _send_private_group_command_response(
                    lark_bot,
                    chat_id=chat_id,
                    open_id=open_id,
                    text=error_text,
                    command=cmd_name,
                )
            else:
                lark_bot.send_text(chat_id, error_text)
        return

    if is_group and is_mentioned and _is_board_query(clean_text):
        from im_copilot.memory.summary_worker import summary_today

        lark_bot.reply_text(message_id, summary_today(chat_id))
        _complete_message(lark_bot, message_id)
        return

    thread_id = _make_thread_id(chat_id)
    source = "feishu"
    if is_group and is_mentioned and not is_bot_message:
        record_event(
            chat_id,
            "feishu",
            "user_message",
            {
                "text": text,
                "chat_id": chat_id,
                "message_id": message_id,
                "source_open_id": open_id,
                "user_id": open_id,
                "mentions": mentions,
                "is_bot_request": True,
            },
        )
    logger.info(
        "lark_message agent_start chat_id=%s message_id=%s thread_id=%s open_id=%s text_len=%s",
        chat_id,
        message_id,
        thread_id,
        open_id,
        len(text),
    )

    if open_id:
        user_session_store.record_session(open_id, thread_id, "feishu", chat_id=chat_id)
        _ws_broadcast(open_id, {"type": "message", "thread_id": thread_id, "data": {"role": "user", "content": text}})

    # Check user token; prompt OAuth if missing
    user_access_token = ""
    if open_id:
        user_access_token = token_store.get(open_id) or ""
        if not user_access_token:
            logger.info(
                "lark_message oauth_required chat_id=%s message_id=%s thread_id=%s open_id=%s",
                chat_id,
                message_id,
                thread_id,
                open_id,
            )
            _send_oauth_prompt(lark_bot, chat_id, open_id)
            _complete_message(lark_bot, message_id)
            return
    logger.info(
        "lark_message oauth_status chat_id=%s message_id=%s thread_id=%s has_user_token=%s",
        chat_id,
        message_id,
        thread_id,
        bool(user_access_token),
    )

    try:
        _sync_recent_chat_context(lark_bot, chat_id, current_message_id=message_id)
        result = run_agent(
            text,
            thread_id=thread_id,
            source=source,
            chat_id=chat_id,
            message_id=message_id,
            user_id=open_id,
            user_access_token=user_access_token,
        )
        logger.info(
            "lark_message agent_result chat_id=%s message_id=%s thread_id=%s status=%s summary_len=%s artifact_keys=%s error_len=%s",
            chat_id,
            message_id,
            thread_id,
            result.status,
            len(result.summary or ""),
            sorted(result.artifacts.keys()),
            len(result.error or ""),
        )
        if result.status == "error":
            _fail_message(lark_bot, message_id)
            lark_bot.reply_text(message_id, f"处理出错：{result.error}")
            return
        missing_artifact_links = _missing_artifact_link_lines(result.artifacts)
        if missing_artifact_links:
            _fail_message(lark_bot, message_id)
            lark_bot.reply_text(
                message_id,
                "处理失败：产物已生成记录不完整，缺少可访问链接。\n" + "\n".join(missing_artifact_links),
            )
            return
        unverified_artifact_links = _unverified_artifact_link_lines(result, text)
        if unverified_artifact_links:
            _fail_message(lark_bot, message_id)
            lark_bot.reply_text(
                message_id,
                "处理失败：回复中包含未验证的飞书产物链接。\n" + "\n".join(unverified_artifact_links),
            )
            return
        logger.info(
            "lark_message reply_result chat_id=%s message_id=%s thread_id=%s artifact_keys=%s summary_preview=%r",
            chat_id,
            message_id,
            thread_id,
            sorted(result.artifacts.keys()),
            (result.summary or "").replace("\n", "\\n")[:200],
        )
        _reply_result(lark_bot, message_id, result)
        _complete_message(lark_bot, message_id)
        if open_id:
            _ws_broadcast(open_id, {
                "type": "complete",
                "thread_id": thread_id,
                "data": {"summary": result.summary, "artifacts": result.artifacts},
            })

    except Exception as exc:
        logger.exception("Pipeline error for thread_id=%s", thread_id)
        _fail_message(lark_bot, message_id)
        lark_bot.send_text(chat_id, f"处理出错：{exc}")
        session_manager.delete_session(thread_id)


def _sync_recent_chat_context(
    lark_bot: LarkBot,
    chat_id: str,
    *,
    current_message_id: str = "",
) -> int:
    if not chat_id:
        return 0
    end_time = int(time.time())
    start_time = end_time - _chat_context_lookback_seconds()
    try:
        messages = lark_bot.list_chat_messages(
            chat_id,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception:
        logger.exception("chat_context sync failed: chat_id=%s", chat_id)
        return 0
    if not messages:
        error = getattr(lark_bot, "last_list_chat_messages_error", None)
        if error:
            logger.warning("chat_context sync empty: chat_id=%s error=%s", chat_id, error)
        return 0

    existing_ids = {
        str(event.get("payload", {}).get("message_id") or "")
        for event in iter_user_messages_for_chat(chat_id, time.time() - 24 * 60 * 60)
    }
    synced = 0
    for item in messages[-50:]:
        message_id = str(item.get("message_id") or "")
        if not message_id or message_id in existing_ids:
            continue
        if bool(item.get("deleted")):
            continue
        if str(item.get("sender_type") or "") != "user":
            continue
        if str(item.get("msg_type") or "") not in {"text", "post"}:
            continue
        mentions = list(item.get("mentions") or [])
        text = _replace_mentions(_extract_text_content(str(item.get("content") or "")), mentions).strip()
        if not text:
            continue
        record_event(
            chat_id,
            "feishu_history",
            "user_message",
            {
                "text": text,
                "chat_id": chat_id,
                "message_id": message_id,
                "source_open_id": str(item.get("sender_id") or ""),
                "user_id": str(item.get("sender_id") or ""),
                "speaker_open_id": str(item.get("sender_id") or ""),
                "mentions": mentions,
                "is_bot_request": message_id == current_message_id or _mentions_bot(text, mentions),
            },
        )
        existing_ids.add(message_id)
        synced += 1
    logger.info("chat_context synced: chat_id=%s count=%s", chat_id, synced)
    return synced


def _chat_context_lookback_seconds() -> int:
    raw = os.getenv("LARK_CHAT_CONTEXT_LOOKBACK_SECONDS", "3600")
    return int(raw) if raw.isdigit() else 3600


def on_message_receive(data: P2ImMessageReceiveV1, lark_bot: LarkBot) -> None:
    """Handle ``im.message.receive_v1`` events from Feishu."""
    logger.debug("Message event received")
    if data.event is None or data.event.message is None:
        logger.error("Invalid message event: missing event or message")
        return

    message = data.event.message
    content_raw = message.content or "{}"
    msg_type = message.message_type or ""

    chat_id = message.chat_id or ""
    message_id = message.message_id or ""
    chat_type = message.chat_type or ""
    mentions = _mention_dicts(message)
    open_id = ""
    sender_type = ""
    if data.event.sender and data.event.sender.sender_id:
        open_id = data.event.sender.sender_id.open_id or ""
        sender_type = data.event.sender.sender_type or ""

    if msg_type == "audio":
        try:
            audio_content = json.loads(content_raw)
            file_key = audio_content.get("file_key", "")
        except (json.JSONDecodeError, AttributeError):
            file_key = ""
        if not file_key:
            logger.warning("audio message missing file_key: message_id=%s", message_id)
            return
        if not _mark_message_processing(message_id):
            logger.info("Duplicate message ignored: message_id=%s", message_id)
            return
        if not chat_id:
            logger.error("Missing chat_id in audio message event")
            return
        worker = threading.Thread(
            target=_process_audio_message,
            args=(file_key, message_id, chat_id, lark_bot, open_id),
            kwargs={"chat_type": chat_type, "mentions": mentions, "sender_type": sender_type},
            daemon=True,
        )
        worker.start()
        logger.debug("Audio message worker started: message_id=%s", message_id)
        return

    text = _replace_mentions(_extract_text_content(content_raw), mentions)
    logger.debug("Message parsed: chat_id=%s message_id=%s open_id=%s text_len=%s", chat_id, message_id, open_id, len(text))

    if not _mark_message_processing(message_id):
        logger.info("Duplicate message ignored: message_id=%s", message_id)
        return

    if not chat_id:
        logger.error("Missing chat_id in message event")
        return

    is_group = (chat_type or "").lower() not in {"", "p2p", "private"}
    if not is_group or _mentions_bot(text, mentions):
        ack_worker = threading.Thread(
            target=_ack_message,
            args=(lark_bot, message_id),
            daemon=True,
        )
        ack_worker.start()

    worker = threading.Thread(
        target=_process_message,
        args=(text, chat_id, message_id, lark_bot, open_id),
        kwargs={"chat_type": chat_type, "mentions": mentions, "sender_type": sender_type},
        daemon=True,
    )
    worker.start()
    logger.debug("Message worker started: chat_id=%s message_id=%s worker=%s", chat_id, message_id, worker.name)


def _resume_card_action(
    thread_id: str,
    decision: Any,
    lark_bot: LarkBot,
) -> None:
    logger.debug("Resume ignored: thread_id=%s decision=%s", thread_id, decision)
    session = session_manager.get_session(thread_id)
    chat_id = session.get("chat_id", thread_id) if session else thread_id
    lark_bot.send_text(chat_id, "当前无待处理任务。")
    session_manager.delete_session(thread_id)


def _handle_group_meeting_card_action(
    *,
    action: str,
    board_item_id: int,
    operator_open_id: str,
    lark_bot: LarkBot,
) -> str:
    from im_copilot.memory.group_board_store import group_board_store
    from im_copilot.skills.lark_calendar import create_calendar_event

    item = group_board_store.get(board_item_id)
    if item is None:
        return "会议事项不存在或已过期。"
    if action == "ignore_group_meeting_event":
        group_board_store.update_status(board_item_id, "deleted")
        return "已忽略。"
    if action != "create_group_meeting_event":
        return "未知会议操作。"
    if item.status == "confirmed":
        return "该会议日程已创建。"
    if not operator_open_id:
        return "无法识别当前操作人。"

    metadata = _json_dict(item.metadata_json)
    start = str(metadata.get("start") or item.due_at or "")
    end = str(metadata.get("end") or "")
    if not start or not end:
        return "缺少明确会议时间，请在群里 @ 我并补充时间后再创建日程。"

    uat = token_store.get(operator_open_id) or ""
    if not uat:
        lark_bot.send_text_to_open_id(operator_open_id, "创建日程需要先完成授权，请在单聊里发送任意消息并按提示授权。")
        return "需要先授权。"

    attendee_ids = [
        attendee_id
        for attendee_id in _json_list(metadata.get("recipients"))
        if attendee_id != operator_open_id
    ]
    result = create_calendar_event(
        summary=_calendar_summary(item.title),
        start=start,
        end=end,
        attendee_ids=attendee_ids,
        description=f"来源群聊：{item.source_text}",
        user_access_token=uat,
    )
    if result.get("status") != "created":
        return f"创建失败：{result.get('error') or '未知错误'}"

    metadata.update({
        "calendar_event_token": result.get("token", ""),
        "calendar_event_url": result.get("url", ""),
        "created_by": operator_open_id,
    })
    group_board_store.update_status(board_item_id, "confirmed", metadata_json=json.dumps(metadata, ensure_ascii=False))
    record_event(
        item.chat_id,
        "feishu",
        "calendar_event_created",
        {
            "board_item_id": board_item_id,
            "title": item.title,
            "start": start,
            "end": end,
            "attendee_ids": attendee_ids,
            "token": result.get("token", ""),
            "url": result.get("url", ""),
        },
    )
    url = result.get("url") or ""
    return f"日程已创建，并包含飞书视频会议。{url}".strip()


def _json_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _calendar_summary(title: str) -> str:
    cleaned = re.sub(r"(今天|今日|明天|明日|后天|\d{1,2}月\d{1,2}日)(早上|上午|中午|下午|晚上|晚间)?(\d{1,2}点)?", "", title)
    cleaned = cleaned.strip(" ，,。")
    return cleaned[:60] or "会议"


def _handle_todo_confirm_action(
    *,
    action: str,
    todo_id: int,
    lark_bot: LarkBot,
    operator_open_id: str = "",
) -> "P2CardActionTriggerResponse":
    from im_copilot.memory.todo_store import todo_store
    record = todo_store.get_by_id(todo_id)
    if record is None:
        return _make_card_response("待办不存在或已处理。")
    if record.status != "awaiting_confirmation":
        return _make_card_response("该待办已处理。")
    if operator_open_id and record.assignee_open_id and operator_open_id != record.assignee_open_id:
        logger.warning(
            "todo_confirm unauthorized: todo_id=%s operator=%s assignee=%s",
            todo_id,
            operator_open_id,
            record.assignee_open_id,
        )
        return _make_card_response("无权操作此待办。")
    if action == "confirm_todo":
        todo_store.update_status(todo_id, "pending")
        logger.info("todo_confirm confirmed: todo_id=%s", todo_id)
        return _make_card_response(f"已确认「{record.title}」，将在到期前提醒你。")
    else:
        todo_store.update_status(todo_id, "deleted")
        logger.info("todo_confirm rejected: todo_id=%s", todo_id)
        return _make_card_response("已忽略。")


def on_card_action(
    data: P2CardActionTrigger,
    lark_bot: LarkBot,
) -> P2CardActionTriggerResponse:
    """Handle card action trigger events from Feishu."""
    logger.debug("Card action event received")
    if data.event is None:
        logger.error("Invalid card action event: missing event")
        return _make_card_response("无效的操作事件")

    action_value = (data.event.action.value or {}) if data.event.action else {}
    user_action = action_value.get("action", "")
    logger.info("Card action received: action=%s value=%s", user_action, action_value)

    if user_action in {"create_group_meeting_event", "ignore_group_meeting_event"}:
        board_item_id = action_value.get("board_item_id")
        if not isinstance(board_item_id, int):
            try:
                board_item_id = int(str(board_item_id))
            except (TypeError, ValueError):
                return _make_card_response("无法识别会议事项")
        operator_open_id = ""
        if data.event.operator:
            operator_open_id = data.event.operator.open_id or ""
        message = _handle_group_meeting_card_action(
            action=user_action,
            board_item_id=board_item_id,
            operator_open_id=operator_open_id,
            lark_bot=lark_bot,
        )
        return _make_card_response(message)

    if user_action in {"confirm_todo", "reject_todo"}:
        todo_id = action_value.get("todo_id")
        if not isinstance(todo_id, int):
            try:
                todo_id = int(str(todo_id))
            except (TypeError, ValueError):
                return _make_card_response("无法识别待办 ID")
        operator_open_id = ""
        if data.event.operator:
            operator_open_id = data.event.operator.open_id or ""
        return _handle_todo_confirm_action(
            action=user_action,
            todo_id=todo_id,
            lark_bot=lark_bot,
            operator_open_id=operator_open_id,
        )

    context = data.event.context
    thread_id = action_value.get("thread_id") or (context.open_chat_id if context else "")
    if not thread_id:
        logger.error("Cannot resolve thread_id from card action event")
        return _make_card_response("无法识别会话")

    logger.debug("Card action context: thread_id=%s action=%s", thread_id, user_action)
    session = session_manager.get_session(thread_id)
    if session is None:
        logger.warning("No active session for thread_id=%s", thread_id)
        return _make_card_response("会话已过期，请重新发起请求。")

    if user_action == "approve":
        decision = {"approved": True, "feedback": "用户同意执行计划"}
    elif user_action == "reject":
        decision = {"approved": False, "feedback": "用户拒绝执行计划"}
    elif user_action == "clarify":
        last_interrupt = session.get("last_interrupt", {})
        questions = last_interrupt.get("questions", [])
        form_value = getattr(data.event.action, "form_value", {}) if data.event.action else {}
        answers = [
            str((form_value or {}).get(f"answer_{i}", ""))
            for i in range(len(questions))
        ]
        decision = answers or action_value.get("answers", [])
    else:
        logger.warning("Unknown card action: %s", user_action)
        return _make_card_response("未知操作")

    logger.debug("Card action decision built: thread_id=%s decision=%s", thread_id, decision)
    worker = threading.Thread(
        target=_resume_card_action,
        args=(thread_id, decision, lark_bot),
        daemon=True,
    )
    worker.start()
    logger.debug("Card action worker started: thread_id=%s worker=%s", thread_id, worker.name)
    return _make_card_response("已收到您的反馈")


def on_message_read(data: P2ImMessageMessageReadV1) -> None:
    """Ignore Feishu message read events."""
    logger.debug("Message read event ignored")
    return None


def on_message_reaction_created(data: P2ImMessageReactionCreatedV1) -> None:
    """Ignore Feishu message reaction created events."""
    logger.debug("Message reaction created event ignored")
    return None


def on_message_reaction_deleted(data: P2ImMessageReactionDeletedV1) -> None:
    """Ignore Feishu message reaction deleted events."""
    logger.debug("Message reaction deleted event ignored")
    return None


def on_bot_added_to_chat(data: P2ImChatMemberBotAddedV1) -> None:
    """Record group chats when the bot is added."""
    if data.event is None:
        logger.error("Invalid bot-added event: missing event")
        return
    chat_id = data.event.chat_id or ""
    if not chat_id:
        logger.error("Invalid bot-added event: missing chat_id")
        return
    from im_copilot.memory.group_history_worker import record_bot_joined_group

    record_bot_joined_group(
        chat_id,
        name=data.event.name or "",
        external=bool(data.event.external),
        tenant_key=data.event.operator_tenant_key or "",
    )


def build_event_handler(lark_bot: LarkBot) -> lark.EventDispatcherHandler:
    """Build and return a ``lark.EventDispatcherHandler`` with both handlers
    registered.
    """
    logger.debug("Building Lark event handler")
    global _reminder_started, _runtime_bot_open_id
    if not _runtime_bot_open_id:
        _runtime_bot_open_id = lark_bot.get_bot_open_id()
        logger.info("lark_bot identity loaded has_open_id=%s", bool(_runtime_bot_open_id))
    with _reminder_lock:
        if not _reminder_started:
            from im_copilot.memory.reminder_worker import start_reminder_loop

            start_reminder_loop(lark_bot)
            _reminder_started = True
    return (
        lark.EventDispatcherHandler.builder(
            lark_bot._encrypt_key,
            lark_bot._verification_token,
        )
        .register_p2_im_message_receive_v1(
            partial(on_message_receive, lark_bot=lark_bot)
        )
        .register_p2_im_message_message_read_v1(on_message_read)
        .register_p2_im_message_reaction_created_v1(on_message_reaction_created)
        .register_p2_im_message_reaction_deleted_v1(on_message_reaction_deleted)
        .register_p2_im_chat_member_bot_added_v1(on_bot_added_to_chat)
        .register_p2_card_action_trigger(
            partial(on_card_action, lark_bot=lark_bot)
        )
        .build()
    )
