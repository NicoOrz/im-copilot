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
import uuid
from functools import partial
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1.model.p2_im_message_receive_v1 import P2ImMessageReceiveV1
from lark_oapi.api.im.v1.model.p2_im_message_message_read_v1 import (
    P2ImMessageMessageReadV1,
)
from lark_oapi.event.callback.model.p2_card_action_trigger import (
    CallBackToast,
    P2CardActionTrigger,
    P2CardActionTriggerResponse,
)

from im_copilot.checkpointer import get_checkpointer
from im_copilot.graph.pipeline import build_pipeline
from im_copilot.lark_bot import LarkBot
from im_copilot.lark_card import (
    create_approval_card,
    create_clarification_card,
    create_progress_card,
    create_result_card,
    create_streaming_card,
)
from im_copilot.session_manager import session_manager
from im_copilot.user_token_store import token_store

logger = logging.getLogger(__name__)

_PROCESSED_MESSAGE_TTL_SECONDS = 300
_PERSISTED_MESSAGE_TTL_SECONDS = 7 * 24 * 60 * 60
_processed_messages: dict[str, float] = {}
_processed_messages_lock = threading.Lock()


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
    """Parse the JSON content string from a Feishu text message."""
    try:
        parsed = json.loads(content)
        return parsed.get("text", "")
    except json.JSONDecodeError:
        return content


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
    for k, v in artifacts.items():
        if isinstance(v, dict):
            artifacts_plain[k] = f"{v.get('title', k)} — {v.get('status', 'unknown')}"
        else:
            artifacts_plain[k] = str(v)

    doc_links: list[dict] | None = None
    if artifacts_plain:
        doc_links = [
            {"title": info, "url": "#"}
            for info in artifacts_plain.values()
        ]

    card = create_result_card(
        summary=summary,
        artifacts=artifacts_plain,
        doc_links=doc_links,
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
    scopes = " ".join([
        "offline_access",
        "docx:document",
        "drive:drive",
        "slides:presentation:read",
        "slides:presentation:create",
        "slides:presentation:write_only",
        "slides:presentation:update",
        "wiki:wiki",
    ])
    import urllib.parse
    auth_url = (
        "https://accounts.feishu.cn/open-apis/authen/v1/authorize"
        f"?app_id={urllib.parse.quote(app_id)}"
        f"&redirect_uri={urllib.parse.quote(callback_url)}"
        f"&response_type=code"
        f"&scope={urllib.parse.quote(scopes)}"
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


def _process_message(text: str, chat_id: str, message_id: str, lark_bot: LarkBot, open_id: str = "") -> None:
    from im_copilot.commands import parse_command, execute_command

    clean_text = _AT_MENTION_RE.sub("", text.strip())
    parsed = parse_command(clean_text)
    if parsed is not None:
        cmd_name, cmd_args = parsed
        thread_id = _make_thread_id(chat_id)
        cmd_result = execute_command(cmd_name, cmd_args, chat_id, thread_id, source="feishu")
        if cmd_result.metadata.get("action") == "reset_thread":
            _reset_chat_thread(chat_id)
        lark_bot.reply_text(message_id, cmd_result.response_text)
        return

    thread_id = _make_thread_id(chat_id)
    source = "feishu"
    logger.debug("Process message start: thread_id=%s message_id=%s text_len=%s", thread_id, message_id, len(text))

    # Check user token; prompt OAuth if missing
    user_access_token = ""
    if open_id:
        user_access_token = token_store.get(open_id) or ""
        if not user_access_token:
            _send_oauth_prompt(lark_bot, chat_id, open_id)
            return

    try:
        with get_checkpointer("sqlite") as checkpointer:
            logger.debug("Checkpointer opened: thread_id=%s", thread_id)
            graph = build_pipeline(checkpointer=checkpointer)
            logger.debug("Pipeline built: thread_id=%s", thread_id)
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = {
                "raw_message": text,
                "chat_id": chat_id,
                "message_id": message_id,
                "source": source,
                "user_id": open_id,
                "user_access_token": user_access_token,
                "message_history": [{"role": "user", "content": text}],
                "errors": [],
                "checks": [],
                "reflection_iteration": 0,
            }

            # Stream from START so the graph runs intent_node itself;
            # peek at the first step to fast-path chat replies without cards.
            logger.debug("Graph stream start: thread_id=%s", thread_id)
            stream = graph.stream(
                initial_state,
                config=config,
                stream_mode="updates",
            )
            final_state: dict[str, Any] = dict(initial_state)

            # Peek first step to detect chat intent and reply directly
            try:
                first_step = next(stream)
            except StopIteration:
                first_step = None

            if first_step is None:
                logger.debug("Empty stream, nothing to process: thread_id=%s", thread_id)
                return

            if not _is_interrupt_update(first_step):
                for node_name, update in first_step.items():
                    if isinstance(update, dict):
                        final_state.update(update)
                if final_state.get("intent_type") == "chat":
                    logger.debug("Chat fast-path: thread_id=%s message_id=%s", thread_id, message_id)
                    for step in stream:
                        if _is_interrupt_update(step):
                            interrupt_info = _get_interrupt_info(step)
                            if interrupt_info:
                                session = session_manager.create_session(
                                    thread_id=thread_id,
                                    graph=graph,
                                    config=config,
                                    chat_id=chat_id,
                                )
                                session_manager.update_session(
                                    thread_id,
                                    last_interrupt=interrupt_info,
                                )
                                _update_card_for_interrupt(lark_bot, session, interrupt_info)
                            logger.debug("Process message paused: thread_id=%s message_id=%s", thread_id, message_id)
                            return
                        for node_name, update in step.items():
                            if isinstance(update, dict):
                                final_state.update(update)
                    summary = final_state.get("summary", "")
                    logger.debug("Chat deliver result: thread_id=%s summary_len=%s", thread_id, len(summary))
                    if summary:
                        lark_bot.reply_text(message_id, summary)
                    logger.debug("Process message done: thread_id=%s message_id=%s mode=chat", thread_id, message_id)
                    return

                # Non-chat: create streaming card and continue
                card_entity_id, msg_id = _send_streaming_card(lark_bot, chat_id)
                if not card_entity_id:
                    lark_bot.send_text(chat_id, "暂时无法创建飞书交互卡片，请稍后重试。")
                    return

                logger.debug("Create session start: thread_id=%s card_entity_id=%s message_id=%s", thread_id, card_entity_id, msg_id)
                session = session_manager.create_session(
                    thread_id=thread_id,
                    graph=graph,
                    config=config,
                    card_id=msg_id,
                    chat_id=chat_id,
                )
                session_manager.update_session(
                    thread_id,
                    card_message_id=msg_id,
                    card_entity_id=card_entity_id,
                )

                detail = f"节点 {node_name} 执行中..."
                _update_progress_card(lark_bot, session, node_name, detail)
            elif _is_interrupt_update(first_step):
                interrupt_info = _get_interrupt_info(first_step)
                if interrupt_info:
                    session = session_manager.create_session(
                        thread_id=thread_id,
                        graph=graph,
                        config=config,
                        chat_id=chat_id,
                    )
                    session_manager.update_session(
                        thread_id,
                        last_interrupt=interrupt_info,
                    )
                    _update_card_for_interrupt(lark_bot, session, interrupt_info)
                logger.debug("Process message paused: thread_id=%s message_id=%s", thread_id, message_id)
                return

            for step in stream:
                logger.debug("Graph stream step: thread_id=%s keys=%s", thread_id, list(step.keys()))
                if _is_interrupt_update(step):
                    interrupt_info = _get_interrupt_info(step)
                    if interrupt_info:
                        session_manager.update_session(
                            thread_id,
                            last_interrupt=interrupt_info,
                        )
                        logger.debug("Graph interrupt received: thread_id=%s gate=%s", thread_id, interrupt_info.get("gate"))
                        _update_card_for_interrupt(lark_bot, session, interrupt_info)
                    logger.debug("Process message paused: thread_id=%s message_id=%s", thread_id, message_id)
                    return

                for node_name, update in step.items():
                    if isinstance(update, dict):
                        final_state.update(update)
                    detail = f"节点 {node_name} 执行中..."
                    _update_progress_card(lark_bot, session, node_name, detail)

            logger.debug("Resume stream finished: thread_id=%s final_keys=%s", thread_id, list(final_state.keys()))
            _finalize_card(lark_bot, session, final_state)
            session_manager.delete_session(thread_id)
            logger.debug("Resume worker done: thread_id=%s", thread_id)

    except Exception as exc:
        logger.exception("Pipeline error for thread_id=%s", thread_id)
        lark_bot.send_text(chat_id, f"处理出错：{exc}")
        session_manager.delete_session(thread_id)


def on_message_receive(data: P2ImMessageReceiveV1, lark_bot: LarkBot) -> None:
    """Handle ``im.message.receive_v1`` events from Feishu."""
    logger.debug("Message event received")
    if data.event is None or data.event.message is None:
        logger.error("Invalid message event: missing event or message")
        return

    message = data.event.message
    content_raw = message.content or "{}"
    text = _extract_text_content(content_raw)

    chat_id = message.chat_id or ""
    message_id = message.message_id or ""
    open_id = ""
    if data.event.sender and data.event.sender.sender_id:
        open_id = data.event.sender.sender_id.open_id or ""
    logger.debug("Message parsed: chat_id=%s message_id=%s open_id=%s text_len=%s", chat_id, message_id, open_id, len(text))

    if not _mark_message_processing(message_id):
        logger.info("Duplicate message ignored: message_id=%s", message_id)
        return

    if not chat_id:
        logger.error("Missing chat_id in message event")
        return

    worker = threading.Thread(
        target=_process_message,
        args=(text, chat_id, message_id, lark_bot, open_id),
        daemon=True,
    )
    worker.start()
    logger.debug("Message worker started: chat_id=%s message_id=%s worker=%s", chat_id, message_id, worker.name)


def _resume_card_action(
    thread_id: str,
    decision: Any,
    lark_bot: LarkBot,
) -> None:
    logger.debug("Resume worker start: thread_id=%s decision=%s", thread_id, decision)
    session: dict[str, Any] | None = None
    try:
        session = session_manager.get_session(thread_id)
        if session is None:
            logger.warning("No active session for resume: thread_id=%s", thread_id)
            return

        config = session["config"]

        from langgraph.types import Command

        with get_checkpointer("sqlite") as checkpointer:
            graph = build_pipeline(checkpointer=checkpointer)
            logger.debug("Resume stream start: thread_id=%s", thread_id)
            stream = graph.stream(
                Command(resume=decision), config=config, stream_mode="updates",
            )
            final_state: dict[str, Any] = {}

            for step in stream:
                logger.debug("Resume stream step: thread_id=%s keys=%s", thread_id, list(step.keys()))
                if _is_interrupt_update(step):
                    interrupt_info = _get_interrupt_info(step)
                    if interrupt_info:
                        session_manager.update_session(
                            thread_id,
                            last_interrupt=interrupt_info,
                        )
                        _update_card_for_interrupt(lark_bot, session, interrupt_info)
                    return

                for node_name, update in step.items():
                    if isinstance(update, dict):
                        final_state.update(update)
                    detail = f"节点 {node_name} 执行中..."
                    _update_progress_card(lark_bot, session, node_name, detail)

            logger.debug("Resume stream finished: thread_id=%s final_keys=%s", thread_id, list(final_state.keys()))
            _finalize_card(lark_bot, session, final_state)
            session_manager.delete_session(thread_id)
            logger.debug("Resume worker done: thread_id=%s", thread_id)

    except Exception as exc:
        logger.exception("Resume error for thread_id=%s", thread_id)
        chat_id = session.get("chat_id", thread_id) if session else thread_id
        lark_bot.send_text(chat_id, f"继续处理时出错：{exc}")
        session_manager.delete_session(thread_id)


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


def build_event_handler(lark_bot: LarkBot) -> lark.EventDispatcherHandler:
    """Build and return a ``lark.EventDispatcherHandler`` with both handlers
    registered.
    """
    logger.debug("Building Lark event handler")
    return (
        lark.EventDispatcherHandler.builder(
            lark_bot._encrypt_key,
            lark_bot._verification_token,
        )
        .register_p2_im_message_receive_v1(
            partial(on_message_receive, lark_bot=lark_bot)
        )
        .register_p2_im_message_message_read_v1(on_message_read)
        .register_p2_card_action_trigger(
            partial(on_card_action, lark_bot=lark_bot)
        )
        .build()
    )
