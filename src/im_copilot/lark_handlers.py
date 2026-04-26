"""Event handlers for Feishu (Lark) bot WebSocket events.

This module wires incoming messages and card actions into the LangGraph
pipeline, manages interactive cards, and handles HITL interrupts via the
session manager.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from im_copilot.checkpointer import get_checkpointer
from im_copilot.graph.pipeline import build_pipeline
from im_copilot.lark_bot import LarkBot
from im_copilot.lark_card import (
    create_approval_card,
    create_clarification_card,
    create_progress_card,
    create_result_card,
    update_progress_payload,
)
from im_copilot.session_manager import session_manager

logger = logging.getLogger(__name__)


def _extract_text_content(content: str) -> str:
    """Parse the JSON content string from a Feishu text message."""
    try:
        parsed = json.loads(content)
        return parsed.get("text", "")
    except json.JSONDecodeError:
        return content


def _is_interrupt_update(step: dict[str, Any]) -> bool:
    """Check whether a stream step represents an ``__interrupt__`` update."""
    return "__interrupt__" in step


def _get_interrupt_info(step: dict[str, Any]) -> dict[str, Any] | None:
    """Extract interrupt payload from a stream step, if present."""
    interrupt_data = step.get("__interrupt__")
    if interrupt_data is None:
        return None
    # LangGraph may return a tuple/list of Interrupt objects
    if isinstance(interrupt_data, (list, tuple)) and interrupt_data:
        interrupt_data = interrupt_data[0]
    # Some versions expose the payload directly; others wrap it
    if hasattr(interrupt_data, "value"):
        return interrupt_data.value  # type: ignore[union-attr]
    return interrupt_data  # type: ignore[return-value]


def _update_card_for_interrupt(
    lark_bot: LarkBot,
    session: dict[str, Any],
    interrupt_info: dict[str, Any],
) -> None:
    """Replace the progress card with an approval or clarification card."""
    card_id = session.get("card_id")
    if not card_id:
        logger.warning("No card_id in session; cannot update card for interrupt")
        return

    gate = interrupt_info.get("gate", "")
    chat_id = session["thread_id"]

    if gate == "plan_approval":
        plan = interrupt_info.get("plan", [])
        intent_type = interrupt_info.get("intent_type", "")
        intent_params = interrupt_info.get("intent_params", {})
        card = create_approval_card(
            plan=plan,
            intent_type=intent_type,
            intent_params=intent_params,
        )
    elif gate == "clarification":
        questions = interrupt_info.get("questions", [])
        card = create_clarification_card(questions=questions)
    else:
        logger.warning("Unknown interrupt gate: %s", gate)
        return

    # Replace the card by sending a new interactive card to the same chat.
    # The new card message will have its own message_id which becomes the new card_id.
    resp = lark_bot.send_card(chat_id, card)
    if resp.get("code") == 0 and resp.get("data", {}).get("message_id"):
        new_card_id = resp["data"]["message_id"]
        session_manager.update_session(session["thread_id"], card_id=new_card_id)
    else:
        logger.error("Failed to send interrupt card: %s", resp.get("msg"))


def _update_progress_card(
    lark_bot: LarkBot,
    session: dict[str, Any],
    node_name: str,
    detail: str,
) -> None:
    """Stream-update the progress card with the current node status."""
    card_id = session.get("card_id")
    if not card_id:
        return

    sequence = session.get("sequence", 0) + 1
    session_manager.update_session(session["thread_id"], sequence=sequence)

    payload = update_progress_payload(
        card_id=card_id,
        step=node_name,
        detail=detail,
        sequence=sequence,
    )
    lark_bot.update_card_stream(
        card_id=card_id,
        element_id="progress_md",
        content=payload["card"]["elements"][0]["content"],
        sequence=sequence,
    )
    lark_bot.update_card_stream(
        card_id=card_id,
        element_id="status_div",
        content=payload["card"]["elements"][1]["text"]["content"],
        sequence=sequence + 1,
    )
    session_manager.update_session(session["thread_id"], sequence=sequence + 1)


def _finalize_card(
    lark_bot: LarkBot,
    session: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Send the final result card when the pipeline finishes."""
    chat_id = session["thread_id"]
    summary = state.get("summary", "处理完成")
    artifacts = state.get("artifacts", {})
    # Convert ContentResult TypedDicts to plain dicts for the card template
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
    lark_bot.send_card(chat_id, card)


def on_message_receive(data: dict[str, Any]) -> None:
    """Handle ``im.message.receive_v1`` events from Feishu.

    Extracts the text message, starts the LangGraph pipeline in streaming
    mode, and drives the interactive card lifecycle.
    """
    lark_bot: LarkBot = data.get("_lark_bot")
    if lark_bot is None:
        logger.error("Missing _lark_bot in event data")
        return

    event = data.get("event", {})
    message = event.get("message", {})
    content_raw = message.get("content", "{}")
    text = _extract_text_content(content_raw)

    chat_id = message.get("chat_id", "")
    message_id = message.get("message_id", "")
    chat_type = message.get("chat_type", "")

    if not chat_id:
        logger.error("Missing chat_id in message event")
        return

    thread_id = chat_id
    source = "feishu"

    # Send initial progress card
    progress_card = create_progress_card(title="正在分析您的请求...")
    card_resp = lark_bot.send_card(chat_id, progress_card)
    card_id = None
    if card_resp.get("code") == 0 and card_resp.get("data", {}).get("message_id"):
        card_id = card_resp["data"]["message_id"]

    try:
        with get_checkpointer("sqlite") as checkpointer:
            graph = build_pipeline(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = {
                "raw_message": text,
                "chat_id": chat_id,
                "message_id": message_id,
                "source": source,
                "errors": [],
                "checks": [],
                "reflection_iteration": 0,
            }

            session = session_manager.create_session(
                thread_id=thread_id,
                graph=graph,
                config=config,
                card_id=card_id,
            )

            stream = graph.stream(initial_state, config=config, stream_mode="updates")
            final_state: dict[str, Any] | None = None

            for step in stream:
                if _is_interrupt_update(step):
                    interrupt_info = _get_interrupt_info(step)
                    if interrupt_info:
                        session_manager.update_session(
                            thread_id,
                            last_interrupt=interrupt_info,
                        )
                        _update_card_for_interrupt(lark_bot, session, interrupt_info)
                    break

                # Regular node update
                for node_name, update in step.items():
                    final_state = update if isinstance(update, dict) else {}
                    detail = f"节点 {node_name} 执行中..."
                    _update_progress_card(lark_bot, session, node_name, detail)

            if final_state is not None:
                _finalize_card(lark_bot, session, final_state)
                session_manager.delete_session(thread_id)

    except Exception as exc:
        logger.exception("Pipeline error for thread_id=%s", thread_id)
        lark_bot.send_text(chat_id, f"处理出错：{exc}")
        session_manager.delete_session(thread_id)


def on_card_action(data: dict[str, Any]) -> None:
    """Handle card action trigger events from Feishu.

    Parses the user's button click, retrieves the saved session, resumes the
    pipeline with the user's decision, and continues streaming updates.
    """
    lark_bot: LarkBot = data.get("_lark_bot")
    if lark_bot is None:
        logger.error("Missing _lark_bot in card action data")
        return

    event = data.get("event", {})
    action = event.get("action", {})
    action_value = action.get("value", {})
    user_action = action_value.get("action", "")

    # thread_id is passed in the card callback context
    open_message_id = event.get("open_message_id", "")
    open_chat_id = event.get("open_chat_id", "")
    # Prefer open_chat_id as thread_id; fallback to extracting from context
    thread_id = open_chat_id or open_message_id

    if not thread_id:
        logger.error("Cannot resolve thread_id from card action event")
        return

    session = session_manager.get_session(thread_id)
    if session is None:
        logger.warning("No active session for thread_id=%s", thread_id)
        lark_bot.send_text(thread_id, "会话已过期，请重新发起请求。")
        return

    # Build decision payload based on action type
    if user_action == "approve":
        decision = {"approved": True, "feedback": "用户同意执行计划"}
    elif user_action == "reject":
        decision = {"approved": False, "feedback": "用户拒绝执行计划"}
    elif user_action == "clarify":
        question = action_value.get("question", "")
        # For clarification, the user clicked a specific question button.
        # In a real implementation the bot might ask for free-text input;
        # here we treat the click as acknowledging that question.
        decision = {"answers": [question]}
    else:
        logger.warning("Unknown card action: %s", user_action)
        return

    try:
        # Resume the graph with the user's decision
        result = session_manager.resume_session(thread_id, decision)
        final_state: dict[str, Any] = result if isinstance(result, dict) else {}

        # After resume, the graph may hit another interrupt or finish.
        # We re-stream from the resumed state to catch any further updates.
        graph = session["graph"]
        config = session["config"]
        stream = graph.stream(None, config=config, stream_mode="updates")

        for step in stream:
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
                final_state = update if isinstance(update, dict) else {}
                detail = f"节点 {node_name} 执行中..."
                _update_progress_card(lark_bot, session, node_name, detail)

        _finalize_card(lark_bot, session, final_state)
        session_manager.delete_session(thread_id)

    except Exception as exc:
        logger.exception("Resume error for thread_id=%s", thread_id)
        lark_bot.send_text(thread_id, f"继续处理时出错：{exc}")
        session_manager.delete_session(thread_id)


def build_event_handler(lark_bot: LarkBot) -> Any:
    """Build and return the WebSocket event handler callable.

    The returned callable accepts a raw event dict and dispatches to the
    appropriate handler based on the event type.

    Parameters
    ----------
    lark_bot:
        Configured ``LarkBot`` instance for sending messages and cards.

    Returns
    -------
    Callable
        A function suitable for passing to ``LarkBot.start_ws``.
    """

    def _handler(data: dict[str, Any]) -> None:
        # Inject the bot instance so handlers can use it without globals
        data["_lark_bot"] = lark_bot
        event_type = data.get("header", {}).get("event_type", "")

        if event_type == "im.message.receive_v1":
            on_message_receive(data)
        elif event_type == "card.action.trigger":
            on_card_action(data)
        else:
            logger.debug("Unhandled event type: %s", event_type)

    return _handler
