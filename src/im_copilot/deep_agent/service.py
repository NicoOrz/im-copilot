from __future__ import annotations

import logging
import re
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from im_copilot.deep_agent.agent import build_agent
from im_copilot.deep_agent.doc_agent import (
    generate_doc_content,
)
from im_copilot.deep_agent.events import iter_user_messages_for_chat, list_events, record_event
from im_copilot.llm import get_llm_for_node, invoke_structured
from im_copilot.skills.lark_doc import (
    create_doc_from_content,
    extract_docx_xml_fields,
    fetch_doc_content,
    summarize_docx_xml_content,
)
from im_copilot.skills.lark_whiteboard import (
    create_whiteboard_from_mermaid,
    generate_whiteboard_mermaid,
)
from im_copilot.skills.lark_slide import create_slide_from_xml, generate_slide_xml

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """判断用户当前请求对应的业务类型。

可选 route：
- chat: 普通聊天、问答、说明、解释，不需要创建任何产物。
- doc: 只需要创建或更新文字型文档产物。
- whiteboard: 只需要创建或更新白板、流程图、思维导图等可视化产物。
- slide: 只需要创建或更新 PPT、幻灯片、演示稿产物。
- multi: 同时需要两种或三种产物。

判断规则：
- 不要依赖固定关键词；根据语义判断用户真实目的。
- 用户提供材料并要求整理、总结、形成 PRD、报告、纪要、方案、说明书等可交付文字产物，通常选择 doc。
- 用户要求图、流程、结构关系、白板表达，通常选择 whiteboard。
- 用户要求 PPT、汇报材料、演示稿，通常选择 slide。
- 用户同时要求文档、白板、PPT 中的至少两类产物，选择 multi。
- 用户只是打招呼、追问、闲聊、询问信息，选择 chat。
- required_artifacts 必须列出需要创建的产物类型，chat 时为空列表。
- 文档产物固定使用 DocxXML，doc_format 必须为 xml。

用户当前消息：
{message}

近几轮历史：
{history}

最近任务上下文：
{task_context}

额外规则：
- 如果当前消息只是极短追问、确认、催促、补充，且最近任务上下文显示上一任务还没有完成，就沿用上一任务的 route，不要改判为 chat。
- 如果最近任务上下文显示上一任务已经完成，且当前消息没有新的创作或修改意图，再判 chat。
"""


class RouteDecision(BaseModel):
    route: Literal["chat", "doc", "whiteboard", "slide", "multi"] = Field(description="业务类型")
    required_artifacts: list[Literal["doc", "whiteboard", "slide"]] = Field(default_factory=list)
    doc_format: Literal["xml"] = Field(default="xml", description="文档格式，固定为 DocxXML")
    reason: str = Field(default="", description="简短判断依据")


@dataclass
class Artifact:
    kind: str
    title: str
    status: Literal["draft", "created", "error"]
    preview: str = ""
    token: str = ""
    url: str = ""


@dataclass
class AgentResult:
    status: str
    summary: str = ""
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_agent(
    message: str,
    *,
    thread_id: str,
    source: str,
    chat_id: str = "",
    message_id: str = "",
    user_id: str = "",
    user_access_token: str = "",
) -> AgentResult:
    artifacts: dict[str, dict[str, Any]] = {}
    message_preview = message.replace("\n", "\\n")[:200]
    logger.info(
        "run_agent start thread_id=%s source=%s chat_id=%s message_id=%s user_id=%s message_len=%s has_user_token=%s preview=%r",
        thread_id,
        source,
        chat_id,
        message_id,
        user_id,
        len(message),
        bool(user_access_token),
        message_preview,
    )
    record_event(
        thread_id,
        source,
        "user_message",
        {
            "text": message,
            "chat_id": chat_id,
            "message_id": message_id,
            "user_id": user_id,
            "source_open_id": user_id,
        },
    )
    record_event(thread_id, source, "agent_stage", {"stage": "start"})

    try:
        route = _route_message(message, thread_id)
        route.doc_format = "xml"
        record_event(
            thread_id,
            source,
            "agent_stage",
            {
                "stage": "route_decision",
                "route": route.route,
                "required_artifacts": route.required_artifacts,
                "doc_format": route.doc_format,
                "reason": route.reason,
            },
        )
        logger.info(
            "run_agent route_decision thread_id=%s route=%s required_artifacts=%s doc_format=%s reason=%s",
            thread_id,
            route.route,
            route.required_artifacts,
            route.doc_format,
            route.reason,
        )
        if route.route != "chat":
            deterministic_result = _run_deterministic_artifacts(
                message=message,
                thread_id=thread_id,
                source=source,
                chat_id=chat_id,
                user_access_token=user_access_token,
                route=route,
                artifacts=artifacts,
            )
            if deterministic_result:
                return deterministic_result
        logger.info(
            "run_agent main_agent selected thread_id=%s source=%s route=%s required_artifacts=%s",
            thread_id,
            source,
            route.route,
            route.required_artifacts,
        )
        agent = build_agent(
            thread_id=thread_id,
            source=source,
            user_access_token=user_access_token,
            artifacts=artifacts,
        )
        history = _message_history_with_route(thread_id, message, route)
        logger.info(
            "run_agent main_agent invoke_start thread_id=%s history_messages=%s",
            thread_id,
            len(history),
        )
        result = agent.invoke(
            {"messages": history},
            config={"configurable": {"thread_id": thread_id}},
        )
        logger.info(
            "run_agent main_agent invoke_end thread_id=%s result_type=%s artifact_keys=%s",
            thread_id,
            type(result).__name__,
            sorted(artifacts.keys()),
        )
        summary = _extract_summary(result)
        if not summary:
            summary = _artifact_summary(artifacts) or "已完成。"
        logger.info(
            "run_agent complete thread_id=%s status=complete summary_len=%s artifact_count=%s artifact_keys=%s",
            thread_id,
            len(summary),
            len(artifacts),
            sorted(artifacts.keys()),
        )
        record_event(thread_id, source, "summary_created", {"summary": summary})
        record_event(thread_id, source, "assistant_message", {"summary": summary})
        return AgentResult(
            status="complete",
            summary=summary,
            artifacts=artifacts,
            events=list_events(thread_id),
        )
    except Exception as exc:
        error = str(exc)
        logger.exception(
            "run_agent failed thread_id=%s source=%s artifact_keys=%s error=%s",
            thread_id,
            source,
            sorted(artifacts.keys()),
            error,
        )
        record_event(thread_id, source, "error", {"error": error})
        return AgentResult(
            status="error",
            artifacts=artifacts,
            events=list_events(thread_id),
            error=error,
        )


def _extract_summary(result: Any) -> str:
    if isinstance(result, dict):
        if isinstance(result.get("summary"), str):
            return result["summary"].strip()
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            for message in reversed(messages):
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                text = _content_to_text(content)
                if text:
                    return text
    return _content_to_text(getattr(result, "content", ""))


def _normalize_route_decision(decision: RouteDecision) -> RouteDecision:
    if decision.route == "chat":
        decision.required_artifacts = []
    elif decision.route == "doc" and not decision.required_artifacts:
        decision.required_artifacts = ["doc"]
    elif decision.route == "whiteboard" and not decision.required_artifacts:
        decision.required_artifacts = ["whiteboard"]
    elif decision.route == "slide" and not decision.required_artifacts:
        decision.required_artifacts = ["slide"]
    elif decision.route == "multi":
        if not decision.required_artifacts:
            decision.required_artifacts = ["doc", "whiteboard", "slide"]
        else:
            order = ["doc", "whiteboard", "slide"]
            decision.required_artifacts = [item for item in order if item in decision.required_artifacts]
    decision.doc_format = "xml"
    return decision


def _parse_route_payload(content: str) -> RouteDecision | None:
    text = content.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        decision = RouteDecision(
            route=str(payload.get("route", "chat")),
            required_artifacts=[
                item for item in payload.get("required_artifacts", [])
                if item in {"doc", "whiteboard", "slide"}
            ],
            doc_format="xml",
            reason=str(payload.get("reason", "")),
        )
    except Exception:
        return None
    return _normalize_route_decision(decision)


def _run_deterministic_artifacts(
    *,
    message: str,
    thread_id: str,
    source: str,
    chat_id: str,
    user_access_token: str,
    route: RouteDecision,
    artifacts: dict[str, dict[str, Any]],
) -> AgentResult | None:
    steps = _artifact_steps(route)
    if not steps:
        return None

    linked_doc_context = _fetch_linked_doc_context(
        message,
        thread_id=thread_id,
        chat_id=chat_id,
        user_access_token=user_access_token,
    )
    for step in steps:
        context = _generation_context(
            thread_id,
            message=message,
            linked_doc_context=linked_doc_context,
        )
        if step == "doc":
            logger.info(
                "run_agent doc_generator selected thread_id=%s source=%s doc_format=xml",
                thread_id,
                source,
            )
            record_event(thread_id, source, "agent_stage", {"stage": "doc_agent"})
            logger.info(
                "run_agent doc_generator invoke_start thread_id=%s message_len=%s context_len=%s",
                thread_id,
                len(message),
                len(context),
            )
            doc_message = _message_with_context(message, context)
            content = generate_doc_content(doc_message)
            missing_requirements = _missing_docx_xml_requirements(context, content)
            if missing_requirements:
                logger.warning(
                    "run_agent doc_generator missing_requirements thread_id=%s requirements=%s",
                    thread_id,
                    missing_requirements,
                )
                content = generate_doc_content(_doc_rewrite_message(doc_message, content, missing_requirements))
            logger.info(
                "run_agent doc_generator invoke_end thread_id=%s content_len=%s",
                thread_id,
                len(content),
            )
            artifact = create_doc_from_content(
                title=_doc_title_from_content(content) or "文档",
                content=content,
                user_access_token=user_access_token,
                doc_format="xml",
            )
            artifacts["doc"] = dict(artifact)
            record_event(thread_id, source, "artifact_created", {"kind": "doc", "artifact": dict(artifact)})
            logger.info(
                "run_agent doc_artifact_created thread_id=%s status=%s token=%s url=%s preview_len=%s",
                thread_id,
                artifact.get("status"),
                artifact.get("token"),
                artifact.get("url"),
                len(str(artifact.get("preview") or "")),
            )
            failed_error = "飞书文档创建失败。"
        elif step == "whiteboard":
            logger.info(
                "run_agent whiteboard_generator selected thread_id=%s source=%s",
                thread_id,
                source,
            )
            record_event(thread_id, source, "agent_stage", {"stage": "whiteboard"})
            parent_doc = artifacts.get("doc") or _latest_artifact(thread_id, "doc")
            logger.info(
                "run_agent whiteboard_generator invoke_start thread_id=%s message_len=%s context_len=%s parent_doc_token=%s",
                thread_id,
                len(message),
                len(context),
                parent_doc.get("token", ""),
            )
            mermaid = generate_whiteboard_mermaid(message, context=context)
            logger.info(
                "run_agent whiteboard_generator invoke_end thread_id=%s mermaid_len=%s",
                thread_id,
                len(mermaid),
            )
            artifact = create_whiteboard_from_mermaid(
                title=_whiteboard_title(message),
                mermaid=mermaid,
                user_access_token=user_access_token,
                parent_doc_token=str(parent_doc.get("token") or ""),
                parent_doc_url=str(parent_doc.get("url") or ""),
            )
            artifacts["whiteboard"] = dict(artifact)
            record_event(thread_id, source, "artifact_created", {"kind": "whiteboard", "artifact": dict(artifact)})
            logger.info(
                "run_agent whiteboard_artifact_created thread_id=%s status=%s token=%s url=%s preview_len=%s",
                thread_id,
                artifact.get("status"),
                artifact.get("token"),
                artifact.get("url"),
                len(str(artifact.get("preview") or "")),
            )
            failed_error = "飞书画板创建失败。"
        elif step == "slide":
            logger.info(
                "run_agent slide_generator selected thread_id=%s source=%s",
                thread_id,
                source,
            )
            record_event(thread_id, source, "agent_stage", {"stage": "slide"})
            logger.info(
                "run_agent slide_generator invoke_start thread_id=%s message_len=%s context_len=%s",
                thread_id,
                len(message),
                len(context),
            )
            slides_xml = generate_slide_xml(message, context=context)
            logger.info(
                "run_agent slide_generator invoke_end thread_id=%s slides_xml_len=%s",
                thread_id,
                len(slides_xml),
            )
            artifact = create_slide_from_xml(
                title=_slide_title(message),
                slides_xml=slides_xml,
                user_access_token=user_access_token,
            )
            if artifact.get("status") != "created":
                retry_error = str(artifact.get("error") or "")
                logger.info(
                    "run_agent slide_generator retry_start thread_id=%s error=%s",
                    thread_id,
                    retry_error[:300],
                )
                slides_xml = generate_slide_xml(
                    message,
                    context=context,
                    previous_xml=slides_xml,
                    error=retry_error,
                )
                logger.info(
                    "run_agent slide_generator retry_end thread_id=%s slides_xml_len=%s",
                    thread_id,
                    len(slides_xml),
                )
                artifact = create_slide_from_xml(
                    title=_slide_title(message),
                    slides_xml=slides_xml,
                    user_access_token=user_access_token,
                )
            artifacts["slide"] = dict(artifact)
            record_event(thread_id, source, "artifact_created", {"kind": "slide", "artifact": dict(artifact)})
            logger.info(
                "run_agent slide_artifact_created thread_id=%s status=%s token=%s url=%s preview_len=%s",
                thread_id,
                artifact.get("status"),
                artifact.get("token"),
                artifact.get("url"),
                len(str(artifact.get("preview") or "")),
            )
            failed_error = "飞书 PPT 创建失败。"
        else:
            continue

        if artifact.get("status") == "error" or (
            user_access_token and artifact.get("status") != "created"
        ):
            record_event(
                thread_id,
                source,
                "error",
                {"error": f"{step} artifact creation failed", "artifact": dict(artifact)},
            )
            logger.warning(
                "run_agent %s artifact creation failed thread_id=%s artifact=%s",
                step,
                thread_id,
                artifact,
            )
            return AgentResult(
                status="error",
                summary=_artifact_summary(artifacts),
                artifacts=artifacts,
                events=list_events(thread_id),
                error=failed_error,
            )

    summary = _artifact_summary(artifacts) or "已完成。"
    logger.info(
        "run_agent complete thread_id=%s status=complete summary_len=%s artifact_count=%s artifact_keys=%s",
        thread_id,
        len(summary),
        len(artifacts),
        sorted(artifacts.keys()),
    )
    record_event(thread_id, source, "summary_created", {"summary": summary})
    record_event(thread_id, source, "assistant_message", {"summary": summary})
    return AgentResult(
        status="complete",
        summary=summary,
        artifacts=artifacts,
        events=list_events(thread_id),
    )


def _artifact_steps(route: RouteDecision) -> list[str]:
    if route.route == "chat":
        return []
    if route.route == "multi":
        desired = route.required_artifacts or ["doc", "whiteboard", "slide"]
    else:
        desired = [route.route]
    order = ["doc", "whiteboard", "slide"]
    return [item for item in order if item in desired]


def _fetch_linked_doc_context(
    message: str,
    *,
    thread_id: str,
    chat_id: str,
    user_access_token: str,
) -> str:
    if not user_access_token:
        logger.info("linked_doc_context skipped thread_id=%s reason=no_user_token", thread_id)
        return ""
    refs = _doc_refs_from_recent_messages(thread_id, chat_id, message)
    logger.info("linked_doc_context refs thread_id=%s count=%s refs=%s", thread_id, len(refs), refs[:3])
    if not refs:
        return ""

    parts: list[str] = [
        "生成约束：用户聊天中的飞书文档链接候选（生成相关链接时必须优先使用）：\n"
        + "\n".join(f"- {ref}" for ref in refs[:4])
    ]
    fetched_refs: list[dict[str, Any]] = []
    for ref in refs[:4]:
        content = fetch_doc_content(ref, user_access_token=user_access_token, doc_format="xml")
        logger.info(
            "linked_doc_context fetched thread_id=%s ref=%s content_len=%s",
            thread_id,
            ref,
            len(content),
        )
        if content:
            fields = extract_docx_xml_fields(content)
            summary = summarize_docx_xml_content(content)
            score = _docx_xml_richness_score(fields)
            logger.info(
                "linked_doc_context summarized thread_id=%s ref=%s score=%s summary_len=%s",
                thread_id,
                ref,
                score,
                len(summary),
            )
            fetched_refs.append({
                "ref": ref,
                "content": content,
                "summary": summary,
                "score": score,
            })
    fetched_refs.sort(key=lambda item: int(item["score"]), reverse=True)
    for item in fetched_refs[:2]:
        parts.append(
            f"引用文档：{item['ref']}\n"
            f"结构化重点：\n{item['summary'] or '未提取到结构化重点'}"
        )
    context = "\n\n".join(parts)
    logger.info("linked_doc_context built thread_id=%s context_len=%s", thread_id, len(context))
    return context


def _doc_refs_from_recent_messages(thread_id: str, chat_id: str, message: str) -> list[str]:
    refs: list[str] = []
    for ref in _doc_refs_from_message(message):
        if ref not in refs:
            refs.append(ref)
    for event in reversed(list_events(thread_id)[-12:]):
        if event.get("event_type") != "user_message":
            continue
        text = str(event.get("payload", {}).get("text") or "")
        for ref in _doc_refs_from_message(text):
            if ref not in refs:
                refs.append(ref)
    if chat_id:
        since_ts = time.time() - 7 * 24 * 60 * 60
        for event in reversed(iter_user_messages_for_chat(chat_id, since_ts)[-30:]):
            text = str(event.get("payload", {}).get("text") or "")
            for ref in _doc_refs_from_message(text):
                if ref not in refs:
                    refs.append(ref)
    return refs


def _docx_xml_richness_score(fields: dict[str, Any]) -> int:
    return (
        len(fields.get("cite_users") or []) * 3
        + len(fields.get("whiteboards") or []) * 6
        + len(fields.get("images") or []) * 2
        + len(fields.get("checkboxes") or []) * 2
        + len(fields.get("links") or []) * 2
        + len(fields.get("grids") or []) * 5
        + len(fields.get("headings") or [])
    )


def _doc_refs_from_message(message: str) -> list[str]:
    refs: list[str] = []
    pattern = re.compile(r"https?://[^\s<>()\"']+/(?:docx|wiki)/[A-Za-z0-9_-]+")
    for match in pattern.findall(message):
        ref = match.strip().rstrip("。.,，；;！？】])》")
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _generation_context(
    thread_id: str,
    *,
    message: str,
    linked_doc_context: str = "",
) -> str:
    parts: list[str] = []
    recent = _recent_context(thread_id, exclude_user_text=message)
    if linked_doc_context:
        parts.append(linked_doc_context)
    if recent:
        parts.append(_truncate_context(recent, 6000))
    return "\n\n".join(parts)


def _message_with_context(message: str, context: str) -> str:
    if not context.strip():
        return message
    return f"{message}\n\n生成约束上下文（必须遵守）：\n{context.strip()}"


def _missing_docx_xml_requirements(context: str, content: str) -> list[str]:
    if not context.strip():
        return []

    missing: list[str] = []
    if _json_array_has_item(context, "cite_users") and "<cite type=\"user\"" not in content:
        missing.append("缺少 <cite type=\"user\">")
    if _json_array_has_item(context, "whiteboards") and "<whiteboard" not in content:
        missing.append("缺少 <whiteboard>")
    if _json_array_has_item(context, "grids") and "<grid" not in content:
        missing.append("缺少 <grid>")
    if _json_array_has_item(context, "checkboxes") and "<checkbox" not in content:
        missing.append("缺少 <checkbox>")
    has_links = "飞书文档链接候选" in context or _json_array_has_item(context, "links")
    has_link_tag = "<a " in content or "<bookmark" in content or "<cite type=\"doc\"" in content
    if has_links and not has_link_tag:
        missing.append("缺少链接标签")
    return missing


def _json_array_has_item(text: str, key: str) -> bool:
    return bool(re.search(rf'"{re.escape(key)}"\s*:\s*\[\s*{{', text))


def _doc_rewrite_message(message: str, previous_content: str, missing_requirements: list[str]) -> str:
    return (
        f"{message}\n\n"
        "上一次生成的 DocxXML 片段：\n"
        f"{_truncate_context(previous_content, 4000)}\n\n"
        "请重新生成完整 DocxXML，并修正以下缺失项：\n"
        + "\n".join(f"- {item}" for item in missing_requirements)
    )


def _truncate_context(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _route_message(message: str, thread_id: str) -> RouteDecision:
    history = _message_history(thread_id)
    history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-6:])
    recent_task_state = _recent_task_state(thread_id)
    task_context = _format_recent_task_context(recent_task_state)
    prompt = ROUTER_PROMPT.format(
        message=message,
        history=history_text or "（无）",
        task_context=task_context,
    )
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            decision = invoke_structured("deep_agent_router", RouteDecision, prompt)
            decision = _normalize_route_decision(decision)
            decision = _continue_recent_task_if_needed(message, decision, recent_task_state)
            return decision
        except Exception as exc:
            last_error = exc
            logger.warning(
                "run_agent route_decision structured attempt failed thread_id=%s attempt=%s error=%s",
                thread_id,
                attempt + 1,
                exc,
            )
            time.sleep(0.2 * (attempt + 1))

    fallback_prompt = (
        "你只输出一个 JSON 对象，不要输出其他文本。\n"
        "字段必须包含 route、required_artifacts、doc_format、reason。\n"
        "route 只能是 chat、doc、whiteboard、slide、multi。\n"
        "required_artifacts 只能包含 doc、whiteboard、slide。\n"
        "doc_format 固定为 xml。\n\n"
        f"用户当前消息：\n{message}\n\n"
        f"近几轮历史：\n{history_text or '（无）'}"
    )
    try:
        raw = llm.invoke(fallback_prompt)
        content = _content_to_text(getattr(raw, "content", raw))
        parsed = _parse_route_payload(content)
        if parsed:
            parsed = _continue_recent_task_if_needed(message, parsed, recent_task_state)
            return parsed
    except Exception as exc:
        last_error = exc
        logger.warning(
            "run_agent route_decision fallback failed thread_id=%s error=%s",
            thread_id,
            exc,
        )

    logger.error(
        "run_agent route_decision fallback to chat thread_id=%s error=%s",
        thread_id,
        last_error,
    )
    return RouteDecision(route="chat", required_artifacts=[], doc_format="xml", reason="route fallback")


def _continue_recent_task_if_needed(
    message: str,
    decision: RouteDecision,
    recent_task_state: dict[str, Any],
) -> RouteDecision:
    if decision.route != "chat":
        return decision
    if not recent_task_state or recent_task_state.get("completed"):
        return decision
    if not _is_weak_followup(message):
        return decision

    route = str(recent_task_state.get("route") or "")
    if route not in {"doc", "whiteboard", "slide", "multi"}:
        return decision

    required_artifacts = recent_task_state.get("required_artifacts") or []
    if route == "multi" and not required_artifacts:
        required_artifacts = ["doc", "whiteboard", "slide"]
    updated = RouteDecision(
        route=route,  # type: ignore[arg-type]
        required_artifacts=[
            item for item in required_artifacts
            if item in {"doc", "whiteboard", "slide"}
        ],
        doc_format="xml",
        reason="recent task unfinished; continue previous route",
    )
    return _normalize_route_decision(updated)


def _is_weak_followup(message: str) -> bool:
    stripped = message.strip()
    if not stripped:
        return True
    if len(stripped) <= 2 and not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", stripped):
        return True
    return False


def _recent_task_state(thread_id: str) -> dict[str, Any]:
    events = list_events(thread_id)
    artifacts_after: list[dict[str, Any]] = []
    summary_after = ""
    for event in reversed(events):
        payload = event.get("payload", {})
        event_type = event.get("event_type")
        if event_type == "agent_stage" and payload.get("stage") == "route_decision":
            route = str(payload.get("route") or "")
            required_artifacts = [
                item for item in payload.get("required_artifacts", [])
                if item in {"doc", "whiteboard", "slide"}
            ]
            created = {
                item.get("kind")
                for item in artifacts_after
                if isinstance(item, dict) and item.get("status") == "created"
            }
            completed = bool(required_artifacts) and set(required_artifacts).issubset(created)
            return {
                "route": route,
                "required_artifacts": required_artifacts,
                "completed": completed,
                "artifacts": list(reversed(artifacts_after)),
                "summary": summary_after,
                "reason": str(payload.get("reason") or ""),
            }
        if event_type == "artifact_created":
            artifact = payload.get("artifact")
            if isinstance(artifact, dict):
                artifacts_after.append(dict(artifact))
        elif event_type == "assistant_message" and not summary_after:
            summary_after = str(payload.get("summary") or payload.get("text") or "").strip()
    return {}


def _format_recent_task_context(state: dict[str, Any]) -> str:
    if not state:
        return "（无最近任务）"
    route = str(state.get("route") or "")
    required = ", ".join(state.get("required_artifacts") or []) or "无"
    completed = "已完成" if state.get("completed") else "未完成"
    parts = [
        f"最近任务路由：{route or '未知'}",
        f"需要产物：{required}",
        f"状态：{completed}",
    ]
    artifacts = state.get("artifacts") or []
    if artifacts:
        preview_parts = []
        for artifact in artifacts[:3]:
            if not isinstance(artifact, dict):
                continue
            title = artifact.get("title") or artifact.get("kind") or "产物"
            status = artifact.get("status") or "draft"
            preview_parts.append(f"{artifact.get('kind') or 'artifact'}:{title}:{status}")
        if preview_parts:
            parts.append(f"近期产物：{'；'.join(preview_parts)}")
    summary = str(state.get("summary") or "").strip()
    if summary:
        parts.append(f"最近回复：{summary[:200]}")
    reason = str(state.get("reason") or "").strip()
    if reason:
        parts.append(f"最近判断：{reason[:120]}")
    return "\n".join(parts)


def _doc_title_from_content(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("<"):
        import re

        match = re.search(r"<title>(.*?)</title>", stripped, re.DOTALL | re.IGNORECASE)
        if match:
            return _strip_text(match.group(1))
    for line in stripped.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def _strip_text(text: str) -> str:
    return (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .strip()
    )


def _latest_artifact(thread_id: str, kind: str) -> dict[str, Any]:
    for event in reversed(list_events(thread_id)):
        if event.get("event_type") != "artifact_created":
            continue
        payload = event.get("payload", {})
        artifact = payload.get("artifact")
        if isinstance(artifact, dict) and artifact.get("kind") == kind:
            return artifact
    return {}


def _recent_context(thread_id: str, *, exclude_user_text: str = "") -> str:
    parts: list[str] = []
    for event in list_events(thread_id)[-12:]:
        payload = event.get("payload", {})
        if event.get("event_type") == "user_message":
            text = str(payload.get("text") or "")
            if text and text.strip() != exclude_user_text.strip():
                parts.append(f"用户：{text}")
        elif event.get("event_type") == "artifact_created":
            artifact = payload.get("artifact")
            if isinstance(artifact, dict):
                preview = str(artifact.get("preview") or "")[:3000]
                parts.append(f"产物：{artifact.get('title') or artifact.get('kind')}\n{preview}")
    return "\n\n".join(parts)


def _whiteboard_title(message: str) -> str:
    cleaned = " ".join(message.split())
    if len(cleaned) > 30:
        cleaned = cleaned[:30]
    return f"白板：{cleaned or '会议思维导图'}"


def _slide_title(message: str) -> str:
    cleaned = " ".join(message.split())
    if len(cleaned) > 30:
        cleaned = cleaned[:30]
    return f"PPT：{cleaned or '汇报'}"


def _message_history_with_route(
    thread_id: str,
    message: str,
    route: RouteDecision,
) -> list[dict[str, str]]:
    history = _message_history(thread_id)
    if route.route == "chat":
        return history

    routed_message = (
        f"业务类型：{route.route}\n"
        f"需要创建的产物：{', '.join(route.required_artifacts) or '无'}\n"
        "文档格式：xml\n"
        "请按该业务目标完成任务；如需产物，必须调用对应 create_*_artifact 工具。\n\n"
        "用户原始请求：\n"
        f"{message}"
    )
    for item in reversed(history):
        if item.get("role") == "user":
            item["content"] = routed_message
            return history
    return history + [{"role": "user", "content": routed_message}]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts).strip()
    return ""


def _message_history(thread_id: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for event in list_events(thread_id):
        payload = event.get("payload", {})
        if event.get("event_type") == "user_message":
            text = str(payload.get("text") or payload.get("message") or "").strip()
            if text:
                messages.append({"role": "user", "content": text})
        elif event.get("event_type") == "assistant_message":
            text = str(payload.get("summary") or payload.get("text") or "").strip()
            if text:
                messages.append({"role": "assistant", "content": text})
    return messages[-20:]


def _artifact_summary(artifacts: dict[str, dict[str, Any]]) -> str:
    if not artifacts:
        return ""
    lines = []
    for artifact in artifacts.values():
        title = artifact.get("title") or artifact.get("kind") or "产物"
        status = artifact.get("status") or "draft"
        url = artifact.get("url") or ""
        line = f"{title}：{status}"
        if url:
            line += f" {url}"
        lines.append(line)
    return "\n".join(lines)
