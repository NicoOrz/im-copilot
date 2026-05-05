from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

from im_copilot.lark_cli import run_lark_cli
from im_copilot.llm import get_llm_for_node
from im_copilot.skills.base import SkillArtifact
from im_copilot.skills.config import get_skill_config

logger = logging.getLogger(__name__)

WHITEBOARD_PROMPT = """{system_prompt}

用户原始请求：
{raw_message}

主题：{topic}

图表规则：
{diagram_rules}

标题规则：
{title_rules}

禁止内容：
{forbidden_text}

要求：
- 直接输出 Mermaid 代码，不要加代码块标记
- 根据内容选择合适图类型
- 忠实反映原始材料的关键信息和关系
- 节点文字使用中文，简洁清晰
- 节点文字需要分行时使用 <br/>；禁止输出字面 \\n
"""

WHITEBOARD_MERMAID_PROMPT = """你是飞书白板 Mermaid 内容生成器。只输出纯 Mermaid 代码。

上下文：
{context}

用户请求：
{message}

吸收的 lark-whiteboard 规则：
- 不靠关键词猜图形，先判断信息结构：层级/结构用 mindmap，极简流程用 flowchart，交互过程用 sequenceDiagram，状态迁移用 stateDiagram-v2，比例分布用 pie，时间排期用 gantt。
- 思维导图适合会议纪要、主题拆解、结构化总结。
- 流程图只用于简单文字流程；步骤数超过 12 时要合并步骤，节点文字尽量不超过 8 字。
- 判断节点只写条件关键词，不写长描述。
- 节点文字使用中文，简洁清晰，避免长句和大段文字。
- 节点文字需要分行时使用 <br/>；禁止输出字面 \\n。
- 只输出 Mermaid 文本，不输出 Markdown 代码块。

输出要求：
- 如果用户要求思维导图或内容是会议总结，优先输出 mindmap。
- 如果用户要求流程、链路、架构流程，输出 flowchart TD 或 LR。
- 如果用户要求角色交互、请求响应、会议对话流，输出 sequenceDiagram。
- 如果用户要求状态变化，输出 stateDiagram-v2。
- 内容必须忠实反映上下文中的事实，不添加无依据信息。
"""


def create(state: Mapping[str, Any]) -> SkillArtifact:
    topic = state.get("intent_params", {}).get("topic", "未命名白板")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")
    title = f"白板：{topic}"
    config = get_skill_config("lark_whiteboard")

    mermaid = get_llm_for_node("whiteboard").invoke(
        WHITEBOARD_PROMPT.format(
            system_prompt=config.get("system_prompt", ""),
            raw_message=raw_message,
            topic=topic,
            diagram_rules=config.get("diagram_rules", ""),
            title_rules=config.get("title_rules", ""),
            forbidden_text=config.get("forbidden_text", ""),
        )
    ).content.strip()

    return create_whiteboard_from_mermaid(
        title=title,
        mermaid=mermaid,
        user_access_token=uat,
    )


def fetch_whiteboard_content(token: str, uat: str) -> str:
    if not token or not uat:
        return ""
    try:
        resp = run_lark_cli([
            "whiteboard", "+query",
            "--whiteboard-token", token,
            "--output_as", "code",
            "--as", "user",
        ], uat=uat)
        content = str(resp.get("data", {}).get("content") or resp.get("content") or "")
        logger.info("fetch_whiteboard_content token=%r content_len=%s", token, len(content))
        return content
    except Exception:
        logger.exception("fetch_whiteboard_content failed token=%r", token)
        return ""


def create_whiteboard_from_mermaid(
    *,
    title: str,
    mermaid: str,
    user_access_token: str = "",
    parent_doc_token: str = "",
    parent_doc_url: str = "",
) -> SkillArtifact:
    mermaid = _normalize_mermaid_text(_strip_code_fence(str(mermaid or "")).strip())
    result: SkillArtifact = {
        "kind": "whiteboard",
        "title": title,
        "status": "draft",
        "preview": mermaid,
        "token": "",
        "url": "",
    }
    if not user_access_token:
        return result

    try:
        if parent_doc_token:
            artifact = _append_whiteboard_to_doc(
                title=title,
                mermaid=mermaid,
                user_access_token=user_access_token,
                doc_token=parent_doc_token,
                doc_url=parent_doc_url,
            )
            if artifact.get("status") == "created":
                return artifact

        content = f"<title>{_escape_xml(title)}</title><whiteboard type=\"blank\"></whiteboard>"
        resp = run_lark_cli([
            "docs", "+create",
            "--api-version", "v2",
            "--content", content,
            "--as", "user",
        ], uat=user_access_token)
        document = resp.get("data", {}).get("document", {})
        doc_token = document.get("document_id", "")
        wb_token = _whiteboard_token(document)
        if not doc_token or not wb_token:
            logger.error("docs +create returned no whiteboard token: %s", resp)
            return result

        update_resp = run_lark_cli([
            "whiteboard", "+update",
            "--whiteboard-token", wb_token,
            "--source", "-",
            "--input_format", "mermaid",
            "--overwrite",
            "--yes",
            "--as", "user",
        ], uat=user_access_token, stdin=mermaid)
        if _cli_ok(update_resp):
            result.update({
                "status": "created",
                "token": wb_token,
                "url": document.get("url") or f"https://www.feishu.cn/docx/{doc_token}",
            })
            logger.info("Created whiteboard %s", wb_token)
        else:
            logger.error("whiteboard +update failed: %s", update_resp)
    except Exception:
        logger.exception("whiteboard skill failed, returning draft")
    return result


def generate_whiteboard_mermaid(message: str, *, context: str = "", existing_content: str = "") -> str:
    update_suffix = (
        f"\n以下是现有白板内容（Mermaid），按用户要求修改，保留无需变更的部分：\n{existing_content}"
        if existing_content
        else ""
    )
    prompt = WHITEBOARD_MERMAID_PROMPT.format(
        context=(context or "（无）")[:9000],
        message=message,
    ) + update_suffix
    content = get_llm_for_node("whiteboard").invoke(prompt).content
    mermaid = _strip_code_fence(_content_to_text(content)).strip()
    return _clean_mermaid(mermaid)


def _append_whiteboard_to_doc(
    *,
    title: str,
    mermaid: str,
    user_access_token: str,
    doc_token: str,
    doc_url: str = "",
) -> SkillArtifact:
    result: SkillArtifact = {
        "kind": "whiteboard",
        "title": title,
        "status": "draft",
        "preview": mermaid,
        "token": "",
        "url": doc_url or f"https://www.feishu.cn/docx/{doc_token}",
    }
    update_resp = run_lark_cli([
        "docs", "+update",
        "--api-version", "v2",
        "--doc", doc_token,
        "--command", "append",
        "--content", '<whiteboard type="blank"></whiteboard>',
        "--as", "user",
    ], uat=user_access_token)
    wb_token = _whiteboard_token(update_resp.get("data", {}).get("document", {}))
    if not wb_token:
        content = update_resp.get("data", {}).get("document", {}).get("content", "")
        match = re.search(r'<whiteboard[^>]+(?:token|block_token)="([^"]+)"', content)
        wb_token = match.group(1) if match else ""
    if not wb_token:
        logger.error("docs +update returned no whiteboard token: %s", update_resp)
        return result

    whiteboard_resp = run_lark_cli([
        "whiteboard", "+update",
        "--whiteboard-token", wb_token,
        "--source", "-",
            "--input_format", "mermaid",
            "--overwrite",
            "--yes",
            "--as", "user",
        ], uat=user_access_token, stdin=mermaid)
    if _cli_ok(whiteboard_resp):
        result.update({"status": "created", "token": wb_token})
        logger.info("Inserted whiteboard %s into doc %s", wb_token, doc_token)
    else:
        logger.error("whiteboard +update failed: %s", whiteboard_resp)
    return result


def _cli_ok(resp: dict[str, Any]) -> bool:
    if not resp:
        return False
    if resp.get("ok") is False or resp.get("error"):
        return False
    code = resp.get("code")
    return code in (None, 0)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return str(content or "")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _clean_mermaid(text: str) -> str:
    text = _normalize_mermaid_text(text)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "mindmap\n  root((主题))\n    要点\n"
    first = lines[0].strip()
    valid_prefixes = (
        "mindmap",
        "flowchart",
        "graph",
        "sequenceDiagram",
        "stateDiagram",
        "classDiagram",
        "pie",
        "gantt",
        "erDiagram",
        "gitGraph",
    )
    if first.startswith(valid_prefixes):
        return "\n".join(lines).strip()
    for index, line in enumerate(lines):
        if line.strip().startswith(valid_prefixes):
            return "\n".join(lines[index:]).strip()
    title = _short_node_text(first)
    children = [_short_node_text(line) for line in lines[1:8]]
    child_text = "\n".join(f"    {child}" for child in children if child)
    return f"mindmap\n  root(({title or '主题'}))\n{child_text or '    要点'}"


def _short_node_text(text: str, limit: int = 16) -> str:
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    cleaned = re.sub(r"[`*_#>\[\]{}()]+", "", text).strip()
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned[:limit]


def _normalize_mermaid_text(text: str) -> str:
    return text.replace("\\\\n", "<br/>").replace("\\n", "<br/>")


def _whiteboard_token(document: dict[str, Any]) -> str:
    for block in document.get("new_blocks", []) or []:
        if block.get("block_type") == "whiteboard" and block.get("block_token"):
            return block["block_token"]
    return ""


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
