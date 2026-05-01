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


def create_whiteboard_from_mermaid(
    *,
    title: str,
    mermaid: str,
    user_access_token: str = "",
    parent_doc_token: str = "",
    parent_doc_url: str = "",
) -> SkillArtifact:
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


def generate_whiteboard_mermaid(message: str, *, context: str = "") -> str:
    prompt = f"""你是白板内容生成器。请根据用户请求和上下文生成 Mermaid 思维导图或流程图。

上下文：
{context or "（无）"}

用户请求：
{message}

要求：
- 只输出 Mermaid 代码
- 不要输出代码块标记
- 优先使用 mindmap 表达思维导图需求
- 忠实反映上下文中的关键信息和结构关系
- 节点文字使用中文，简洁清晰
"""
    content = get_llm_for_node("whiteboard").invoke(prompt).content
    return _strip_code_fence(_content_to_text(content)).strip()


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
