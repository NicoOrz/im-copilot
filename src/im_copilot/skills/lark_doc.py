from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from im_copilot.lark_cli import run_lark_cli
from im_copilot.llm import get_llm_for_node
from im_copilot.skills.base import SkillArtifact
from im_copilot.skills.config import get_skill_config

logger = logging.getLogger(__name__)

DOC_PROMPT = """{system_prompt}

用户原始请求：
{raw_message}

主题：{topic}

输出格式：{doc_format}

样式规则：
{style_rules}

块规则：
{block_rules}

标题规则：
{title_rules}

禁止内容：
{forbidden_text}

要求：
- 直接输出完整内容
- 忠实反映原始材料的关键信息，不添加原始材料中没有的内容
- DocxXML 必须包含唯一 <title>
- 固定使用 DocxXML
"""


def create(state: Mapping[str, Any]) -> SkillArtifact:
    topic = state.get("intent_params", {}).get("topic", "未命名文档")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")
    title = f"文档：{topic}"
    doc_format = "xml"
    config = get_skill_config("lark_doc")

    content = get_llm_for_node("doc").invoke(
        DOC_PROMPT.format(
            system_prompt=config.get("system_prompt", ""),
            raw_message=raw_message,
            topic=topic,
            doc_format="DocxXML",
            style_rules=config.get("style_rules", ""),
            block_rules=config.get("block_rules", ""),
            title_rules=config.get("title_rules", ""),
            forbidden_text=config.get("forbidden_text", ""),
        )
    ).content.strip()

    return create_doc_from_content(
        title=title,
        content=content,
        user_access_token=uat,
        doc_format=doc_format,
    )


def create_doc_from_content(
    *,
    title: str,
    content: str,
    user_access_token: str = "",
    doc_format: str = "xml",
) -> SkillArtifact:
    doc_format = "xml"
    logger.info(
        "create_doc_from_content start title=%r doc_format=%s content_len=%s has_user_token=%s",
        title,
        doc_format,
        len(content or ""),
        bool(user_access_token),
    )
    result: SkillArtifact = {
        "kind": "doc",
        "title": title,
        "status": "draft",
        "preview": content,
        "token": "",
        "url": "",
    }
    if not user_access_token:
        logger.info("create_doc_from_content draft_no_token title=%r preview_len=%s", title, len(content or ""))
        return result

    try:
        args = [
            "docs", "+create",
            "--api-version", "v2",
            "--content", content,
            "--as", "user",
        ]
        logger.info(
            "create_doc_from_content lark_cli_start title=%r doc_format=%s args=%s content_len=%s",
            title,
            doc_format,
            [arg for arg in args if arg != content],
            len(content or ""),
        )
        resp = run_lark_cli(args, uat=user_access_token)
        logger.info(
            "create_doc_from_content lark_cli_response title=%r keys=%s code=%s msg=%s",
            title,
            sorted(resp.keys()),
            resp.get("code"),
            resp.get("msg"),
        )
        document = resp.get("data", {}).get("document", {})
        doc_token = document.get("document_id", "")
        if doc_token:
            result.update({
                "status": "created",
                "token": doc_token,
                "url": document.get("url") or f"https://www.feishu.cn/docx/{doc_token}",
            })
            logger.info("Created doc %s", doc_token)
        else:
            logger.error("docs +create returned no token: %s", resp)
    except Exception:
        logger.exception("doc skill failed, returning draft")
    logger.info(
        "create_doc_from_content complete title=%r status=%s token=%s url=%s",
        title,
        result.get("status"),
        result.get("token"),
        result.get("url"),
    )
    return result


def fetch_doc_content(
    doc: str,
    *,
    user_access_token: str = "",
    doc_format: str = "xml",
) -> str:
    if not user_access_token or not doc:
        return ""

    try:
        resp = run_lark_cli([
            "docs", "+fetch",
            "--api-version", "v2",
            "--doc", doc,
            "--doc-format", doc_format,
            "--detail", "simple",
            "--as", "user",
        ], uat=user_access_token)
        content = str(resp.get("data", {}).get("document", {}).get("content") or "")
        logger.info("fetch_doc_content complete doc=%r content_len=%s", doc, len(content))
        return content
    except Exception:
        logger.exception("fetch_doc_content failed doc=%r", doc)
        return ""
