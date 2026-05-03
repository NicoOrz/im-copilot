from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from html import escape
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
- DocxXML 必须包含唯一 <title>；标题必须来自用户主题或材料中的核心事项
- 禁止使用 Untitled、无标题、默认标题或空标题
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
    preview = content
    content = _normalize_doc_title(content, title)
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
        "preview": preview,
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


def _normalize_doc_title(content: str, title: str) -> str:
    safe_title = _safe_doc_title(title, content)
    title_re = re.compile(r"<title\b[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
    match = title_re.search(content or "")
    escaped = escape(safe_title, quote=True)
    if match:
        current = _strip_tags(match.group(1)).strip()
        if _is_placeholder_title(current):
            return title_re.sub(f"<title>{escaped}</title>", content, count=1)
        return content
    return f"<title>{escaped}</title>\n{content or ''}".strip()


def _safe_doc_title(title: str, content: str) -> str:
    candidates = [
        title,
        _strip_tags(_first_match(r"<h[1-9]\b[^>]*>(.*?)</h[1-9]>", content or "")),
    ]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not _is_placeholder_title(value):
            return value
    return "文档"


def _is_placeholder_title(title: str) -> bool:
    normalized = re.sub(r"\s+", "", str(title or "")).lower()
    return normalized in {"", "untitled", "untitleddocument", "无标题"}


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
            "--detail", "with-ids",
            "--as", "user",
        ], uat=user_access_token)
        content = str(resp.get("data", {}).get("document", {}).get("content") or "")
        logger.info("fetch_doc_content complete doc=%r content_len=%s", doc, len(content))
        return content
    except Exception:
        logger.exception("fetch_doc_content failed doc=%r", doc)
        return ""


def summarize_docx_xml_content(content: str) -> str:
    fields = extract_docx_xml_fields(content)
    if not fields:
        return ""

    compact_fields = _compact_docx_xml_fields(fields)
    lines = [
        "结构化字段 JSON:",
        json.dumps(compact_fields, ensure_ascii=False, separators=(",", ":")),
    ]
    reuse_requirements = _docx_xml_reuse_requirements(compact_fields)
    if reuse_requirements:
        lines.append("生成要求:")
        lines.extend(f"- {item}" for item in reuse_requirements)
    return "\n".join(lines)


def extract_docx_xml_fields(content: str) -> dict[str, Any]:
    root = _parse_docx_xml(content)
    if root is None:
        return _extract_docx_xml_fields_fallback(content)

    parent_by_child = {
        child: parent
        for parent in root.iter()
        for child in list(parent)
    }
    fields: dict[str, Any] = {
        "title": "",
        "headings": [],
        "callouts": [],
        "cite_users": [],
        "whiteboards": [],
        "images": [],
        "checkboxes": [],
        "links": [],
        "grids": [],
        "tables": [],
    }

    for node in root.iter():
        tag = _tag_name(node.tag)
        if tag == "title" and not fields["title"]:
            fields["title"] = _compact_text(node)
        elif tag in {f"h{i}" for i in range(1, 10)}:
            fields["headings"].append({
                "level": tag,
                "text": _compact_text(node),
            })
        elif tag == "callout":
            fields["callouts"].append({
                "emoji": node.attrib.get("emoji", ""),
                "text": _truncate(_compact_text(node), 260),
            })
        elif tag == "cite":
            cite_type = node.attrib.get("type", "")
            if cite_type == "user":
                parent = parent_by_child.get(node)
                fields["cite_users"].append({
                    "user_id": node.attrib.get("user-id", ""),
                    "context": _compact_text(parent) if parent is not None else "",
                })
            elif cite_type == "doc":
                fields["links"].append({
                    "kind": "doc_cite",
                    "text": _compact_text(parent_by_child.get(node)),
                    "doc_id": node.attrib.get("doc-id", ""),
                })
        elif tag == "whiteboard":
            fields["whiteboards"].append({
                "token": node.attrib.get("token", ""),
                "type": node.attrib.get("type", ""),
                "text": _truncate(_compact_text(node), 240),
            })
        elif tag == "img":
            fields["images"].append({
                "name": node.attrib.get("name", ""),
                "caption": node.attrib.get("caption", "").strip(),
                "href": node.attrib.get("href", ""),
                "src": node.attrib.get("src", ""),
                "width": node.attrib.get("width", ""),
                "height": node.attrib.get("height", ""),
            })
        elif tag == "checkbox":
            fields["checkboxes"].append({
                "done": node.attrib.get("done", "false"),
                "text": _truncate(_compact_text(node), 260),
                "user_ids": [
                    cite.attrib.get("user-id", "")
                    for cite in node.iter()
                    if _tag_name(cite.tag) == "cite" and cite.attrib.get("type") == "user"
                ],
            })
        elif tag == "a":
            fields["links"].append({
                "kind": "a",
                "text": _compact_text(node),
                "href": node.attrib.get("href", ""),
            })
        elif tag == "bookmark":
            fields["links"].append({
                "kind": "bookmark",
                "text": node.attrib.get("name", ""),
                "href": node.attrib.get("href", ""),
            })
        elif tag == "grid":
            fields["grids"].append(_grid_summary(node))
        elif tag == "table":
            fields["tables"].append(_table_summary(node))

    return fields


def _parse_docx_xml(content: str) -> ET.Element | None:
    if not content.strip():
        return None
    try:
        return ET.fromstring(f"<root>{content}</root>")
    except ET.ParseError:
        logger.warning("docx xml parse failed content_len=%s", len(content))
        return None


def _extract_docx_xml_fields_fallback(content: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "title": _strip_tags(_first_match(r"<title\b[^>]*>(.*?)</title>", content)),
        "headings": [
            {"level": level, "text": _strip_tags(text)}
            for level, text in re.findall(r"<(h[1-9])\b[^>]*>(.*?)</h[1-9]>", content, re.S)
        ],
        "callouts": [
            {"emoji": attrs.get("emoji", ""), "text": _truncate(_strip_tags(text), 260)}
            for attrs, text in (
                (_attrs(attrs), text)
                for attrs, text in re.findall(r"<callout\b([^>]*)>(.*?)</callout>", content, re.S)
            )
        ],
        "cite_users": [
            {"user_id": user_id, "context": ""}
            for user_id in re.findall(r"<cite\b[^>]*type=[\"']user[\"'][^>]*user-id=[\"']([^\"']+)[\"'][^>]*/?>", content)
        ],
        "whiteboards": [
            {"token": attrs.get("token", ""), "type": attrs.get("type", ""), "text": ""}
            for attrs in (_attrs(match) for match in re.findall(r"<whiteboard\b([^>]*)>", content, re.S))
        ],
        "images": [
            {
                "name": attrs.get("name", ""),
                "caption": attrs.get("caption", "").strip(),
                "href": attrs.get("href", ""),
                "src": attrs.get("src", ""),
                "width": attrs.get("width", ""),
                "height": attrs.get("height", ""),
            }
            for attrs in (_attrs(match) for match in re.findall(r"<img\b([^>]*)>", content, re.S))
        ],
        "checkboxes": [
            {
                "done": attrs.get("done", "false"),
                "text": _truncate(_strip_tags(text), 260),
                "user_ids": re.findall(r"user-id=[\"']([^\"']+)[\"']", text),
            }
            for attrs, text in (
                (_attrs(attrs), text)
                for attrs, text in re.findall(r"<checkbox\b([^>]*)>(.*?)</checkbox>", content, re.S)
            )
        ],
        "links": [
            {"kind": "a", "text": _strip_tags(text), "href": attrs.get("href", "")}
            for attrs, text in (
                (_attrs(attrs), text)
                for attrs, text in re.findall(r"<a\b([^>]*)>(.*?)</a>", content, re.S)
            )
        ],
        "grids": [
            {"columns": [], "text": _truncate(_strip_tags(text), 400)}
            for text in re.findall(r"<grid\b[^>]*>(.*?)</grid>", content, re.S)
        ],
        "tables": [
            {"headers": [], "text": _truncate(_strip_tags(text), 400)}
            for text in re.findall(r"<table\b[^>]*>(.*?)</table>", content, re.S)
        ],
    }
    return fields


def _first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.S)
    return match.group(1) if match else ""


def _attrs(text: str) -> dict[str, str]:
    return {
        key: value
        for key, value in re.findall(r"([\w:-]+)=[\"']([^\"']*)[\"']", text)
    }


def _strip_tags(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", "", text).split())


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _compact_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _grid_summary(node: ET.Element) -> dict[str, Any]:
    columns = [child for child in list(node) if _tag_name(child.tag) == "column"]
    return {
        "columns": [
            {
                "width_ratio": column.attrib.get("width-ratio", ""),
                "text": _truncate(_compact_text(column), 220),
                "images": [
                    {
                        "name": image.attrib.get("name", ""),
                        "caption": image.attrib.get("caption", "").strip(),
                        "href": image.attrib.get("href", ""),
                    }
                    for image in column.iter()
                    if _tag_name(image.tag) == "img"
                ][:4],
            }
            for column in columns
        ],
    }


def _table_summary(node: ET.Element) -> dict[str, Any]:
    headers = [
        _compact_text(cell)
        for cell in node.iter()
        if _tag_name(cell.tag) == "th"
    ][:6]
    rows = [
        [
            _truncate(_compact_text(cell), 80)
            for cell in row
            if _tag_name(cell.tag) in {"td", "th"}
        ]
        for row in node.iter()
        if _tag_name(row.tag) == "tr"
    ][:4]
    return {"headers": headers, "rows": rows}


def _compact_docx_xml_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": fields.get("title", ""),
        "headings": fields.get("headings", [])[:8],
        "callouts": fields.get("callouts", [])[:3],
        "cite_users": [
            item for item in fields.get("cite_users", [])
            if item.get("user_id")
        ][:8],
        "whiteboards": [
            item for item in fields.get("whiteboards", [])
            if item.get("token") or item.get("type") or item.get("text")
        ][:3],
        "images": [
            item for item in fields.get("images", [])
            if item.get("href") or item.get("src")
        ][:4],
        "checkboxes": fields.get("checkboxes", [])[:8],
        "links": [
            item for item in fields.get("links", [])
            if item.get("href") or item.get("doc_id")
        ][:8],
        "grids": fields.get("grids", [])[:3],
        "tables": fields.get("tables", [])[:4],
    }


def _docx_xml_reuse_requirements(fields: dict[str, Any]) -> list[str]:
    requirements: list[str] = []
    cite_users = [
        item.get("user_id", "")
        for item in fields.get("cite_users", [])
        if item.get("user_id")
    ]
    if cite_users:
        requirements.append(
            "参会人和待办负责人必须优先使用 cite_users 中的 user_id 生成 "
            f"<cite type=\"user\" user-id=\"...\">；可用 user_id: {', '.join(cite_users[:8])}"
        )

    whiteboards = fields.get("whiteboards", [])
    if whiteboards:
        tokens = [
            item.get("token", "")
            for item in whiteboards
            if item.get("token")
        ]
        if tokens:
            requirements.append(
                "总结区域必须保留至少一个已有白板："
                f"<whiteboard token=\"{tokens[0]}\"></whiteboard>"
            )
        else:
            requirements.append(
                "总结区域必须创建一个 <whiteboard type=\"mermaid\">...</whiteboard>。"
            )

    images = fields.get("images", [])
    grids = fields.get("grids", [])
    if grids:
        requirements.append(
            "总结区域必须保留至少一个 <grid><column>...</column></grid> 分栏结构。"
        )
    if images:
        reusable = [
            item.get("href") or item.get("src", "")
            for item in images
            if item.get("href") or item.get("src")
        ]
        if reusable:
            requirements.append(
                "若内容需要配图，必须优先复用 images 中的 href/src，不得编造图片地址；"
                f"首个可用图片标识: {reusable[0]}"
            )

    checkboxes = fields.get("checkboxes", [])
    if checkboxes:
        requirements.append(
            "待办区域必须使用 <checkbox done=\"true|false\">，并参考 checkboxes 的完成状态、任务文本和负责人。"
        )

    links = fields.get("links", [])
    if links:
        requirements.append(
            "相关资源区域必须放入 links 中的 href/doc_id，不得写“未提及相关链接”。"
        )
    return requirements


def _format_items(label: str, items: list[Any], limit: int = 12) -> list[str]:
    if not items:
        return []
    lines = [f"{label}:"]
    for item in items[:limit]:
        lines.append(f"- {item}")
    if len(items) > limit:
        lines.append(f"- ... 共 {len(items)} 项")
    return lines
