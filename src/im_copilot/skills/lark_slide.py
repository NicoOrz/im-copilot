from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from html import escape
from typing import Any

from im_copilot.lark_cli import run_lark_cli
from im_copilot.llm import get_llm_for_node
from im_copilot.skills.base import SkillArtifact
from im_copilot.skills.config import get_skill_config

logger = logging.getLogger(__name__)

SLIDE_PROMPT = """{system_prompt}

用户原始请求：
{raw_message}

主题：{topic}

样式规则：
{style_rules}

页面规则：
{page_rules}

标题规则：
{title_rules}

禁止内容：
{forbidden_text}

要求：
- 输出 JSON 字符串数组，每个元素是一页完整 <slide xmlns="http://www.larkoffice.com/sml/2.0">...</slide>
- 每页内容必须位于 <data> 内
- 使用 <shape> 标签放置文字和基础形状
- 第一页为标题页，正文页保留清晰层级
- 忠实反映原始材料的关键信息
- 不要输出 JSON 以外的任何内容
"""

SLIDE_OUTLINE_PROMPT = """你是飞书汇报 PPT 内容规划助手。只输出 JSON，不要输出 Markdown 或说明文字。

用户请求：
{message}

参考内容：
{context}

创建失败信息：
{error}

要求：
- 输出 JSON 数组，数组长度 5 到 8
- 每个元素包含 title、subtitle、bullets
- title 是页面标题
- subtitle 是可选短说明，没有则为空字符串
- bullets 是 2 到 5 条中文要点
- 内容忠实覆盖参考内容，不添加无依据事实
- 不要输出 XML
"""


def create(state: Mapping[str, Any]) -> SkillArtifact:
    topic = state.get("intent_params", {}).get("topic", "未命名PPT")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")
    title = f"PPT：{topic}"

    slides_xml = generate_slide_xml(raw_message)

    return create_slide_from_xml(
        title=title,
        slides_xml=slides_xml,
        user_access_token=uat,
    )


def create_slide_from_xml(
    *,
    title: str,
    slides_xml: str,
    user_access_token: str = "",
) -> SkillArtifact:
    result: SkillArtifact = {
        "kind": "slide",
        "title": title,
        "status": "draft",
        "preview": slides_xml,
        "token": "",
        "url": "",
    }
    if not user_access_token:
        return result

    try:
        resp = run_lark_cli([
            "slides", "+create",
            "--title", title,
            "--slides", _slides_json(slides_xml),
            "--as", "user",
        ], uat=user_access_token)
        if resp.get("ok") is False or resp.get("error"):
            error = _cli_error_message(resp)
            result.update({"status": "error", "error": error})
            logger.error("slides +create failed: %s", resp)
            return result
        data = resp.get("data", {})
        presentation = data.get("presentation", {})
        pres_token = (
            data.get("xml_presentation_id")
            or presentation.get("presentation_token")
            or presentation.get("obj_token")
            or presentation.get("id", "")
        )
        if pres_token:
            result.update({
                "status": "created",
                "token": pres_token,
                "url": data.get("url") or presentation.get("url") or f"https://www.feishu.cn/slides/{pres_token}",
            })
            logger.info("Created slide %s", pres_token)
        else:
            result.update({"status": "error", "error": "slides +create returned no token"})
            logger.error("slides +create returned no token: %s", resp)
    except Exception as exc:
        result.update({"error": str(exc)})
        logger.exception("slide skill failed")
    return result


def generate_slide_xml(
    message: str,
    *,
    context: str = "",
    previous_xml: str = "",
    error: str = "",
) -> str:
    prompt = SLIDE_OUTLINE_PROMPT.format(
        message=message,
        context=(context or "（无）")[:9000],
        error=(error or "（无）")[:1200],
    )
    content = get_llm_for_node("slide", timeout=60, max_retries=1).invoke(prompt).content
    pages = _parse_outline(_strip_code_fence(_content_to_text(content)).strip())
    if not pages:
        pages = _fallback_outline(message, context)
    return json.dumps(_render_slides(pages), ensure_ascii=False)


def _slides_json(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return json.dumps(parsed, ensure_ascii=False)

    fragments = re.findall(r"<slide\b.*?</slide>", raw, flags=re.DOTALL)
    if fragments:
        return json.dumps(fragments, ensure_ascii=False)
    return json.dumps([raw], ensure_ascii=False)


def _parse_outline(raw: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict):
        parsed = parsed.get("slides") or parsed.get("pages") or []
    if not isinstance(parsed, list):
        return []

    pages: list[dict[str, Any]] = []
    for item in parsed[:8]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        subtitle = str(item.get("subtitle") or "").strip()
        raw_bullets = item.get("bullets") or item.get("points") or []
        if isinstance(raw_bullets, str):
            bullets = [line.strip(" -•\t") for line in raw_bullets.splitlines()]
        elif isinstance(raw_bullets, list):
            bullets = [str(value).strip() for value in raw_bullets]
        else:
            bullets = []
        bullets = [value for value in bullets if value][:5]
        pages.append({"title": title, "subtitle": subtitle, "bullets": bullets})
    return pages


def _fallback_outline(message: str, context: str) -> list[dict[str, Any]]:
    text = _plain_text(f"{message}\n{context}")
    chunks = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = chunks[:20] or ["根据原始文档整理核心信息", "形成汇报结构", "输出后续行动建议"]
    return [
        {"title": "汇报概览", "subtitle": "", "bullets": bullets[:3]},
        {"title": "核心信息", "subtitle": "", "bullets": bullets[3:8] or bullets[:3]},
        {"title": "关键事项", "subtitle": "", "bullets": bullets[8:13] or bullets[:3]},
        {"title": "行动计划", "subtitle": "", "bullets": bullets[13:18] or bullets[:3]},
        {"title": "总结", "subtitle": "", "bullets": bullets[-3:]},
    ]


def _render_slides(pages: list[dict[str, Any]]) -> list[str]:
    safe_pages = pages[:8] or _fallback_outline("", "")
    while len(safe_pages) < 5:
        safe_pages.append({
            "title": "补充说明",
            "subtitle": "",
            "bullets": ["围绕前述内容展开汇报", "保持信息完整和结构清晰"],
        })
    slides = [_cover_slide(safe_pages[0])]
    for page in safe_pages[1:]:
        slides.append(_content_slide(page))
    return slides


def _cover_slide(page: dict[str, Any]) -> str:
    title = _xml_text(str(page.get("title") or "汇报"))
    subtitle = _xml_text(str(page.get("subtitle") or "根据文档内容生成"))
    return (
        '<slide xmlns="http://www.larkoffice.com/sml/2.0">'
        '<style><fill><fillColor color="rgb(30,60,114)"/></fill></style>'
        '<data>'
        '<shape type="text" topLeftX="80" topLeftY="160" width="800" height="120">'
        '<content textType="title" fontSize="38" color="rgb(255,255,255)" bold="true">'
        f"<p>{title}</p>"
        '</content></shape>'
        '<shape type="text" topLeftX="84" topLeftY="290" width="760" height="80">'
        '<content textType="body" fontSize="20" color="rgb(226,232,240)">'
        f"<p>{subtitle}</p>"
        '</content></shape>'
        '</data></slide>'
    )


def _content_slide(page: dict[str, Any]) -> str:
    title = _xml_text(str(page.get("title") or "内容页"))
    subtitle = _xml_text(str(page.get("subtitle") or ""))
    bullets = [str(item) for item in page.get("bullets", []) if str(item).strip()][:5]
    bullet_items = "".join(f"<li><p>{_xml_text(item)}</p></li>" for item in bullets)
    subtitle_shape = ""
    if subtitle:
        subtitle_shape = (
            '<shape type="text" topLeftX="70" topLeftY="90" width="820" height="42">'
            '<content textType="caption" fontSize="14" color="rgb(71,85,105)">'
            f"<p>{subtitle}</p>"
            '</content></shape>'
        )
    return (
        '<slide xmlns="http://www.larkoffice.com/sml/2.0">'
        '<style><fill><fillColor color="rgb(248,250,252)"/></fill></style>'
        '<data>'
        '<shape type="rect" topLeftX="0" topLeftY="0" width="960" height="10">'
        '<fill><fillColor color="rgb(30,60,114)"/></fill>'
        '</shape>'
        '<shape type="text" topLeftX="64" topLeftY="34" width="830" height="56">'
        '<content textType="headline" fontSize="28" color="rgb(30,60,114)" bold="true">'
        f"<p>{title}</p>"
        '</content></shape>'
        f"{subtitle_shape}"
        '<shape type="text" topLeftX="86" topLeftY="140" width="800" height="330">'
        '<content textType="body" fontSize="20" color="rgb(30,41,59)">'
        f"<ul>{bullet_items}</ul>"
        '</content></shape>'
        '</data></slide>'
    )


def _plain_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _xml_text(text: str) -> str:
    return escape(text.strip(), quote=True)


def _cli_error_message(resp: dict[str, Any]) -> str:
    error = resp.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("type") or error)
    return str(resp.get("msg") or error or "slides +create failed")


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
