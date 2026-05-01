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

logger = logging.getLogger(__name__)

SLIDE_OUTLINE_PROMPT = """你是飞书汇报 PPT 内容规划助手。只输出 JSON，不要输出 Markdown 或说明文字。

用户请求：
{message}

参考内容：
{context}

创建失败信息：
{error}

吸收的 lark-slides 设计规则：
- 这是演示文稿，不是文档；每页信息密度低于文档，必须保留清晰层级和留白。
- 根据主题选择风格：科技/AI/产品用 tech；商务汇报/季度总结用 business；周报/日常汇报用 weekly；培训教程用 fresh；未指定用 business。
- 页面类型从这些值中选择：cover、summary、section、content、metrics、action、timeline、closing。
- 推荐结构：封面页、核心结论或数据概览、主题内容页、Action 项或时间线、结尾页。
- 内容页不要堆长段落；每页 bullets 控制在 2 到 5 条。
- Action 页适合负责人、任务、期限；timeline 页适合里程碑；metrics 页适合数字、目标、百分比、耗时等指标。
- 不要输出 XML；XML 由代码按飞书 slides 协议渲染。

要求：
- 输出 JSON 对象：{{"style":"business|tech|weekly|fresh","slides":[...]}}
- slides 数组长度 5 到 8
- 每个 slide 包含 layout、title、subtitle、bullets、metrics
- layout 只能是 cover、summary、section、content、metrics、action、timeline、closing
- title 是页面标题
- subtitle 是可选短说明，没有则为空字符串
- bullets 是 2 到 5 条中文要点
- metrics 是可选数组，每个元素包含 label、value、note；无指标则为空数组
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
    deck = _parse_deck(_strip_code_fence(_content_to_text(content)).strip())
    if not deck["slides"]:
        deck = {"style": "business", "slides": _fallback_outline(message, context)}
    return json.dumps(_render_slides(deck["slides"], style=deck["style"]), ensure_ascii=False)


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


def _parse_deck(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", raw)
        if not match:
            return {"style": "business", "slides": []}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"style": "business", "slides": []}

    style = "business"
    if isinstance(parsed, dict):
        style_value = str(parsed.get("style") or "").strip()
        if style_value in {"business", "tech", "weekly", "fresh"}:
            style = style_value
        parsed = parsed.get("slides") or parsed.get("pages") or []
    if not isinstance(parsed, list):
        return {"style": style, "slides": []}

    pages: list[dict[str, Any]] = []
    for item in parsed[:8]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        layout = str(item.get("layout") or "").strip()
        if layout not in {"cover", "summary", "section", "content", "metrics", "action", "timeline", "closing"}:
            layout = "content"
        subtitle = str(item.get("subtitle") or "").strip()
        raw_bullets = item.get("bullets") or item.get("points") or []
        if isinstance(raw_bullets, str):
            bullets = [line.strip(" -•\t") for line in raw_bullets.splitlines()]
        elif isinstance(raw_bullets, list):
            bullets = [str(value).strip() for value in raw_bullets]
        else:
            bullets = []
        bullets = [value for value in bullets if value][:5]
        raw_metrics = item.get("metrics") or []
        metrics: list[dict[str, str]] = []
        if isinstance(raw_metrics, list):
            for metric in raw_metrics[:4]:
                if not isinstance(metric, dict):
                    continue
                value = str(metric.get("value") or "").strip()
                label = str(metric.get("label") or "").strip()
                if value or label:
                    metrics.append({
                        "label": label,
                        "value": value,
                        "note": str(metric.get("note") or "").strip(),
                    })
        pages.append({
            "layout": layout,
            "title": title,
            "subtitle": subtitle,
            "bullets": bullets,
            "metrics": metrics,
        })
    return {"style": style, "slides": pages}


def _fallback_outline(message: str, context: str) -> list[dict[str, Any]]:
    text = _plain_text(f"{message}\n{context}")
    chunks = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = chunks[:20] or ["根据原始文档整理核心信息", "形成汇报结构", "输出后续行动建议"]
    return [
        {"layout": "cover", "title": "汇报概览", "subtitle": "", "bullets": bullets[:3], "metrics": []},
        {"layout": "summary", "title": "核心结论", "subtitle": "", "bullets": bullets[:4], "metrics": []},
        {"layout": "content", "title": "关键事项", "subtitle": "", "bullets": bullets[4:9] or bullets[:3], "metrics": []},
        {"layout": "action", "title": "行动计划", "subtitle": "", "bullets": bullets[9:14] or bullets[:3], "metrics": []},
        {"layout": "closing", "title": "总结", "subtitle": "", "bullets": bullets[-3:], "metrics": []},
    ]


def _render_slides(pages: list[dict[str, Any]], *, style: str = "business") -> list[str]:
    safe_pages = pages[:8] or _fallback_outline("", "")
    while len(safe_pages) < 5:
        safe_pages.append({
            "layout": "content",
            "title": "补充说明",
            "subtitle": "",
            "bullets": ["围绕前述内容展开汇报", "保持信息完整和结构清晰"],
            "metrics": [],
        })
    safe_pages[0]["layout"] = "cover"
    if safe_pages[-1].get("layout") == "content":
        safe_pages[-1]["layout"] = "closing"
    slides = []
    for index, page in enumerate(safe_pages):
        layout = str(page.get("layout") or "content")
        if index == 0 or layout == "cover":
            slides.append(_cover_slide(page, style=style))
        elif layout == "metrics":
            slides.append(_metrics_slide(page, style=style))
        elif layout == "action":
            slides.append(_action_slide(page, style=style))
        elif layout == "timeline":
            slides.append(_timeline_slide(page, style=style))
        elif layout == "closing":
            slides.append(_closing_slide(page, style=style))
        else:
            slides.append(_content_slide(page, style=style))
    return slides


def _cover_slide(page: dict[str, Any], *, style: str) -> str:
    title = _xml_text(str(page.get("title") or "汇报"))
    subtitle = _xml_text(str(page.get("subtitle") or "根据文档内容生成"))
    theme = _theme(style)
    return (
        '<slide xmlns="http://www.larkoffice.com/sml/2.0">'
        f'<style><fill><fillColor color="{theme["cover_bg"]}"/></fill></style>'
        '<data>'
        '<shape type="text" topLeftX="80" topLeftY="150" width="800" height="120">'
        f'<content textType="title" fontSize="40" color="{theme["cover_text"]}" bold="true">'
        f"<p>{title}</p>"
        '</content></shape>'
        '<line startX="84" startY="280" endX="220" endY="280">'
        f'<border color="{theme["accent"]}" width="4"/>'
        '</line>'
        '<shape type="text" topLeftX="84" topLeftY="310" width="760" height="80">'
        f'<content textType="body" fontSize="20" color="{theme["cover_muted"]}">'
        f"<p>{subtitle}</p>"
        '</content></shape>'
        '</data></slide>'
    )


def _content_slide(page: dict[str, Any], *, style: str) -> str:
    title = _xml_text(str(page.get("title") or "内容页"))
    subtitle = _xml_text(str(page.get("subtitle") or ""))
    bullets = [str(item) for item in page.get("bullets", []) if str(item).strip()][:5]
    theme = _theme(style)
    bullet_items = "".join(f"<li><p>{_xml_text(item)}</p></li>" for item in bullets)
    subtitle_shape = ""
    if subtitle:
        subtitle_shape = (
            '<shape type="text" topLeftX="70" topLeftY="90" width="820" height="42">'
            f'<content textType="caption" fontSize="14" color="{theme["muted"]}">'
            f"<p>{subtitle}</p>"
            '</content></shape>'
        )
    return (
        '<slide xmlns="http://www.larkoffice.com/sml/2.0">'
        f'<style><fill><fillColor color="{theme["bg"]}"/></fill></style>'
        '<data>'
        '<shape type="rect" topLeftX="60" topLeftY="40" width="4" height="38">'
        f'<fill><fillColor color="{theme["accent"]}"/></fill>'
        '</shape>'
        '<shape type="text" topLeftX="76" topLeftY="34" width="820" height="56">'
        f'<content textType="headline" fontSize="28" color="{theme["title"]}" bold="true">'
        f"<p>{title}</p>"
        '</content></shape>'
        f"{subtitle_shape}"
        '<shape type="text" topLeftX="72" topLeftY="145" width="820" height="330">'
        f'<content textType="body" fontSize="18" color="{theme["text"]}" lineSpacing="multiple:1.5">'
        f"<ul>{bullet_items}</ul>"
        '</content></shape>'
        '</data></slide>'
    )


def _metrics_slide(page: dict[str, Any], *, style: str) -> str:
    title = _xml_text(str(page.get("title") or "数据概览"))
    theme = _theme(style)
    metrics = page.get("metrics") or []
    if not metrics:
        bullets = [str(item) for item in page.get("bullets", []) if str(item).strip()][:3]
        metrics = [
            {"value": str(i + 1), "label": bullet[:18], "note": ""}
            for i, bullet in enumerate(bullets)
        ]
    cards = []
    for i, metric in enumerate(metrics[:3]):
        x = 60 + i * 290
        value = _xml_text(str(metric.get("value") or "-"))
        label = _xml_text(str(metric.get("label") or "指标"))
        note = _xml_text(str(metric.get("note") or ""))
        cards.append(
            f'<shape type="rect" topLeftX="{x}" topLeftY="130" width="260" height="190">'
            f'<fill><fillColor color="{theme["card"]}"/></fill>'
            '<border color="rgba(0,0,0,0.08)" width="1"/>'
            '</shape>'
            f'<shape type="text" topLeftX="{x}" topLeftY="155" width="260" height="60">'
            f'<content textType="headline" fontSize="34" color="{theme["accent"]}" bold="true">'
            f'<p textAlign="center">{value}</p></content></shape>'
            f'<shape type="text" topLeftX="{x + 20}" topLeftY="230" width="220" height="32">'
            f'<content textType="body" fontSize="16" color="{theme["title"]}" bold="true">'
            f'<p textAlign="center">{label}</p></content></shape>'
            f'<shape type="text" topLeftX="{x + 20}" topLeftY="270" width="220" height="36">'
            f'<content textType="caption" fontSize="12" color="{theme["muted"]}">'
            f'<p textAlign="center">{note}</p></content></shape>'
        )
    return (
        '<slide xmlns="http://www.larkoffice.com/sml/2.0">'
        f'<style><fill><fillColor color="{theme["bg"]}"/></fill></style>'
        '<data>'
        f'<shape type="text" topLeftX="60" topLeftY="40" width="820" height="55">'
        f'<content textType="headline" fontSize="28" color="{theme["title"]}" bold="true">'
        f'<p>{title}</p></content></shape>'
        f'{"".join(cards)}'
        '</data></slide>'
    )


def _action_slide(page: dict[str, Any], *, style: str) -> str:
    title = _xml_text(str(page.get("title") or "行动计划"))
    bullets = [str(item) for item in page.get("bullets", []) if str(item).strip()][:5]
    theme = _theme(style)
    rows = []
    y = 135
    for i, bullet in enumerate(bullets):
        rows.append(
            f'<shape type="text" topLeftX="78" topLeftY="{y}" width="70" height="28">'
            f'<content textType="body" fontSize="16" color="{theme["accent"]}" bold="true"><p>{i + 1:02d}</p></content></shape>'
            f'<shape type="text" topLeftX="150" topLeftY="{y}" width="730" height="34">'
            f'<content textType="body" fontSize="16" color="{theme["text"]}"><p>{_xml_text(bullet)}</p></content></shape>'
        )
        y += 58
    return (
        '<slide xmlns="http://www.larkoffice.com/sml/2.0">'
        f'<style><fill><fillColor color="{theme["bg"]}"/></fill></style>'
        '<data>'
        f'<shape type="text" topLeftX="60" topLeftY="38" width="820" height="55">'
        f'<content textType="headline" fontSize="28" color="{theme["title"]}" bold="true">'
        f'<p>{title}</p></content></shape>'
        f'<line startX="60" startY="102" endX="900" endY="102"><border color="{theme["accent"]}" width="2"/></line>'
        f'{"".join(rows)}'
        '</data></slide>'
    )


def _timeline_slide(page: dict[str, Any], *, style: str) -> str:
    title = _xml_text(str(page.get("title") or "时间线"))
    bullets = [str(item) for item in page.get("bullets", []) if str(item).strip()][:5]
    theme = _theme(style)
    items = []
    for i, bullet in enumerate(bullets):
        x = 90 + i * 160
        items.append(
            f'<line startX="{x}" startY="245" endX="{x + 120}" endY="245"><border color="{theme["muted"]}" width="2"/></line>'
            f'<shape type="rect" topLeftX="{x}" topLeftY="220" width="34" height="34">'
            f'<fill><fillColor color="{theme["accent"]}"/></fill></shape>'
            f'<shape type="text" topLeftX="{x - 18}" topLeftY="275" width="140" height="95">'
            f'<content textType="body" fontSize="14" color="{theme["text"]}"><p>{_xml_text(bullet)}</p></content></shape>'
        )
    return (
        '<slide xmlns="http://www.larkoffice.com/sml/2.0">'
        f'<style><fill><fillColor color="{theme["bg"]}"/></fill></style>'
        '<data>'
        f'<shape type="text" topLeftX="60" topLeftY="42" width="820" height="55">'
        f'<content textType="headline" fontSize="28" color="{theme["title"]}" bold="true">'
        f'<p>{title}</p></content></shape>'
        f'{"".join(items)}'
        '</data></slide>'
    )


def _closing_slide(page: dict[str, Any], *, style: str) -> str:
    title = _xml_text(str(page.get("title") or "总结"))
    subtitle = _xml_text(str(page.get("subtitle") or "感谢"))
    theme = _theme(style)
    return (
        '<slide xmlns="http://www.larkoffice.com/sml/2.0">'
        f'<style><fill><fillColor color="{theme["cover_bg"]}"/></fill></style>'
        '<data>'
        '<shape type="text" topLeftX="80" topLeftY="190" width="800" height="70">'
        f'<content textType="headline" fontSize="36" color="{theme["cover_text"]}" bold="true">'
        f'<p textAlign="center">{title}</p></content></shape>'
        f'<line startX="410" startY="275" endX="550" endY="275"><border color="{theme["accent"]}" width="2"/></line>'
        '<shape type="text" topLeftX="80" topLeftY="300" width="800" height="40">'
        f'<content textType="body" fontSize="16" color="{theme["cover_muted"]}">'
        f'<p textAlign="center">{subtitle}</p></content></shape>'
        '</data></slide>'
    )


def _plain_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _theme(style: str) -> dict[str, str]:
    themes = {
        "tech": {
            "bg": "rgb(15,23,42)",
            "cover_bg": "linear-gradient(135deg,rgba(15,23,42,1) 0%,rgba(56,97,140,1) 100%)",
            "cover_text": "rgb(255,255,255)",
            "cover_muted": "rgb(203,213,225)",
            "title": "rgb(226,232,240)",
            "text": "rgb(226,232,240)",
            "muted": "rgb(148,163,184)",
            "accent": "rgb(59,130,246)",
            "card": "rgb(30,41,59)",
        },
        "weekly": {
            "bg": "rgb(248,250,252)",
            "cover_bg": "linear-gradient(135deg,rgba(30,60,114,1) 0%,rgba(59,130,246,1) 100%)",
            "cover_text": "rgb(255,255,255)",
            "cover_muted": "rgb(219,234,254)",
            "title": "rgb(15,23,42)",
            "text": "rgb(30,41,59)",
            "muted": "rgb(100,116,139)",
            "accent": "rgb(59,130,246)",
            "card": "rgb(255,255,255)",
        },
        "fresh": {
            "bg": "rgb(255,255,255)",
            "cover_bg": "linear-gradient(135deg,rgba(22,163,74,1) 0%,rgba(34,197,94,1) 100%)",
            "cover_text": "rgb(255,255,255)",
            "cover_muted": "rgb(220,252,231)",
            "title": "rgb(20,83,45)",
            "text": "rgb(51,65,85)",
            "muted": "rgb(71,85,105)",
            "accent": "rgb(34,197,94)",
            "card": "rgb(240,253,244)",
        },
        "business": {
            "bg": "rgb(248,250,252)",
            "cover_bg": "linear-gradient(135deg,rgba(30,60,114,1) 0%,rgba(56,97,140,1) 100%)",
            "cover_text": "rgb(255,255,255)",
            "cover_muted": "rgb(226,232,240)",
            "title": "rgb(30,60,114)",
            "text": "rgb(30,41,59)",
            "muted": "rgb(71,85,105)",
            "accent": "rgb(30,60,114)",
            "card": "rgb(255,255,255)",
        },
    }
    return themes.get(style, themes["business"])


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
