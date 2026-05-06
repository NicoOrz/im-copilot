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

SLIDE_XML_PROMPT = """你是飞书 PPT 生成助手。输出一个 JSON 数组，每个元素是一页幻灯片的完整 XML 字符串。
只输出 JSON 数组，不要输出 Markdown 代码块、说明文字或其他内容。

画布尺寸：宽 960px，高 540px。

关键约束：
1. 渐变色必须用 rgba() + 百分比停靠点，如 `linear-gradient(135deg,rgba(15,23,42,1) 0%,rgba(56,97,140,1) 100%)`；用 rgb() 或省略停靠点会变白
2. `<content>` 的直接子元素只能是 `<p>`、`<ul>`、`<ol>`
3. 文字颜色和字号通过 `<span color="..." fontSize="...">文字</span>` 内联样式设置
4. 每页必须有明确标题和可见中文内容
5. 新建场景：slides 数量 5 到 8 页，第一页封面，最后一页结尾；修改场景：严格按用户要求的页数增删，不要自行凑数
6. bullets 每页 2 到 5 条，不要大段正文
7. 根据主题选择风格：科技/AI → 深色背景；商务汇报 → 浅色背景；未指定默认浅色
8. 参考内容只作为事实来源，不要把 XML/JSON 结构当页面内容
9. 标题不得以“我们”开头
10. 不要添加用户未要求的信息，包括密级、保密声明、汇报对象、日期、署名、部门名称
11. 用户请求中有特定标题、页数、风格、结构或内容要求时，必须优先遵循

---

## 模板示例

### 深色封面页（第一页必须是这种风格）

```xml
<slide xmlns="http://www.larkoffice.com/sml/2.0">
  <style><fill><fillColor color="linear-gradient(135deg,rgba(15,23,42,1) 0%,rgba(56,97,140,1) 100%)"/></fill></style>
  <data>
    <shape type="text" topLeftX="80" topLeftY="160" width="800" height="70">
      <content><p textAlign="center"><strong><span color="rgb(255,255,255)" fontSize="44">主标题</span></strong></p></content>
    </shape>
    <shape type="text" topLeftX="80" topLeftY="250" width="800" height="35">
      <content><p textAlign="center"><span color="rgb(148,163,184)" fontSize="20">副标题</span></p></content>
    </shape>
    <shape type="text" topLeftX="80" topLeftY="420" width="800" height="25">
      <content><p textAlign="center"><span color="rgb(100,116,139)" fontSize="14">底部信息</span></p></content>
    </shape>
  </data>
</slide>
```

### 浅色内容页（通用正文页）

```xml
<slide xmlns="http://www.larkoffice.com/sml/2.0">
  <style><fill><fillColor color="rgb(248,250,252)"/></fill></style>
  <data>
    <shape type="rect" topLeftX="60" topLeftY="40" width="4" height="35">
      <fill><fillColor color="rgb(59,130,246)"/></fill>
    </shape>
    <shape type="text" topLeftX="76" topLeftY="36" width="600" height="45">
      <content><p><strong><span color="rgb(15,23,42)" fontSize="28">页面标题</span></strong></p></content>
    </shape>
    <shape type="text" topLeftX="60" topLeftY="100" width="840" height="380">
      <content textType="body" lineSpacing="multiple:1.8">
        <p><span color="rgb(51,65,85)" fontSize="15">正文段落</span></p>
        <ul>
          <li><p><span color="rgb(51,65,85)" fontSize="15">要点一</span></p></li>
          <li><p><span color="rgb(51,65,85)" fontSize="15">要点二</span></p></li>
          <li><p><span color="rgb(51,65,85)" fontSize="15">要点三</span></p></li>
        </ul>
      </content>
    </shape>
  </data>
</slide>
```

### 数据卡片页（横排指标）

```xml
<slide xmlns="http://www.larkoffice.com/sml/2.0">
  <style><fill><fillColor color="rgb(248,250,252)"/></fill></style>
  <data>
    <shape type="text" topLeftX="60" topLeftY="36" width="600" height="45">
      <content><p><strong><span color="rgb(15,23,42)" fontSize="28">数据概览</span></strong></p></content>
    </shape>
    <shape type="rect" topLeftX="60" topLeftY="100" width="260" height="140">
      <fill><fillColor color="rgb(255,255,255)"/></fill>
      <border color="rgba(0,0,0,0.08)" width="1"/>
    </shape>
    <shape type="text" topLeftX="60" topLeftY="115" width="260" height="50">
      <content><p textAlign="center"><strong><span color="rgb(59,130,246)" fontSize="36">数值</span></strong></p></content>
    </shape>
    <shape type="text" topLeftX="60" topLeftY="175" width="260" height="25">
      <content><p textAlign="center"><span color="rgb(100,116,139)" fontSize="14">指标名称</span></p></content>
    </shape>
    <shape type="rect" topLeftX="350" topLeftY="100" width="260" height="140">
      <fill><fillColor color="rgb(255,255,255)"/></fill>
      <border color="rgba(0,0,0,0.08)" width="1"/>
    </shape>
    <shape type="text" topLeftX="350" topLeftY="115" width="260" height="50">
      <content><p textAlign="center"><strong><span color="rgb(59,130,246)" fontSize="36">数值2</span></strong></p></content>
    </shape>
    <shape type="text" topLeftX="350" topLeftY="175" width="260" height="25">
      <content><p textAlign="center"><span color="rgb(100,116,139)" fontSize="14">指标名称2</span></p></content>
    </shape>
    <shape type="rect" topLeftX="640" topLeftY="100" width="260" height="140">
      <fill><fillColor color="rgb(255,255,255)"/></fill>
      <border color="rgba(0,0,0,0.08)" width="1"/>
    </shape>
    <shape type="text" topLeftX="640" topLeftY="115" width="260" height="50">
      <content><p textAlign="center"><strong><span color="rgb(59,130,246)" fontSize="36">数值3</span></strong></p></content>
    </shape>
    <shape type="text" topLeftX="640" topLeftY="175" width="260" height="25">
      <content><p textAlign="center"><span color="rgb(100,116,139)" fontSize="14">指标名称3</span></p></content>
    </shape>
  </data>
</slide>
```

### 行动计划页（编号列表）

```xml
<slide xmlns="http://www.larkoffice.com/sml/2.0">
  <style><fill><fillColor color="rgb(248,250,252)"/></fill></style>
  <data>
    <shape type="text" topLeftX="60" topLeftY="38" width="820" height="55">
      <content><p><strong><span color="rgb(15,23,42)" fontSize="28">行动计划</span></strong></p></content>
    </shape>
    <line startX="60" startY="102" endX="900" endY="102">
      <border color="rgb(59,130,246)" width="2"/>
    </line>
    <shape type="text" topLeftX="78" topLeftY="135" width="70" height="28">
      <content><p><strong><span color="rgb(59,130,246)" fontSize="16">01</span></strong></p></content>
    </shape>
    <shape type="text" topLeftX="150" topLeftY="135" width="730" height="34">
      <content><p><span color="rgb(30,41,59)" fontSize="16">第一条行动项</span></p></content>
    </shape>
    <shape type="text" topLeftX="78" topLeftY="193" width="70" height="28">
      <content><p><strong><span color="rgb(59,130,246)" fontSize="16">02</span></strong></p></content>
    </shape>
    <shape type="text" topLeftX="150" topLeftY="193" width="730" height="34">
      <content><p><span color="rgb(30,41,59)" fontSize="16">第二条行动项</span></p></content>
    </shape>
    <shape type="text" topLeftX="78" topLeftY="251" width="70" height="28">
      <content><p><strong><span color="rgb(59,130,246)" fontSize="16">03</span></strong></p></content>
    </shape>
    <shape type="text" topLeftX="150" topLeftY="251" width="730" height="34">
      <content><p><span color="rgb(30,41,59)" fontSize="16">第三条行动项</span></p></content>
    </shape>
  </data>
</slide>
```

### 深色结尾页（最后一页必须是这种风格）

```xml
<slide xmlns="http://www.larkoffice.com/sml/2.0">
  <style><fill><fillColor color="linear-gradient(135deg,rgba(15,23,42,1) 0%,rgba(56,97,140,1) 100%)"/></fill></style>
  <data>
    <shape type="text" topLeftX="80" topLeftY="190" width="800" height="55">
      <content><p textAlign="center"><strong><span color="rgb(255,255,255)" fontSize="36">感谢语或行动号召</span></strong></p></content>
    </shape>
    <line startX="410" startY="260" endX="550" endY="260">
      <border color="rgb(59,130,246)" width="2"/>
    </line>
    <shape type="text" topLeftX="80" topLeftY="280" width="800" height="30">
      <content><p textAlign="center"><span color="rgb(148,163,184)" fontSize="16">补充说明</span></p></content>
    </shape>
  </data>
</slide>
```

---

用户请求：{message}

参考内容：
{context}

创建失败信息：
{error}

输出 JSON 数组，如：["<slide ...>...</slide>", "<slide ...>...</slide>", ...]
"""


def create(state: Mapping[str, Any]) -> SkillArtifact:
    topic = state.get("intent_params", {}).get("topic", "未命名PPT")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")

    slides_xml, cover_title = generate_slide_xml(raw_message)
    title = cover_title or f"PPT：{topic}"

    return create_slide_from_xml(
        title=title,
        slides_xml=slides_xml,
        user_access_token=uat,
    )


def fetch_slide_content(token: str, uat: str) -> str:
    if not token or not uat:
        return ""
    try:
        resp = run_lark_cli([
            "slides", "xml_presentations", "get",
            "--params", json.dumps({"xml_presentation_id": token}),
            "--as", "user",
        ], uat=uat)
        raw = json.dumps(resp, ensure_ascii=False)
        logger.info("fetch_slide_content token=%r resp_len=%s", token, len(raw))
        return raw
    except Exception:
        logger.exception("fetch_slide_content failed token=%r", token)
        return ""


def fetch_slide_ids(token: str, uat: str) -> list[str]:
    if not token or not uat:
        return []
    try:
        resp = run_lark_cli([
            "slides", "xml_presentations", "get",
            "--params", json.dumps({"xml_presentation_id": token}),
            "--as", "user",
        ], uat=uat)
        # 优先从 JSON 数组取（兼容未来 API 变更）
        slides = (
            resp.get("data", {}).get("slides")
            or resp.get("slides")
            or []
        )
        if slides and isinstance(slides[0], dict):
            ids = [str(s.get("slide_id") or s.get("id") or "") for s in slides if isinstance(s, dict)]
        else:
            # 从 presentation XML 中解析 <slide id="...">
            xml_content = (
                resp.get("data", {}).get("xml_presentation", {}).get("content", "")
                or resp.get("content", "")
            )
            ids = re.findall(r'<slide\b[^>]*\bid="([^"]+)"', xml_content)
        ids = [i for i in ids if i]
        logger.info("fetch_slide_ids token=%r count=%s", token, len(ids))
        return ids
    except Exception:
        logger.exception("fetch_slide_ids failed token=%r", token)
        return []


def _cli_ok(resp: dict[str, Any]) -> bool:
    if not resp:
        return False
    if resp.get("ok") is False or resp.get("error"):
        return False
    code = resp.get("code")
    return code in (None, 0)


def update_slide_from_xml(token: str, slides_xml: str, uat: str) -> SkillArtifact:
    url = f"https://www.feishu.cn/slides/{token}"
    result: SkillArtifact = {
        "kind": "slide",
        "title": "",
        "status": "error",
        "preview": slides_xml,
        "token": token,
        "url": url,
    }
    if not token or not uat:
        logger.warning("update_slide_from_xml skipped: missing token or uat")
        return result
    try:
        slides_payload, validation_error = _validated_slides_json(slides_xml, max_pages=20)
        if validation_error:
            result.update({"error": validation_error})
            logger.error("update_slide_from_xml validation failed: %s", validation_error)
            return result

        existing_ids = fetch_slide_ids(token, uat)
        logger.info("update_slide_from_xml token=%r existing_slides=%s", token, len(existing_ids))

        for slide_id in existing_ids:
            del_resp = run_lark_cli([
                "slides", "xml_presentation.slide", "delete",
                "--params", json.dumps({"xml_presentation_id": token, "slide_id": slide_id}),
                "--yes",
                "--as", "user",
            ], uat=uat)
            if not _cli_ok(del_resp):
                logger.warning("update_slide_from_xml delete failed slide_id=%r: %s", slide_id, del_resp)

        new_slides: list[str] = json.loads(slides_payload)
        created_count = 0
        for slide_xml in new_slides:
            clean_xml = _sanitize_slide_xml(slide_xml)
            create_resp = run_lark_cli([
                "slides", "xml_presentation.slide", "create",
                "--params", json.dumps({"xml_presentation_id": token}),
                "--data", json.dumps({"slide": {"content": clean_xml}}),
                "--yes",
                "--as", "user",
            ], uat=uat)
            if _cli_ok(create_resp):
                created_count += 1
            else:
                logger.warning("update_slide_from_xml create failed: %s", create_resp)

        if created_count > 0:
            result.update({"status": "updated"})
            logger.info("update_slide_from_xml success token=%r slides=%s/%s", token, created_count, len(new_slides))
        else:
            result.update({"status": "error", "error": "all slide creates failed"})
            logger.error("update_slide_from_xml all creates failed token=%r", token)
    except Exception:
        logger.exception("update_slide_from_xml failed token=%r", token)
    return result


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
    slides_payload, validation_error = _validated_slides_json(slides_xml)
    if validation_error:
        result.update({"status": "error", "error": validation_error})
        logger.error("slide content validation failed: %s", validation_error)
        return result

    try:
        resp = run_lark_cli([
            "slides", "+create",
            "--title", title,
            "--slides", slides_payload,
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
    existing_content: str = "",
) -> tuple[str, str]:
    """返回 (slides_xml, cover_title)，cover_title 是封面标题，可用于演示文稿命名。"""
    if existing_content:
        return _generate_update_slides(message, context=context, error=error, existing_content=existing_content)

    prompt = SLIDE_XML_PROMPT.format(
        message=message,
        context=(context or "（无）")[:9000],
        error=(error or "（无）")[:1200],
    )
    content = get_llm_for_node("slide", timeout=60, max_retries=1).invoke(prompt).content
    raw = _strip_code_fence(_content_to_text(content)).strip()

    slides, xml_error = _extract_slide_xml_list(raw)
    if slides and not xml_error:
        slides = _ensure_closing_slide_last(slides)
        cover_title = _cover_title_from_xml(slides[0])
        return json.dumps(slides, ensure_ascii=False), cover_title

    deck = _parse_deck(raw)
    if not deck["slides"]:
        deck = {"style": "business", "slides": _fallback_outline(message, context)}
    cover_title = _cover_title_from_deck(deck)
    return json.dumps(_render_slides(deck["slides"], style=deck["style"]), ensure_ascii=False), cover_title


SLIDE_UPDATE_PROMPT = """你是飞书 PPT 编辑助手。用户要求对现有演示文稿进行修改。
只输出【新增或修改的页面】的 JSON 数组，不要输出已有的未变更页面。
只输出 JSON 数组，不要输出 Markdown 代码块、说明文字或其他内容。

画布尺寸：宽 960px，高 540px。

关键约束：
1. 渐变色必须用 rgba() + 百分比停靠点
2. `<content>` 的直接子元素只能是 `<p>`、`<ul>`、`<ol>`
3. 文字颜色和字号通过 `<span color="..." fontSize="...">文字</span>` 设置
4. 每页必须有明确标题和可见中文内容
5. 严格按用户要求的页数生成，用户说加 1 页就只输出 1 页
6. bullets 每页 2 到 5 条
7. 不要输出封面页或结尾页
8. 不要添加用户未要求的信息
9. 使用浅色内容页风格（背景 rgb(248,250,252)），与现有页面保持一致

现有演示文稿完整内容：
{existing_content}

用户请求：{message}

参考内容：
{context}

创建失败信息：
{error}

输出 JSON 数组（只含新增页面）：["<slide ...>...</slide>"]
"""


def _generate_update_slides(
    message: str,
    *,
    context: str = "",
    error: str = "",
    existing_content: str = "",
) -> tuple[str, str]:
    """更新场景：LLM 只生成新增页面，代码负责插入到结尾页之前。"""
    existing_slides = _extract_slides_from_presentation(existing_content)
    annotated = _annotate_existing_slides(existing_content)

    prompt = SLIDE_UPDATE_PROMPT.format(
        message=message,
        context=(context or "（无）")[:9000],
        error=(error or "（无）")[:1200],
        existing_content=annotated,
    )
    content = get_llm_for_node("slide", timeout=60, max_retries=1).invoke(prompt).content
    raw = _strip_code_fence(_content_to_text(content)).strip()

    new_slides, xml_error = _extract_slide_xml_list(raw)
    if not new_slides or xml_error:
        logger.warning("_generate_update_slides LLM output invalid: %s", xml_error)
        return json.dumps(existing_slides, ensure_ascii=False), ""

    merged = _insert_before_closing(existing_slides, new_slides)
    cover_title = _cover_title_from_xml(merged[0]) if merged else ""
    return json.dumps(merged, ensure_ascii=False), cover_title


def _extract_slides_from_presentation(content: str) -> list[str]:
    """从 presentation XML 或 JSON 响应中提取各页 slide XML。"""
    try:
        parsed = json.loads(content)
        xml_str = (
            parsed.get("data", {}).get("xml_presentation", {}).get("content", "")
            or parsed.get("content", "")
        )
        if xml_str:
            content = xml_str
    except (json.JSONDecodeError, AttributeError):
        pass

    slides = re.findall(r"<slide\b[^>]*>.*?</slide>", content, flags=re.DOTALL)
    return [_sanitize_slide_xml(s) for s in slides]


def _insert_before_closing(existing: list[str], new_pages: list[str]) -> list[str]:
    """将新页面插入到结尾页之前。"""
    if not existing:
        return new_pages

    # 从后往前找结尾页
    for i in range(len(existing) - 1, 0, -1):
        if _is_closing_slide(existing[i]):
            return existing[:i] + new_pages + existing[i:]

    # 没找到结尾页，追加到最后
    return existing + new_pages


def _sanitize_slide_xml(xml: str) -> str:
    """去除服务端专属属性，使 XML 可被 create API 接受。"""
    xml = re.sub(r'\s+id="[^"]*"', "", xml)
    xml = re.sub(r'\s+presetHandlers="[^"]*"', "", xml)
    xml = re.sub(r'\s+fontFamily="[^"]*"', "", xml)
    xml = re.sub(r"<note[^>]*>.*?</note>", "", xml, flags=re.DOTALL)
    return xml


def _annotate_existing_slides(content: str) -> str:
    """解析现有 presentation，为每页标注角色（封面页/内容页/结尾页），方便 LLM 理解结构。"""
    try:
        parsed = json.loads(content)
        xml_str = (
            parsed.get("data", {}).get("xml_presentation", {}).get("content", "")
            or parsed.get("content", "")
        )
        if xml_str:
            content = xml_str
    except (json.JSONDecodeError, AttributeError):
        pass

    slides = re.findall(r"<slide\b[^>]*>.*?</slide>", content, flags=re.DOTALL)
    if not slides:
        return "（空演示文稿）"

    parts: list[str] = []
    for i, slide in enumerate(slides):
        clean = _sanitize_slide_xml(slide)
        title = _cover_title_from_xml(clean) or "(无标题)"
        if i == 0:
            role = "封面页"
        elif _is_closing_slide(clean):
            role = "结尾页"
        else:
            role = f"内容页{i}"
        parts.append(f"### 第{i+1}页 【{role}】标题：{title}\n```xml\n{clean}\n```")

    return "\n\n".join(parts)


def _is_closing_slide(slide_xml: str) -> bool:
    """检测是否为结尾页：深色渐变背景 + 无 bullet list + 居中文字。"""
    has_gradient = "linear-gradient" in slide_xml
    has_bullet = "<ul>" in slide_xml or "<ol>" in slide_xml
    has_center = 'textAlign="center"' in slide_xml or "textAlign=\\\"center\\\"" in slide_xml
    return has_gradient and not has_bullet and has_center


def _is_cover_slide(slide_xml: str) -> bool:
    """检测是否为封面页：深色渐变背景 + 居中 + 大字号标题（>=40）。"""
    has_gradient = "linear-gradient" in slide_xml
    has_large_title = bool(re.search(r'fontSize="4[0-9]"', slide_xml) or re.search(r'fontSize=\\"4[0-9]\\"', slide_xml))
    return has_gradient and has_large_title


def _ensure_closing_slide_last(slides: list[str]) -> list[str]:
    """确保只有一个结尾页且在最后位置。去重 + 归位。"""
    if len(slides) <= 2:
        return slides

    # 分离：封面（第一页）、内容页、结尾页
    cover = slides[0]
    closing_candidates: list[str] = []
    content_pages: list[str] = []

    for slide in slides[1:]:
        if _is_closing_slide(slide):
            closing_candidates.append(slide)
        else:
            content_pages.append(slide)

    # 保留最后一个结尾页（通常是原始的那个）
    if closing_candidates:
        closing = closing_candidates[-1]
        return [cover] + content_pages + [closing]

    # 没有检测到结尾页，保持原样
    return slides


def _cover_title_from_deck(deck: dict[str, Any]) -> str:
    for slide in (deck.get("slides") or []):
        if isinstance(slide, dict):
            title = str(slide.get("title") or "").strip()
            if title:
                return title
    return ""


def _cover_title_from_xml(slide_xml: str) -> str:
    matches = re.findall(r'fontSize="(\d+)"[^>]*>([^<]+)</span>', slide_xml)
    if not matches:
        return ""
    candidates = [(int(fs), text.strip()) for fs, text in matches if text.strip()]
    if not candidates:
        return ""
    return max(candidates, key=lambda c: c[0])[1]



def _validated_slides_json(raw: str, *, max_pages: int = 10) -> tuple[str, str]:
    slides, error = _extract_slide_xml_list(raw)
    if error:
        return "", error
    if max_pages and len(slides) > max_pages:
        return "", f"PPT 内容过多：slides +create 单次最多支持 {max_pages} 页。"
    for index, slide in enumerate(slides, start=1):
        if not _slide_has_visible_text(slide):
            return "", f"PPT 第 {index} 页缺少可见文本内容。"
    return json.dumps(slides, ensure_ascii=False), ""


def _extract_slide_xml_list(raw: str) -> tuple[list[str], str]:
    raw = _strip_code_fence(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        slides = [item.strip() for item in parsed if item.strip()]
        if not slides:
            return [], "PPT 内容为空。"
        invalid = [item for item in slides if not _is_slide_xml(item)]
        if invalid:
            return [], "PPT 内容必须是 <slide> XML 数组。"
        return slides, ""

    fragments = re.findall(r"<slide\b.*?</slide>", raw, flags=re.DOTALL)
    if fragments:
        return [item.strip() for item in fragments], ""
    return [], "PPT 内容没有有效 <slide> XML。"


def _is_slide_xml(text: str) -> bool:
    return bool(re.match(r"^\s*<slide\b[\s\S]*?</slide>\s*$", text))


def _slide_has_visible_text(slide: str) -> bool:
    text = re.sub(r"<[^>]+>", " ", slide)
    text = re.sub(r"\s+", "", text)
    return bool(text)


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
        if not bullets and not metrics and layout not in {"cover", "closing"}:
            continue
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
