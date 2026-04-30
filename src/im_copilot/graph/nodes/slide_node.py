import logging

from im_copilot.lark_cli import run_lark_cli
from im_copilot.llm import get_llm_for_node
from im_copilot.state import PipelineState

logger = logging.getLogger(__name__)

SLIDE_PROMPT = """你是一位专业的PPT设计助手。请根据用户提供的原始内容和主题，生成飞书幻灯片的 XML 内容。

用户原始请求：
{raw_message}

主题：{topic}

要求：
- 生成5-8个 <slide> 片段，用英文逗号分隔，直接输出不加其他内容
- 每个 <slide> 包含 xmlns 声明和 <data> 子元素，使用 <shape> 标签放置文字
- <shape> 属性：type="text"，topLeftX、topLeftY（位置）、width、height（尺寸）
- <shape> 内用 <content textType="title"> 或 <content textType="body"> 包裹 <p> 文字
- 第一页为标题页，字号建议36；正文页字号建议24
- 忠实反映原始材料的关键信息
- 示例格式：
  <slide xmlns="http://www.larkoffice.com/sml/2.0"><data><shape type="text" topLeftX="80" topLeftY="200" width="800" height="100"><content textType="title"><p>标题</p></content></shape></data></slide>"""


def slide_node(state: PipelineState) -> dict:
    topic = state.get("intent_params", {}).get("topic", "未命名PPT")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")
    title = f"PPT：{topic}"

    slides_xml = get_llm_for_node("slide").invoke(SLIDE_PROMPT.format(topic=topic, raw_message=raw_message)).content.strip()

    result = {
        "kind": "slide",
        "title": title,
        "status": "draft",
        "preview": slides_xml,
        "token": "",
        "url": "",
    }

    try:
        resp = run_lark_cli([
            "slides", "+create",
            "--title", title,
            "--slides", f"[{slides_xml}]",
            "--as", "user",
        ], uat=uat)
        presentation = resp.get("data", {}).get("presentation", {})
        pres_token = (
            presentation.get("presentation_token")
            or presentation.get("obj_token")
            or presentation.get("id", "")
        )
        if pres_token:
            url = f"https://www.feishu.cn/slides/{pres_token}"
            result.update({"status": "created", "token": pres_token, "url": url})
            logger.info("Created slide %s", pres_token)
        else:
            logger.error("slides +create returned no token: %s", resp)
    except Exception:
        logger.exception("slide API failed, falling back to draft")

    return {"artifacts": {**state.get("artifacts", {}), "slide": result}}
