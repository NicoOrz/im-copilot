import logging

from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

logger = logging.getLogger(__name__)

SLIDE_PROMPT = """你是一位专业的PPT设计助手。请根据用户提供的原始内容和主题，生成飞书幻灯片的 XML 内容。

用户原始请求：
{raw_message}

主题：{topic}

要求：
- 生成5-8个 <slide> 片段，用英文逗号分隔，直接输出不加其他内容
- 每个 <slide> 包含 <elements> 子元素，使用 <text> 标签放置文字
- <text> 属性：content（文字内容）、x、y（位置，单位pt）、width、height（尺寸）、fontSize（字号）
- 第一页为标题页，fontSize 建议36；正文页 fontSize 建议24
- 忠实反映原始材料的关键信息，不要添加原始材料中没有的内容
- 示例格式（仅供参考，不要照抄）：
  <slide><elements><text content="标题" x="100" y="200" width="600" height="80" fontSize="36"/></elements></slide>,<slide><elements><text content="要点1" x="80" y="150" width="640" height="60" fontSize="24"/></elements></slide>"""


def _get_llm():
    if not hasattr(_get_llm, "_instance"):
        _get_llm._instance = get_llm()
    return _get_llm._instance


def _get_lark_client(uat: str):
    from im_copilot.lark_doc import LarkDocClient
    return LarkDocClient(user_access_token=uat or None)


def slide_node(state: PipelineState) -> dict:
    topic = state.get("intent_params", {}).get("topic", "未命名PPT")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")
    title = f"PPT：{topic}"

    prompt = SLIDE_PROMPT.format(topic=topic, raw_message=raw_message)
    slides_xml = _get_llm().invoke(prompt).content.strip()

    result = {
        "kind": "slide",
        "title": title,
        "status": "draft",
        "preview": slides_xml,
        "token": "",
        "url": "",
    }

    try:
        client = _get_lark_client(uat)
        pres_token = client.create_slide_with_content(title, slides_xml)
        if pres_token:
            url = client.get_share_link(pres_token, "slide")
            result.update({"status": "created", "token": pres_token, "url": url})
    except Exception:
        logger.exception("slide API failed, falling back to draft")

    return {
        "artifacts": {
            **state.get("artifacts", {}),
            "slide": result,
        }
    }
