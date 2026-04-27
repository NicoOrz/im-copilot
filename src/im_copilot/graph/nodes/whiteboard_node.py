import logging

from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

logger = logging.getLogger(__name__)

WHITEBOARD_PROMPT = """你是一位专业的可视化设计助手。请根据用户提供的原始内容和主题，生成一段 Mermaid 图表代码。

用户原始请求：
{raw_message}

主题：{topic}

要求：
- 直接输出 Mermaid 代码，不要加代码块标记（```）
- 根据内容选择合适的图类型（flowchart、sequenceDiagram、classDiagram、mindmap 等）
- 忠实反映原始材料的关键信息和关系
- 节点文字使用中文
- 不要添加额外解释"""


def _get_llm():
    if not hasattr(_get_llm, "_instance"):
        _get_llm._instance = get_llm()
    return _get_llm._instance


def _get_lark_client(uat: str):
    from im_copilot.lark_doc import LarkDocClient
    return LarkDocClient(user_access_token=uat or None)


def whiteboard_node(state: PipelineState) -> dict:
    topic = state.get("intent_params", {}).get("topic", "未命名白板")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")
    title = f"白板：{topic}"

    prompt = WHITEBOARD_PROMPT.format(topic=topic, raw_message=raw_message)
    mermaid = _get_llm().invoke(prompt).content.strip()

    result = {
        "kind": "whiteboard",
        "title": title,
        "status": "draft",
        "preview": mermaid,
        "token": "",
        "url": "",
    }

    try:
        client = _get_lark_client(uat)
        wb_token = client.create_whiteboard(title)
        if wb_token:
            client.update_whiteboard(wb_token, mermaid)
            url = client.get_share_link(wb_token, "whiteboard")
            result.update({"status": "created", "token": wb_token, "url": url})
    except Exception:
        logger.exception("whiteboard API failed, falling back to draft")

    return {
        "artifacts": {
            **state.get("artifacts", {}),
            "whiteboard": result,
        }
    }
