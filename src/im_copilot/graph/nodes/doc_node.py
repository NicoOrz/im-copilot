import logging

from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

logger = logging.getLogger(__name__)

DOC_PROMPT = """你是一位专业的文档撰写助手。请根据用户提供的原始内容和主题，生成一份完整的 Markdown 文档。

用户原始请求：
{raw_message}

主题：{topic}

要求：
- 直接输出完整 Markdown 内容，包含标题（# 开头）和各章节正文
- 忠实反映原始材料的关键信息，不要添加原始材料中没有的内容
- 不要加代码块标记（```），直接输出 Markdown 文本
- 不要添加额外解释"""


def _get_llm():
    if not hasattr(_get_llm, "_instance"):
        _get_llm._instance = get_llm()
    return _get_llm._instance


def _get_lark_client(uat: str):
    from im_copilot.lark_doc import LarkDocClient
    return LarkDocClient(user_access_token=uat or None)


def doc_node(state: PipelineState) -> dict:
    topic = state.get("intent_params", {}).get("topic", "未命名文档")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")
    title = f"文档：{topic}"

    prompt = DOC_PROMPT.format(topic=topic, raw_message=raw_message)
    markdown = _get_llm().invoke(prompt).content

    result = {
        "kind": "doc",
        "title": title,
        "status": "draft",
        "preview": markdown,
        "token": "",
        "url": "",
    }

    try:
        client = _get_lark_client(uat)
        doc_token = client.create_doc(title, content=markdown)
        if doc_token:
            url = client.get_share_link(doc_token, "doc")
            result.update({"status": "created", "token": doc_token, "url": url})
    except Exception:
        logger.exception("doc API failed, falling back to draft")

    return {
        "artifacts": {
            **state.get("artifacts", {}),
            "doc": result,
        }
    }
