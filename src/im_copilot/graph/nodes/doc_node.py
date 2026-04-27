import logging

from im_copilot.lark_cli import run_lark_cli
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


def doc_node(state: PipelineState) -> dict:
    topic = state.get("intent_params", {}).get("topic", "未命名文档")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")
    title = f"文档：{topic}"

    markdown = _get_llm().invoke(DOC_PROMPT.format(topic=topic, raw_message=raw_message)).content.strip()

    result = {
        "kind": "doc",
        "title": title,
        "status": "draft",
        "preview": markdown,
        "token": "",
        "url": "",
    }

    try:
        resp = run_lark_cli([
            "docs", "+create",
            "--api-version", "v2",
            "--title", title,
            "--content", markdown,
            "--doc-format", "markdown",
            "--as", "user",
        ], uat=uat)
        doc_token = resp.get("data", {}).get("document", {}).get("document_id", "")
        if doc_token:
            url = f"https://www.feishu.cn/docx/{doc_token}"
            result.update({"status": "created", "token": doc_token, "url": url})
            logger.info("Created doc %s", doc_token)
        else:
            logger.error("doc +create returned no token: %s", resp)
    except Exception:
        logger.exception("doc API failed, falling back to draft")

    return {"artifacts": {**state.get("artifacts", {}), "doc": result}}
