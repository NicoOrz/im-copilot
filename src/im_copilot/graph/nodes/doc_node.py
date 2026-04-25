from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

DOC_PROMPT = """你是一位专业的文档撰写助手。请根据用户提供的主题，生成一份结构化的文档大纲和内容摘要。

主题：{topic}

请生成：
1. 文档标题
2. 核心内容摘要（200字以内）
3. 文档结构大纲

直接输出内容，不要添加额外解释。"""


def _get_llm():
    """Lazy-load the LLM client to avoid import-time construction."""
    if not hasattr(_get_llm, "_instance"):
        _get_llm._instance = get_llm()
    return _get_llm._instance


def doc_node(state: PipelineState) -> dict:
    topic = state.get("intent_params", {}).get("topic", "未命名文档")
    prompt = DOC_PROMPT.format(topic=topic)
    content = _get_llm().invoke(prompt).content
    return {
        "artifacts": {
            **state.get("artifacts", {}),
            "doc": {
                "kind": "doc",
                "title": f"文档：{topic}",
                "status": "created",
                "preview": content,
            },
        }
    }
