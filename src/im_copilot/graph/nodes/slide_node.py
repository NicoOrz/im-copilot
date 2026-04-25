from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

SLIDE_PROMPT = """你是一位专业的PPT设计助手。请根据用户提供的主题，设计一份演示文稿的内容框架。

主题：{topic}

请生成：
1. PPT标题
2. 每页幻灯片的核心内容要点（建议5-8页）
3. 整体结构逻辑

直接输出内容，不要添加额外解释。"""


def _get_llm():
    """Lazy-load the LLM client to avoid import-time construction."""
    if not hasattr(_get_llm, "_instance"):
        _get_llm._instance = get_llm()
    return _get_llm._instance


def slide_node(state: PipelineState) -> dict:
    topic = state.get("intent_params", {}).get("topic", "未命名PPT")
    prompt = SLIDE_PROMPT.format(topic=topic)
    content = _get_llm().invoke(prompt).content
    return {
        "artifacts": {
            **state.get("artifacts", {}),
            "slide": {
                "kind": "slide",
                "title": f"PPT：{topic}",
                "status": "created",
                "preview": content,
            },
        }
    }
