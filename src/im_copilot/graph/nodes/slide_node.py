from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

SLIDE_PROMPT = """你是一位专业的PPT设计助手。请根据用户提供的原始内容和主题，设计一份演示文稿的内容框架。

用户原始请求：
{raw_message}

主题：{topic}

请基于用户原始请求中的具体内容进行提取和设计，生成：
1. PPT标题
2. 每页幻灯片的核心内容要点（建议5-8页，忠实反映原始材料的关键信息）
3. 整体结构逻辑

注意：必须基于用户提供的原始内容，不要添加原始材料中没有的信息，不要偏离主题。
直接输出内容，不要添加额外解释。"""


def _get_llm():
    """Lazy-load the LLM client to avoid import-time construction."""
    if not hasattr(_get_llm, "_instance"):
        _get_llm._instance = get_llm()
    return _get_llm._instance


def slide_node(state: PipelineState) -> dict:
    topic = state.get("intent_params", {}).get("topic", "未命名PPT")
    raw_message = state.get("raw_message", "")
    prompt = SLIDE_PROMPT.format(topic=topic, raw_message=raw_message)
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
