from pydantic import BaseModel, Field

from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

INTENT_PROMPT = """分析用户的输入消息，判断其意图类型。

可选意图：
- create_doc: 用户想要创建文档、报告、纪要、方案等文字内容
- create_whiteboard: 用户想要创建白板、流程图、思维导图等可视化内容
- create_slide: 用户想要创建PPT、幻灯片、演示稿等
- create_multi: 用户同时要求创建多种类型的内容（至少两种）
- chat: 普通聊天，没有明确的内容创建需求

用户消息：{message}

请输出意图类型和提取的关键参数（如主题）。"""


class IntentOutput(BaseModel):
    intent_type: str = Field(
        description="意图类型: create_doc | create_whiteboard | create_slide | create_multi | chat"
    )
    topic: str = Field(description="用户请求的主题或核心内容")
    confidence: float = Field(
        description="意图识别置信度，0.0-1.0。意图明确时接近1.0，模糊时接近0.0",
        ge=0.0,
        le=1.0,
    )


def _get_llm():
    """Lazy-load the LLM client to avoid import-time construction."""
    if not hasattr(_get_llm, "_instance"):
        _get_llm._instance = get_llm().with_structured_output(IntentOutput)
    return _get_llm._instance


def intent_node(state: PipelineState) -> dict:
    raw_message = state.get("raw_message", "")
    prompt = INTENT_PROMPT.format(message=raw_message)
    result: IntentOutput = _get_llm().invoke(prompt)
    return {
        "intent_type": result.intent_type,
        "intent_params": {"topic": result.topic},
        "intent_confidence": result.confidence,
    }
