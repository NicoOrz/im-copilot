import os

from pydantic import BaseModel, Field

from im_copilot.graph.nodes.history_utils import format_history
from im_copilot.llm import get_llm_for_node
from im_copilot.state import PipelineState

_CLARIFICATION_THRESHOLD = float(os.getenv("CLARIFICATION_CONFIDENCE_THRESHOLD", "0.7"))

INTENT_PROMPT = """分析用户的输入消息，判断其意图类型，并制定执行计划。

可选意图：
- create_doc: 用户想要创建文档、报告、纪要、方案等文字内容
- create_whiteboard: 用户想要创建白板、流程图、思维导图等可视化内容
- create_slide: 用户想要创建PPT、幻灯片、演示稿等
- create_multi: 用户同时要求创建多种类型的内容（至少两种）
- chat: 普通聊天，没有明确的内容创建需求

可用执行步骤：
- doc: 生成文档内容
- whiteboard: 生成白板/流程图内容
- slide: 生成PPT内容
- deliver: 汇总结果并交付给用户

规则：
1. 计划必须按顺序排列
2. 最后一步必须是 deliver
3. create_multi 时，根据用户需求选择需要的步骤（doc/whiteboard/slide），然后 deliver
4. chat 时，直接 deliver
5. 用户消息中已包含足够内容（如原文、数据、会议纪要等）时，直接输出计划，不要再问用户要材料

历史对话：
{history}

用户消息：{message}

请结合历史对话上下文，输出意图类型、主题、置信度、是否需要澄清、澄清问题和执行计划。
只有在意图真正模糊且无法推断时，才输出需要澄清的问题。"""


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
    needs_clarification: bool = Field(
        default=False,
        description="是否需要向用户澄清问题"
    )
    questions: list[str] = Field(
        default_factory=list,
        description="需要澄清的问题列表"
    )
    plan: list[str] = Field(
        default_factory=list,
        description="执行计划步骤列表，如 ['doc', 'slide', 'deliver']"
    )


def intent_node(state: PipelineState) -> dict:
    raw_message = state.get("raw_message", "")
    history = format_history(state.get("message_history", [])[:-1])
    prompt = INTENT_PROMPT.format(message=raw_message, history=history)
    llm = get_llm_for_node("intent").with_structured_output(IntentOutput)
    result: IntentOutput = llm.invoke(prompt)
    intent_type = result.intent_type
    plan = getattr(result, "plan", None)
    if not isinstance(plan, list) or not plan:
        plan = _default_plan(intent_type)

    questions = getattr(result, "questions", None)
    if not isinstance(questions, list):
        questions = []

    needs_clarification = getattr(result, "needs_clarification", False)
    if not isinstance(needs_clarification, bool):
        needs_clarification = False

    pending_questions = (
        questions
        if needs_clarification and result.confidence < _CLARIFICATION_THRESHOLD
        else []
    )

    return {
        "intent_type": intent_type,
        "intent_params": {"topic": result.topic},
        "intent_confidence": result.confidence,
        "plan": [] if pending_questions else plan,
        "pending_questions": pending_questions,
    }


_INTENT_TO_PLAN: dict[str, list[str]] = {
    "create_doc": ["doc", "deliver"],
    "create_whiteboard": ["whiteboard", "deliver"],
    "create_slide": ["slide", "deliver"],
    "create_multi": ["doc", "whiteboard", "slide", "deliver"],
    "chat": ["deliver"],
}


def _default_plan(intent_type: str) -> list[str]:
    return list(_INTENT_TO_PLAN.get(intent_type, ["deliver"]))
