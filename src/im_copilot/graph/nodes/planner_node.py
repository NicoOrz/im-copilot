from pydantic import BaseModel, Field

import os

from im_copilot.graph.nodes.history_utils import format_history
from im_copilot.llm import get_llm_for_node, invoke_structured_with_llm
from im_copilot.skills.registry import planner_capability_text
from im_copilot.state import PipelineState

_CLARIFICATION_THRESHOLD = float(os.getenv("CLARIFICATION_CONFIDENCE_THRESHOLD", "0.7"))

PLANNER_PROMPT = """根据用户的意图类型和主题，制定一个执行计划。

意图类型说明：
- create_doc: 创建文档
- create_whiteboard: 创建白板/流程图
- create_slide: 创建PPT
- create_multi: 同时创建多种内容
- chat: 普通聊天

可用执行步骤：
- doc: 生成文档内容
- whiteboard: 生成白板/流程图内容
- slide: 生成PPT内容
- deliver: 汇总结果并交付给用户

runtime skills：
{runtime_skills}

规则：
1. 计划必须按顺序排列
2. 最后一步必须是 deliver
3. create_multi 时，根据用户需求选择需要的步骤（doc/whiteboard/slide），然后 deliver
4. chat 时，直接 deliver
5. 用户消息中已包含足够内容（如原文、数据、会议纪要等）时，直接输出计划，不要再问用户要材料

用户意图类型：{intent_type}
用户主题：{topic}
用户原始消息：
{raw_message}
历史对话：
{message_history}
历史澄清问答：
{clarification_history}

请判断是否需要向用户澄清问题。如果意图明确或原始消息中已有足够内容，直接输出计划；只有在意图真正模糊且无法推断时，才输出需要澄清的问题。

请输出：
- needs_clarification: true/false
- questions: 需要澄清的问题列表（仅在 needs_clarification 为 true 时填写）
- plan: 执行计划步骤列表（仅在 needs_clarification 为 false 时填写）"""


class PlannerOutput(BaseModel):
    needs_clarification: bool = Field(
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


def planner_node(state: PipelineState) -> dict:
    if state.get("pending_questions"):
        return {
            "plan": [],
            "pending_questions": state["pending_questions"],
        }
    if state.get("plan"):
        return {"plan": state["plan"]}

    intent_type = state.get("intent_type", "chat")
    topic = state.get("intent_params", {}).get("topic", "")
    confidence = state.get("intent_confidence", 1.0)
    allow_clarification = confidence < _CLARIFICATION_THRESHOLD

    # Build clarification history text
    history = state.get("clarification_history", [])
    if history:
        history_text = "\n".join(
            f"Q: {turn['question']}\nA: {turn['answer']}"
            for turn in history
        )
    else:
        history_text = "（无历史澄清记录）"

    prompt = PLANNER_PROMPT.format(
        intent_type=intent_type,
        topic=topic,
        raw_message=state.get("raw_message", ""),
        message_history=format_history(state.get("message_history", [])[:-1]),
        clarification_history=history_text,
        runtime_skills=planner_capability_text(),
    )
    result = invoke_structured_with_llm(get_llm_for_node("planner"), PlannerOutput, prompt)

    if result.needs_clarification and allow_clarification:
        return {
            "plan": [],
            "pending_questions": result.questions,
        }

    plan = result.plan
    if not plan:
        plan = _default_plan(intent_type)
    return {"plan": plan}


_INTENT_TO_PLAN: dict[str, list[str]] = {
    "create_doc": ["doc", "deliver"],
    "create_whiteboard": ["whiteboard", "deliver"],
    "create_slide": ["slide", "deliver"],
    "create_multi": ["doc", "whiteboard", "slide", "deliver"],
    "chat": ["deliver"],
}


def _default_plan(intent_type: str) -> list[str]:
    return list(_INTENT_TO_PLAN.get(intent_type, ["deliver"]))
