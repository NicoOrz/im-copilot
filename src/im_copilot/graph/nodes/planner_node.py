from pydantic import BaseModel, Field

from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

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

规则：
1. 计划必须按顺序排列
2. 最后一步必须是 deliver
3. create_multi 时，根据用户需求选择需要的步骤（doc/whiteboard/slide），然后 deliver
4. chat 时，直接 deliver

用户意图类型：{intent_type}
用户主题：{topic}
历史澄清问答：
{clarification_history}

请判断是否需要向用户澄清问题。如果意图明确，直接输出计划；如果意图模糊或缺少关键信息，输出需要澄清的问题。

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


_llm = get_llm().with_structured_output(PlannerOutput)


def planner_node(state: PipelineState) -> dict:
    intent_type = state.get("intent_type", "chat")
    topic = state.get("intent_params", {}).get("topic", "")

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
        clarification_history=history_text,
    )
    result: PlannerOutput = _llm.invoke(prompt)

    if result.needs_clarification:
        return {
            "plan": [],
            "pending_questions": result.questions,
        }

    return {"plan": result.plan}
