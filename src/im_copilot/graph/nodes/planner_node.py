from pydantic import BaseModel, Field

from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

PLANNER_PROMPT = """根据用户的意图类型，制定一个执行计划。

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

请输出执行计划步骤列表。"""


class PlannerOutput(BaseModel):
    plan: list[str] = Field(
        description="执行计划步骤列表，如 ['doc', 'slide', 'deliver']"
    )


_llm = get_llm().with_structured_output(PlannerOutput)


def planner_node(state: PipelineState) -> dict:
    intent_type = state.get("intent_type", "chat")
    topic = state.get("intent_params", {}).get("topic", "")
    prompt = PLANNER_PROMPT.format(intent_type=intent_type, topic=topic)
    result: PlannerOutput = _llm.invoke(prompt)
    return {"plan": result.plan}
