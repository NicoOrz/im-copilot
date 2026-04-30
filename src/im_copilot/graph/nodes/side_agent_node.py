from pydantic import BaseModel, Field

from im_copilot.llm import get_llm_for_node
from im_copilot.state import PipelineState

SIDE_AGENT_PROMPT = """你是一位并行的质量验证助手。请独立审核以下内容是否满足用户需求。

用户原始请求：{raw_message}
用户意图：{intent_type}
当前任务：{task}
生成的内容预览：
{preview}

请从以下维度评估：
1. 相关性：内容是否与用户请求高度相关
2. 完整性：是否覆盖了用户需求的要点
3. 准确性：信息是否准确无误
4. 可读性：表达是否清晰易懂

请输出评估结果。"""


class SideAgentOutput(BaseModel):
    validation_score: float = Field(
        description="综合质量评分 (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    relevance: str = Field(description="相关性评价")
    completeness: str = Field(description="完整性评价")
    accuracy: str = Field(description="准确性评价")
    readability: str = Field(description="可读性评价")
    issues: list[str] = Field(
        default_factory=list,
        description="发现的问题列表",
    )


def side_agent_node(state: PipelineState) -> dict:
    """Parallel validation agent that checks content quality.

    Runs alongside verify_node to provide additional validation.
    """
    plan = state.get("plan", [])
    artifacts = state.get("artifacts", {})
    raw_message = state.get("raw_message", "")
    intent_type = state.get("intent_type", "chat")

    # Find the most recently completed task
    task = None
    for step in reversed(plan):
        if step != "deliver" and step in artifacts:
            task = step
            break

    if not task:
        return {"side_agent_results": [{"task": "none", "status": "no_content"}]}

    result = artifacts[task]
    preview = result.get("preview", "")[:1000]

    prompt = SIDE_AGENT_PROMPT.format(
        raw_message=raw_message,
        intent_type=intent_type,
        task=task,
        preview=preview,
    )

    output: SideAgentOutput = get_llm_for_node("side_agent").with_structured_output(SideAgentOutput).invoke(prompt)

    return {
        "side_agent_results": [{
            "task": task,
            "validation_score": output.validation_score,
            "relevance": output.relevance,
            "completeness": output.completeness,
            "accuracy": output.accuracy,
            "readability": output.readability,
            "issues": output.issues,
        }]
    }
