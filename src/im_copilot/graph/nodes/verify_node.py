from pydantic import BaseModel, Field

from im_copilot.llm import get_llm_for_node, invoke_structured_with_llm
from im_copilot.state import CheckResult, PipelineState

VERIFY_PROMPT = """你是一位严格的内容质量审核员。请审核以下内容是否满足用户需求。

用户原始请求：{raw_message}
用户意图：{intent_type}
当前任务：{task}
生成的内容预览：
{preview}

请判断：
- pass: 内容质量合格，可以直接交付
- revise: 内容需要修改（如信息不完整、逻辑混乱、与主题不符）
- clarify: 需要向用户澄清某些问题才能继续

请输出审核结果和具体原因。"""


class VerifyOutput(BaseModel):
    status: str = Field(
        description="审核结果: pass | revise | clarify"
    )
    reason: str = Field(description="审核原因和修改建议")


def verify_node(state: PipelineState) -> dict:
    """Verify the most recently generated content.

    Determines which step just completed by checking artifacts keys
    against the plan. Returns a CheckResult.
    """
    plan = state.get("plan", [])
    artifacts = state.get("artifacts", {})
    raw_message = state.get("raw_message", "")
    intent_type = state.get("intent_type", "chat")

    # Find the most recently completed task (last in plan that has a result)
    task = None
    for step in reversed(plan):
        if step != "deliver" and step in artifacts:
            task = step
            break

    if not task:
        # No content to verify, pass through
        return {
            "checks": [CheckResult(task="none", status="pass", reason="无内容需审核")],
            "reflection_iteration": state.get("reflection_iteration", 0) + 1,
        }

    result = artifacts[task]
    preview = result.get("preview", "")[:1000]  # Limit preview length

    prompt = VERIFY_PROMPT.format(
        raw_message=raw_message,
        intent_type=intent_type,
        task=task,
        preview=preview,
    )

    output = invoke_structured_with_llm(get_llm_for_node("verify"), VerifyOutput, prompt)

    return {
        "checks": [CheckResult(task=task, status=output.status, reason=output.reason)],
        "reflection_iteration": state.get("reflection_iteration", 0) + 1,
    }
