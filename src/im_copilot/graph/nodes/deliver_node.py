from im_copilot.graph.nodes.history_utils import format_history
from im_copilot.llm import get_llm_for_node
from im_copilot.state import PipelineState

CHAT_PROMPT = """你是一位友好的智能助手。请回复用户的聊天消息。

历史对话：
{history}

当前用户消息：{message}

请结合历史对话上下文，给出自然、有帮助的中文回复。"""


def _artifact_status_text(status: str) -> str:
    if status == "created":
        return "已创建"
    if status == "draft":
        return "已生成草稿"
    if status == "error":
        return "生成失败"
    return status or "已生成"


def _artifact_summary_lines(plan: list[str], artifacts: dict) -> list[str]:
    lines: list[str] = []
    emitted: set[str] = set()
    ordered_steps = [step for step in plan if step != "deliver"] + [
        step for step in artifacts.keys() if step not in plan
    ]

    for step in ordered_steps:
        if step in emitted:
            continue
        emitted.add(step)
        result = artifacts.get(step)
        if not isinstance(result, dict):
            continue

        title = result.get("title") or step
        url = result.get("url") or ""
        if url:
            lines.append(f"- {title}：{url}")
            continue

        lines.append(f"- {title}：{_artifact_status_text(result.get('status', ''))}")

    return lines


def deliver_node(state: PipelineState) -> dict:
    errors = state.get("errors", [])
    if errors:
        return {"summary": "任务执行出现错误：" + "；".join(errors)}

    intent_type = state.get("intent_type", "chat")
    raw_message = state.get("raw_message", "")

    if intent_type == "chat":
        llm = get_llm_for_node("deliver")
        history = format_history(state.get("message_history", [])[:-1])
        prompt = CHAT_PROMPT.format(message=raw_message, history=history)
        summary = llm.invoke(prompt).content
        return {
            "summary": summary,
            "message_history": [{"role": "assistant", "content": summary}],
        }

    plan = state.get("plan", [])
    artifacts = state.get("artifacts", {})
    artifact_lines = _artifact_summary_lines(plan, artifacts)
    summary = "已完成。"
    if artifact_lines:
        summary += "\n" + "\n".join(artifact_lines)
    else:
        summary = "已完成处理，但没有生成可交付资源。"

    return {
        "summary": summary,
        "message_history": [{"role": "assistant", "content": summary}],
    }
