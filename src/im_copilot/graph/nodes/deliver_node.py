from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

DELIVER_PROMPT = """你是一位智能助手，负责向用户汇总任务执行结果。

用户原始请求：{raw_message}
意图类型：{intent_type}

执行结果：
{results}

请生成一段自然、友好的中文回复，向用户说明已完成的工作和关键成果。如果存在错误，请说明并道歉。"""

CHAT_PROMPT = """你是一位友好的智能助手。请回复用户的聊天消息。

用户消息：{message}

请给出自然、有帮助的中文回复。"""


def _get_llm():
    """Lazy-load the LLM client to avoid import-time construction."""
    if not hasattr(_get_llm, "_instance"):
        _get_llm._instance = get_llm()
    return _get_llm._instance


def deliver_node(state: PipelineState) -> dict:
    errors = state.get("errors", [])
    if errors:
        return {"summary": "任务执行出现错误：" + "；".join(errors)}

    intent_type = state.get("intent_type", "chat")
    raw_message = state.get("raw_message", "")

    if intent_type == "chat":
        prompt = CHAT_PROMPT.format(message=raw_message)
        summary = _get_llm().invoke(prompt).content
        return {"summary": summary}

    plan = state.get("plan", [])
    artifacts = state.get("artifacts", {})
    result_lines = []
    for step in plan:
        if step == "deliver":
            continue
        result = artifacts.get(step)
        if result:
            result_lines.append(f"【{result['title']}】\n{result['preview']}")

    results_text = "\n\n".join(result_lines) if result_lines else "无执行结果"
    prompt = DELIVER_PROMPT.format(
        raw_message=raw_message,
        intent_type=intent_type,
        results=results_text,
    )
    summary = _get_llm().invoke(prompt).content
    return {"summary": summary}
