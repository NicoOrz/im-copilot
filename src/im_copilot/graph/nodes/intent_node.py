from im_copilot.state import PipelineState


DOC_KEYWORDS = ("文档", "报告", "纪要", "方案")
WHITEBOARD_KEYWORDS = ("白板", "流程图", "思维导图")
SLIDE_KEYWORDS = ("PPT", "ppt", "幻灯片", "演示稿")


def _contains_any(message: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in message for keyword in keywords)


def intent_node(state: PipelineState) -> dict:
    raw_message = state.get("raw_message", "")
    matches = {
        "create_doc": _contains_any(raw_message, DOC_KEYWORDS),
        "create_whiteboard": _contains_any(raw_message, WHITEBOARD_KEYWORDS),
        "create_slide": _contains_any(raw_message, SLIDE_KEYWORDS),
    }

    matched_intents = [intent for intent, matched in matches.items() if matched]
    if len(matched_intents) >= 2:
        intent_type = "create_multi"
    elif matched_intents:
        intent_type = matched_intents[0]
    else:
        intent_type = "chat"

    return {
        "intent_type": intent_type,
        "intent_params": {"topic": raw_message},
    }
