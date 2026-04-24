from im_copilot.state import PipelineState


PLAN_BY_INTENT = {
    "create_doc": ["doc", "deliver"],
    "create_whiteboard": ["whiteboard", "deliver"],
    "create_slide": ["slide", "deliver"],
    "create_multi": ["doc", "whiteboard", "slide", "deliver"],
    "chat": ["deliver"],
}


def planner_node(state: PipelineState) -> dict:
    intent_type = state.get("intent_type", "chat")
    return {"plan": list(PLAN_BY_INTENT.get(intent_type, ["deliver"]))}
