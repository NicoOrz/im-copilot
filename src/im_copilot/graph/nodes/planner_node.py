from im_copilot.state import PipelineState

_INTENT_TO_PLAN: dict[str, list[str]] = {
    "create_doc": ["doc", "deliver"],
    "create_whiteboard": ["whiteboard", "deliver"],
    "create_slide": ["slide", "deliver"],
    "create_multi": ["doc", "whiteboard", "slide", "deliver"],
    "chat": ["deliver"],
}


def planner_node(state: PipelineState) -> dict:
    if state.get("pending_questions"):
        return {
            "plan": [],
            "pending_questions": state["pending_questions"],
        }
    if state.get("plan"):
        return {"plan": state["plan"]}

    intent_type = state.get("intent_type", "chat")
    return {"plan": _default_plan(intent_type)}


def _default_plan(intent_type: str) -> list[str]:
    return list(_INTENT_TO_PLAN.get(intent_type, ["deliver"]))

