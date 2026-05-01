from im_copilot.skills.registry import get_skill
from im_copilot.state import PipelineState


def whiteboard_node(state: PipelineState) -> dict:
    result = get_skill("lark_whiteboard.create").handler(state)
    return {"artifacts": {**state.get("artifacts", {}), "whiteboard": result}}
