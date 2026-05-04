from im_copilot.skills.registry import get_skill
from im_copilot.state import PipelineState


def slide_node(state: PipelineState) -> dict:
    result = get_skill("lark_slide.create").handler(state)
    return {"artifacts": {**state.get("artifacts", {}), "slide": result}}
