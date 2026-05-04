from im_copilot.skills.registry import get_skill
from im_copilot.state import PipelineState


def doc_node(state: PipelineState) -> dict:
    result = get_skill("lark_doc.create").handler(state)
    return {"artifacts": {**state.get("artifacts", {}), "doc": result}}
