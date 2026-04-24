from langgraph.graph import END, START, StateGraph

from im_copilot.graph.nodes.deliver_node import deliver_node
from im_copilot.graph.nodes.doc_node import doc_node
from im_copilot.graph.nodes.intent_node import intent_node
from im_copilot.graph.nodes.planner_node import planner_node
from im_copilot.graph.nodes.slide_node import slide_node
from im_copilot.graph.nodes.whiteboard_node import whiteboard_node
from im_copilot.state import PipelineState


def route_after_planner(state: PipelineState) -> str:
    plan = state.get("plan", [])
    if "doc" in plan:
        return "doc"
    if "whiteboard" in plan:
        return "whiteboard"
    if "slide" in plan:
        return "slide"
    return "deliver"


def route_after_doc(state: PipelineState) -> str:
    plan = state.get("plan", [])
    if "whiteboard" in plan:
        return "whiteboard"
    if "slide" in plan:
        return "slide"
    return "deliver"


def route_after_whiteboard(state: PipelineState) -> str:
    if "slide" in state.get("plan", []):
        return "slide"
    return "deliver"


def build_pipeline():
    builder = StateGraph(PipelineState)
    builder.add_node("intent", intent_node)
    builder.add_node("planner", planner_node)
    builder.add_node("doc", doc_node)
    builder.add_node("whiteboard", whiteboard_node)
    builder.add_node("slide", slide_node)
    builder.add_node("deliver", deliver_node)

    builder.add_edge(START, "intent")
    builder.add_edge("intent", "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        ["doc", "whiteboard", "slide", "deliver"],
    )
    builder.add_conditional_edges(
        "doc",
        route_after_doc,
        ["whiteboard", "slide", "deliver"],
    )
    builder.add_conditional_edges(
        "whiteboard",
        route_after_whiteboard,
        ["slide", "deliver"],
    )
    builder.add_edge("slide", "deliver")
    builder.add_edge("deliver", END)
    return builder.compile()


def run_pipeline(
    message: str,
    *,
    chat_id: str = "cli",
    message_id: str = "cli",
    source: str = "cli",
) -> PipelineState:
    graph = build_pipeline()
    initial_state: PipelineState = {
        "raw_message": message,
        "chat_id": chat_id,
        "message_id": message_id,
        "source": source,
        "errors": [],
        "checks": [],
    }
    return graph.invoke(initial_state)
