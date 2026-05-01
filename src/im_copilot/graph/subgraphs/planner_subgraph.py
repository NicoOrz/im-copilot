from langgraph.graph import END, START, StateGraph

from im_copilot.graph.nodes.clarification_node import clarification_node
from im_copilot.graph.nodes.planner_node import planner_node
from im_copilot.state import PipelineState


def route_after_planner(state: PipelineState) -> str:
    """Route after planner based on whether clarification is needed."""
    pending = state.get("pending_questions", [])
    if pending:
        return "clarification"
    return END


def build_planner_subgraph():
    """Build the planner subgraph with clarification only."""
    builder = StateGraph(PipelineState)
    builder.add_node("planner", planner_node)
    builder.add_node("clarification", clarification_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        ["clarification", END],
    )
    builder.add_edge("clarification", "planner")  # loop back after clarification
    return builder.compile()
