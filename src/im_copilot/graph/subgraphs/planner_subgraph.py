from langgraph.graph import END, START, StateGraph
from typing import Literal

from im_copilot.graph.nodes.clarification_node import clarification_node
from im_copilot.graph.nodes.plan_approval_node import plan_approval_node
from im_copilot.graph.nodes.planner_node import planner_node
from im_copilot.state import PipelineState


def route_after_planner(state: PipelineState) -> Literal["clarification", "plan_approval"]:
    """Route after planner based on whether clarification is needed."""
    pending = state.get("pending_questions", [])
    if pending:
        return "clarification"
    return "plan_approval"


def route_after_approval(state: PipelineState) -> Literal[END, "planner"]:
    """Route after plan approval.

    If rejected, loop back to planner for revision.
    If approved, exit subgraph.
    """
    approvals = state.get("approvals", [])
    if not approvals:
        return END

    latest = approvals[-1]
    if latest.get("status") == "rejected":
        return "planner"
    return END


def build_planner_subgraph():
    """Build the planner subgraph with clarification and approval."""
    builder = StateGraph(PipelineState)
    builder.add_node("planner", planner_node)
    builder.add_node("clarification", clarification_node)
    builder.add_node("plan_approval", plan_approval_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        ["clarification", "plan_approval"],
    )
    builder.add_edge("clarification", "planner")  # loop back after clarification
    builder.add_conditional_edges(
        "plan_approval",
        route_after_approval,
        [END, "planner"],  # rejected -> back to planner
    )
    return builder.compile()
