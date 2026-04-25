from langgraph.graph import END, START, StateGraph

from im_copilot.checkpointer import get_checkpointer
from im_copilot.graph.nodes.clarification_node import clarification_node
from im_copilot.graph.nodes.deliver_node import deliver_node
from im_copilot.graph.nodes.doc_node import doc_node
from im_copilot.graph.nodes.intent_node import intent_node
from im_copilot.graph.nodes.plan_approval_node import plan_approval_node
from im_copilot.graph.nodes.planner_node import planner_node
from im_copilot.graph.nodes.slide_node import slide_node
from im_copilot.graph.nodes.side_agent_node import side_agent_node
from im_copilot.graph.nodes.verify_node import verify_node
from im_copilot.graph.nodes.whiteboard_node import whiteboard_node
from im_copilot.state import PipelineState


def route_after_planner(state: PipelineState) -> str:
    pending = state.get("pending_questions", [])
    if pending:
        return "clarification"
    return "plan_approval"


def route_after_approval(state: PipelineState) -> str:
    approvals = state.get("approvals", [])
    if not approvals:
        return "route_content"

    latest = approvals[-1]
    if latest.get("status") == "rejected":
        return "planner"
    return "route_content"


def route_content(state: PipelineState) -> str:
    plan = state.get("plan", [])
    if "doc" in plan:
        return "doc"
    if "whiteboard" in plan:
        return "whiteboard"
    if "slide" in plan:
        return "slide"
    return "deliver"


def route_content_node(state: PipelineState) -> dict:
    """No-op node that just routes to content nodes based on plan."""
    return {}


def fan_out_verify(state: PipelineState) -> list:
    """Fan out to verify and side_agent in parallel."""
    from langgraph.types import Send
    return [Send("verify", state), Send("side_agent", state)]


def route_after_verify_logic(state: PipelineState) -> str:
    """Route after verify_node based on the latest check result.

    - pass: continue to next content node or deliver
    - revise: route back to the same content node (if under max iterations)
    - clarify: route to deliver (for now; future: clarification node)
    """
    checks = state.get("checks", [])
    if not checks:
        return "deliver"

    latest = checks[-1]
    status = latest.get("status", "pass")
    iteration = state.get("reflection_iteration", 0)
    max_iterations = 3

    if status == "revise" and iteration < max_iterations:
        task = latest.get("task", "doc")
        if task in ("doc", "whiteboard", "slide"):
            return task
        return "deliver"

    # pass or clarify or max iterations reached
    # Find next step in plan after the verified task
    plan = state.get("plan", [])
    task = latest.get("task", "")
    try:
        idx = plan.index(task)
        for step in plan[idx + 1 :]:
            if step in ("doc", "whiteboard", "slide"):
                return step
    except ValueError:
        pass
    return "deliver"


def route_after_verify_node(state: PipelineState) -> dict:
    """No-op node that routes after verification."""
    return {}


def build_pipeline(checkpointer=None):
    builder = StateGraph(PipelineState)
    builder.add_node("intent", intent_node)
    builder.add_node("planner", planner_node)
    builder.add_node("clarification", clarification_node)
    builder.add_node("plan_approval", plan_approval_node)
    builder.add_node("doc", doc_node)
    builder.add_node("whiteboard", whiteboard_node)
    builder.add_node("slide", slide_node)
    builder.add_node("verify", verify_node)
    builder.add_node("side_agent", side_agent_node)
    builder.add_node("route_content", route_content_node)
    builder.add_node("route_after_verify", route_after_verify_node)
    builder.add_node("deliver", deliver_node)

    builder.add_edge(START, "intent")
    builder.add_edge("intent", "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        ["clarification", "plan_approval"],
    )
    builder.add_edge("clarification", "planner")
    builder.add_conditional_edges(
        "plan_approval",
        route_after_approval,
        ["route_content", "planner"],
    )
    builder.add_conditional_edges(
        "route_content",
        route_content,
        ["doc", "whiteboard", "slide", "deliver"],
    )
    builder.add_conditional_edges(
        "doc",
        fan_out_verify,
        ["verify", "side_agent"],
    )
    builder.add_conditional_edges(
        "whiteboard",
        fan_out_verify,
        ["verify", "side_agent"],
    )
    builder.add_conditional_edges(
        "slide",
        fan_out_verify,
        ["verify", "side_agent"],
    )
    builder.add_edge("verify", "route_after_verify")
    builder.add_edge("side_agent", "route_after_verify")
    builder.add_conditional_edges(
        "route_after_verify",
        route_after_verify_logic,
        ["doc", "whiteboard", "slide", "deliver"],
    )
    builder.add_edge("deliver", END)
    return builder.compile(checkpointer=checkpointer)


def run_pipeline(
    message: str,
    *,
    chat_id: str = "cli",
    message_id: str = "cli",
    source: str = "cli",
    thread_id: str | None = None,
    checkpointer_type: str | None = None,
) -> PipelineState:
    with get_checkpointer(checkpointer_type) as checkpointer:
        graph = build_pipeline(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id or f"{chat_id}-{message_id}"}}
        initial_state: PipelineState = {
            "raw_message": message,
            "chat_id": chat_id,
            "message_id": message_id,
            "source": source,
            "errors": [],
            "checks": [],
            "reflection_iteration": 0,
        }
        return graph.invoke(initial_state, config=config)


def debug_pipeline(message: str) -> None:
    """Stream the graph execution and print each step's state updates."""
    graph = build_pipeline()
    initial_state: PipelineState = {
        "raw_message": message,
        "chat_id": "cli",
        "message_id": "cli",
        "source": "cli",
        "errors": [],
        "checks": [],
        "reflection_iteration": 0,
    }
    print(f"{'='*60}")
    print(f"Input: {message!r}")
    print(f"{'='*60}\n")
    for step in graph.stream(initial_state, stream_mode="updates"):
        # Handle both dict updates and tuple interrupts
        if isinstance(step, tuple):
            print(f"--- Step: {step} ---")
            print()
            continue
        for node_name, update in step.items():
            print(f"--- Node: {node_name} ---")
            if isinstance(update, dict):
                for key, value in update.items():
                    if key == "artifacts":
                        print(f"  {key}: {list(value.keys())}")
                    elif key == "summary" and isinstance(value, str):
                        preview = value[:80].replace("\n", " ")
                        print(f"  {key}: {preview}...")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {update}")
            print()
    print(f"{'='*60}")
    print("Pipeline complete")
    print(f"{'='*60}")


def draw_mermaid() -> str:
    """Return a Mermaid diagram of the graph structure."""
    return """
graph TD
    START --> intent
    intent --> planner
    planner -->|needs clarification| clarification
    clarification --> planner
    planner -->|plan ready| plan_approval
    plan_approval -->|rejected| planner
    plan_approval -->|approved| route_content
    route_content -->|plan has doc| doc
    route_content -->|plan has whiteboard| whiteboard
    route_content -->|plan has slide| slide
    route_content -->|chat only| deliver
    doc --> verify
    whiteboard --> verify
    slide --> verify
    verify -->|pass| route_next
    verify -->|revise| doc
    verify -->|revise| whiteboard
    verify -->|revise| slide
    verify -->|clarify / max iter| deliver
    route_next -->|next content| whiteboard
    route_next -->|next content| slide
    route_next -->|done| deliver
    deliver --> END
""".strip()
