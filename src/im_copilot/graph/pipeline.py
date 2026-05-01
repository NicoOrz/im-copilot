import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from langgraph.graph import END, START, StateGraph

from im_copilot.checkpointer import get_checkpointer
from im_copilot.graph.nodes.clarification_node import clarification_node
from im_copilot.graph.nodes.deliver_node import deliver_node
from im_copilot.graph.nodes.doc_node import doc_node
from im_copilot.graph.nodes.intent_node import intent_node
from im_copilot.graph.nodes.slide_node import slide_node
from im_copilot.graph.nodes.whiteboard_node import whiteboard_node
from im_copilot.state import PipelineState

logger = logging.getLogger(__name__)


def _thread_id_from_config(config: dict | None) -> str:
    if not isinstance(config, dict):
        return ""
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        return ""
    return str(configurable.get("thread_id") or "")


def _status_code_from_exception(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def _classify_exception(exc: BaseException) -> str:
    if _status_code_from_exception(exc) == 429:
        return "rate_limited"
    message = str(exc).lower()
    if "429" in message or "rate limit" in message or "too many requests" in message:
        return "rate_limited"
    if type(exc).__name__ == "GraphInterrupt":
        return "interrupted"
    return "error"


def _timed_node(node_name: str, node: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(node)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        status = "success"
        exc: BaseException | None = None
        try:
            return node(*args, **kwargs)
        except BaseException as err:
            exc = err
            status = _classify_exception(err)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "langgraph node timing: node=%s status=%s duration_ms=%.2f error=%s status_code=%s",
                node_name,
                status,
                duration_ms,
                type(exc).__name__ if exc else "",
                _status_code_from_exception(exc) if exc else "",
            )
            from im_copilot.timing import get_collector

            collector = get_collector()
            if collector is not None:
                collector.record(node_name, start * 1000, duration_ms, status)

    return wrapped


class TimedGraph:
    def __init__(self, graph: Any, graph_name: str = "pipeline") -> None:
        self._graph = graph
        self._graph_name = graph_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)

    def invoke(self, input: Any, config: dict | None = None, **kwargs: Any) -> Any:
        start = time.perf_counter()
        status = "success"
        exc: BaseException | None = None
        try:
            result = self._graph.invoke(input, config=config, **kwargs)
            if isinstance(result, dict) and "__interrupt__" in result:
                status = "interrupted"
            return result
        except BaseException as err:
            exc = err
            status = _classify_exception(err)
            raise
        finally:
            self._log_graph_timing("invoke", config, start, status, exc)

    def stream(self, input: Any, config: dict | None = None, **kwargs: Any) -> Any:
        start = time.perf_counter()
        status = "success"
        exc: BaseException | None = None
        thread_id = _thread_id_from_config(config)

        try:
            for step in self._graph.stream(input, config=config, **kwargs):
                if isinstance(step, dict) and "__interrupt__" in step:
                    status = "interrupted"
                yield step
        except GeneratorExit:
            if status == "success":
                status = "closed"
            raise
        except BaseException as err:
            exc = err
            status = _classify_exception(err)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "langgraph call timing: graph=%s mode=stream thread_id=%s status=%s duration_ms=%.2f error=%s status_code=%s",
                self._graph_name,
                thread_id,
                status,
                duration_ms,
                type(exc).__name__ if exc else "",
                _status_code_from_exception(exc) if exc else "",
            )

    def _log_graph_timing(
        self,
        mode: str,
        config: dict | None,
        start: float,
        status: str,
        exc: BaseException | None,
    ) -> None:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "langgraph call timing: graph=%s mode=%s thread_id=%s status=%s duration_ms=%.2f error=%s status_code=%s",
            self._graph_name,
            mode,
            _thread_id_from_config(config),
            status,
            duration_ms,
            type(exc).__name__ if exc else "",
            _status_code_from_exception(exc) if exc else "",
        )


def route_after_intent(state: PipelineState) -> str:
    pending = state.get("pending_questions", [])
    if pending:
        return "clarification"
    return "route_content"


def route_content(state: PipelineState) -> str:
    plan = state.get("plan", [])
    artifacts = state.get("artifacts", {})
    for step in plan:
        if step in ("doc", "whiteboard", "slide") and step not in artifacts:
            return step
    return "deliver"


def route_content_node(state: PipelineState) -> dict:
    """No-op node that just routes to content nodes based on plan."""
    return {}


def build_pipeline(checkpointer=None):
    builder = StateGraph(PipelineState)
    builder.add_node("intent", _timed_node("intent", intent_node))
    builder.add_node("clarification", _timed_node("clarification", clarification_node))
    builder.add_node("doc", _timed_node("doc", doc_node))
    builder.add_node("whiteboard", _timed_node("whiteboard", whiteboard_node))
    builder.add_node("slide", _timed_node("slide", slide_node))
    builder.add_node("route_content", _timed_node("route_content", route_content_node))
    builder.add_node("deliver", _timed_node("deliver", deliver_node))

    builder.add_edge(START, "intent")
    builder.add_conditional_edges(
        "intent",
        route_after_intent,
        ["clarification", "route_content"],
    )
    builder.add_edge("clarification", "intent")
    builder.add_conditional_edges(
        "route_content",
        route_content,
        ["doc", "whiteboard", "slide", "deliver"],
    )
    builder.add_edge("doc", "route_content")
    builder.add_edge("whiteboard", "route_content")
    builder.add_edge("slide", "route_content")
    builder.add_edge("deliver", END)
    return TimedGraph(builder.compile(checkpointer=checkpointer))


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
    intent -->|needs clarification| clarification
    clarification --> intent
    intent -->|ready| route_content
    route_content -->|next doc| doc
    route_content -->|next whiteboard| whiteboard
    route_content -->|next slide| slide
    doc --> route_content
    whiteboard --> route_content
    slide --> route_content
    route_content -->|done/chat| deliver
    deliver --> END
""".strip()
