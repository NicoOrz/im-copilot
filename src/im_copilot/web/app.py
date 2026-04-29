import json
import os
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.types import Command

import lark_oapi as lark

from im_copilot.checkpointer import get_checkpointer
from im_copilot.graph.pipeline import build_pipeline
from im_copilot.state import PipelineState

# In-memory cache for pending interrupts per thread_id
# Maps thread_id -> interrupt payload
_interrupt_cache: dict[str, dict] = {}

# Checkpointer DB path (same as checkpointer.py default)
DB_PATH = os.getenv("CHECKPOINTER_DB", ".copilot_checkpoints.sqlite")


def _get_db_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _list_sessions() -> list[dict]:
    """Query the SQLite checkpointer for distinct thread_ids."""
    if not os.path.exists(DB_PATH):
        return []
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        # LangGraph sqlite checkpointer stores metadata in a table
        # The exact schema may vary; query for distinct thread_ids from checkpoints
        cursor.execute(
            "SELECT DISTINCT thread_id, MAX(json_extract(metadata, '$.step')) as latest_step "
            "FROM checkpoints GROUP BY thread_id ORDER BY latest_step DESC"
        )
        rows = cursor.fetchall()
        sessions = []
        for row in rows:
            sessions.append({
                "thread_id": row["thread_id"],
                "latest_step": row["latest_step"] or 0,
            })
        return sessions
    except sqlite3.OperationalError:
        # Table may not exist yet
        return []
    finally:
        conn.close()


def _get_session_history(thread_id: str) -> list[dict]:
    """Get the full execution history for a thread from the checkpointer.

    Returns a list of step records, each containing:
    - step: int
    - node: str (which node produced this step)
    - state: dict (the state after this step)
    - interrupt: dict | None (if this step triggered an interrupt)
    """
    if not os.path.exists(DB_PATH):
        return []

    from langgraph.checkpoint.sqlite import SqliteSaver

    checkpoints = []
    try:
        with SqliteSaver.from_conn_string(DB_PATH) as saver:
            config = {"configurable": {"thread_id": thread_id}}
            for checkpoint_tuple in saver.list(config):
                metadata = checkpoint_tuple.metadata or {}
                step = metadata.get("step", -1)
                state = checkpoint_tuple.checkpoint.get("channel_values", {})

                # Extract interrupt info from pending_writes
                interrupt = None
                pending_writes = checkpoint_tuple.pending_writes or []
                for write in pending_writes:
                    # write is a tuple: (task_id, channel, value)
                    if len(write) >= 3 and write[1] == "__interrupt__":
                        interrupt_value = write[2]
                        if isinstance(interrupt_value, list) and len(interrupt_value) > 0:
                            iv = interrupt_value[0]
                            interrupt = {
                                "gate": iv.value.get("gate") if hasattr(iv, "value") else str(iv),
                                "message": iv.value.get("message", "") if hasattr(iv, "value") else "",
                                "questions": iv.value.get("questions", []) if hasattr(iv, "value") else [],
                                "plan": iv.value.get("plan", []) if hasattr(iv, "value") else [],
                            }

                checkpoints.append({
                    "step": step,
                    "state": state,
                    "interrupt": interrupt,
                    "timestamp": checkpoint_tuple.checkpoint.get("ts", ""),
                })
    except Exception as e:
        # Fallback: return empty history if checkpointer API fails
        print(f"Error reading history for {thread_id}: {e}")
        return []

    checkpoints.sort(key=lambda x: x["step"])
    history = []
    previous_state: dict = {}
    for checkpoint in checkpoints:
        state = checkpoint["state"]
        clean_state = _clean_state_for_display(state)
        nodes = _infer_node_names(previous_state, state, checkpoint["step"])
        for index, node_name in enumerate(nodes):
            history.append({
                "step": checkpoint["step"],
                "node": node_name,
                "state": clean_state,
                "interrupt": checkpoint["interrupt"] if index == 0 else None,
                "timestamp": checkpoint["timestamp"],
            })
        previous_state = state

    history.sort(key=lambda x: x["step"])
    return history


def _infer_node_names(previous_state: dict, state: dict, step: int) -> list[str]:
    """Infer executed nodes from checkpoint state differences."""
    if step == -1:
        return ["input"]

    changed_keys = {
        key
        for key, value in state.items()
        if previous_state.get(key) != value
    }
    changed_keys.update(
        key for key in previous_state if key not in state
    )

    nodes = []
    if {"raw_message", "message_id"} & changed_keys:
        nodes.append("input")
    if "intent_type" in changed_keys or "intent_params" in changed_keys:
        nodes.append("intent")
    if ("plan" in changed_keys and state.get("plan")) or state.get("pending_questions"):
        nodes.append("planner")
    if "clarification_history" in changed_keys and state.get("clarification_history"):
        nodes.append("clarification")
    if "approvals" in changed_keys and state.get("approvals"):
        nodes.append("plan_approval")
    if "artifacts" in changed_keys and state.get("artifacts"):
        nodes.extend(_changed_artifact_nodes(previous_state, state))
    if "checks" in changed_keys and state.get("checks"):
        nodes.append("verify")
    if "side_agent_results" in changed_keys and state.get("side_agent_results"):
        nodes.append("side_agent")
    if "summary" in changed_keys:
        nodes.append("deliver")

    branch_nodes = {
        "branch:to:intent": "intent",
        "branch:to:planner": "planner",
        "branch:to:clarification": "clarification",
        "branch:to:plan_approval": "plan_approval",
        "branch:to:route_content": "route_content",
        "branch:to:doc": "doc",
        "branch:to:whiteboard": "whiteboard",
        "branch:to:slide": "slide",
        "branch:to:verify": "verify",
        "branch:to:side_agent": "side_agent",
        "branch:to:route_after_verify": "route_after_verify",
        "branch:to:deliver": "deliver",
    }
    if not nodes:
        for branch_key, node in branch_nodes.items():
            if branch_key in changed_keys:
                nodes.append(node)
                break

    return _unique(nodes) or [f"step_{step}"]


def _changed_artifact_nodes(previous_state: dict, state: dict) -> list[str]:
    previous = previous_state.get("artifacts", {}) or {}
    current = state.get("artifacts", {}) or {}
    nodes = []
    for key in ("doc", "whiteboard", "slide"):
        if previous.get(key) != current.get(key) and key in current:
            nodes.append(key)
    return nodes or ["content"]


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _clean_state_for_display(state: dict) -> dict:
    """Remove internal branch channels from state for display."""
    cleaned = {}
    for key, value in state.items():
        if key.startswith("branch:to:"):
            continue
        if key.startswith("__"):
            continue
        cleaned[key] = value
    return cleaned


def _delete_session(thread_id: str) -> bool:
    """Delete a session from the checkpointer."""
    if not os.path.exists(DB_PATH):
        return False
    conn = _get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def _start_lark_bot_thread() -> threading.Thread | None:
    """Start the Lark WebSocket client in a background daemon thread."""
    from im_copilot.lark_bot import LarkBot
    from im_copilot.lark_handlers import build_event_handler

    app_id = os.getenv("LARK_APP_ID")
    app_secret = os.getenv("LARK_APP_SECRET")
    if not app_id or not app_secret:
        print("Warning: LARK_BOT_ENABLED set but LARK_APP_ID or LARK_APP_SECRET missing")
        return None

    def _run() -> None:
        bot = LarkBot(
            app_id=app_id,
            app_secret=app_secret,
            encrypt_key=os.getenv("LARK_ENCRYPT_KEY"),
            verification_token=os.getenv("LARK_VERIFICATION_TOKEN"),
            domain=os.getenv("LARK_DOMAIN", lark.FEISHU_DOMAIN),
            debug=os.getenv("LARK_BOT_DEBUG") == "1",
        )
        handler = build_event_handler(bot)
        bot.start_ws(handler)

    t = threading.Thread(target=_run, daemon=True, name="lark-ws")
    t.start()
    return t


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    ws_thread = None
    if os.getenv("LARK_BOT_ENABLED"):
        ws_thread = _start_lark_bot_thread()
        if ws_thread:
            print("Lark WebSocket client started in background thread")

    # Startup: ensure DB exists
    if not os.path.exists(DB_PATH):
        conn = _get_db_connection()
        conn.close()
    yield
    # Shutdown: cleanup
    _interrupt_cache.clear()


app = FastAPI(title="IM Copilot", lifespan=lifespan)

# Static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_dir)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    sessions = _list_sessions()
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"sessions": sessions, "thread_id": None},
    )


@app.get("/chat/{thread_id}", response_class=HTMLResponse)
async def chat_page(request: Request, thread_id: str):
    sessions = _list_sessions()
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"sessions": sessions, "thread_id": thread_id},
    )


@app.get("/api/sessions")
async def api_list_sessions():
    sessions = _list_sessions()
    return {"sessions": sessions}


@app.post("/api/sessions")
async def api_create_session():
    thread_id = str(uuid.uuid4())
    return {"thread_id": thread_id, "created": True}


@app.get("/api/sessions/{thread_id}")
async def api_get_session(thread_id: str):
    history = _get_session_history(thread_id)
    if not history:
        return {"thread_id": thread_id, "history": [], "state": None}

    # Get latest state
    latest = history[-1]["state"] if history else {}
    return {"thread_id": thread_id, "history": history, "state": latest}


@app.get("/api/sessions/{thread_id}/history")
async def api_get_session_history(thread_id: str):
    history = _get_session_history(thread_id)
    return {"thread_id": thread_id, "history": history}


@app.delete("/api/sessions/{thread_id}")
async def api_delete_session(thread_id: str):
    deleted = _delete_session(thread_id)
    if thread_id in _interrupt_cache:
        del _interrupt_cache[thread_id]
    return {"deleted": deleted}


@app.post("/api/sessions/{thread_id}/chat")
async def api_chat(thread_id: str, message: str = Form(...)):
    """Send a message to the graph.

    Returns:
        - status: "complete" | "interrupted"
        - For complete: summary, artifacts
        - For interrupted: gate, questions/plan, etc.
    """
    cp_type = os.getenv("CHECKPOINTER_TYPE", "sqlite")

    with get_checkpointer(cp_type) as checkpointer:
        graph = build_pipeline(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        initial_state: PipelineState = {
            "raw_message": message,
            "chat_id": thread_id,
            "message_id": str(uuid.uuid4()),
            "source": "web",
            "errors": [],
            "checks": [],
            "reflection_iteration": 0,
        }

        try:
            result = graph.invoke(initial_state, config=config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Handle interrupt
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0]
        payload = {
            "gate": interrupt_data.value["gate"],
            "message": interrupt_data.value.get("message", ""),
        }
        if "plan" in interrupt_data.value:
            payload["plan"] = interrupt_data.value["plan"]
        if "questions" in interrupt_data.value:
            payload["questions"] = interrupt_data.value["questions"]
        if "intent_type" in interrupt_data.value:
            payload["intent_type"] = interrupt_data.value["intent_type"]
        if "intent_params" in interrupt_data.value:
            payload["intent_params"] = interrupt_data.value["intent_params"]

        _interrupt_cache[thread_id] = payload
        return JSONResponse({
            "status": "interrupted",
            **payload,
        })

    # Complete
    if thread_id in _interrupt_cache:
        del _interrupt_cache[thread_id]

    return JSONResponse({
        "status": "complete",
        "summary": result.get("summary", ""),
        "artifacts": result.get("artifacts", {}),
        "plan": result.get("plan", []),
        "checks": result.get("checks", []),
    })


@app.post("/api/sessions/{thread_id}/resume")
async def api_resume(thread_id: str, decision: str = Form(...)):
    """Resume from an interrupt.

    decision: JSON string with the user's response.
        - plan_approval: {"approved": true/false, "feedback": "..."}
        - clarification: ["answer1", "answer2", ...] or single string
    """
    if thread_id not in _interrupt_cache:
        raise HTTPException(status_code=400, detail="No pending interrupt for this session")

    gate = _interrupt_cache[thread_id]["gate"]
    parsed_decision = json.loads(decision)

    cp_type = os.getenv("CHECKPOINTER_TYPE", "sqlite")

    with get_checkpointer(cp_type) as checkpointer:
        graph = build_pipeline(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = graph.invoke(Command(resume=parsed_decision), config=config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Handle another interrupt
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0]
        payload = {
            "gate": interrupt_data.value["gate"],
            "message": interrupt_data.value.get("message", ""),
        }
        if "plan" in interrupt_data.value:
            payload["plan"] = interrupt_data.value["plan"]
        if "questions" in interrupt_data.value:
            payload["questions"] = interrupt_data.value["questions"]
        if "intent_type" in interrupt_data.value:
            payload["intent_type"] = interrupt_data.value["intent_type"]
        if "intent_params" in interrupt_data.value:
            payload["intent_params"] = interrupt_data.value["intent_params"]

        _interrupt_cache[thread_id] = payload
        return JSONResponse({
            "status": "interrupted",
            **payload,
        })

    # Complete
    del _interrupt_cache[thread_id]

    return JSONResponse({
        "status": "complete",
        "summary": result.get("summary", ""),
        "artifacts": result.get("artifacts", {}),
        "plan": result.get("plan", []),
        "checks": result.get("checks", []),
    })


@app.get("/api/sessions/{thread_id}/status")
async def api_status(thread_id: str):
    """Check if there's a pending interrupt for this session."""
    if thread_id in _interrupt_cache:
        return JSONResponse({
            "status": "interrupted",
            **_interrupt_cache[thread_id],
        })
    return JSONResponse({"status": "idle"})
