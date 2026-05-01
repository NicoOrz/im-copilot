import asyncio
import json
import os
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.types import Command
from starlette.middleware.sessions import SessionMiddleware

import lark_oapi as lark

from im_copilot.checkpointer import get_checkpointer
from im_copilot.graph.pipeline import build_pipeline
from im_copilot.state import PipelineState
from im_copilot.timing import start_collecting, stop_collecting
from im_copilot.user_session_store import session_store as user_session_store
from im_copilot.user_token_store import token_store
from im_copilot.web.auth import get_current_user, is_login_exempt, login_redirect_url
from im_copilot.web.ws import ws_manager

# In-memory cache for pending interrupts per thread_id
# Maps thread_id -> interrupt payload
_interrupt_cache: dict[str, dict] = {}
_interrupt_cache_lock = asyncio.Lock()
_resume_status_cache: dict[str, dict] = {}
_resume_status_lock = threading.Lock()

# Checkpointer DB path (same as checkpointer.py default)
DB_PATH = os.getenv("CHECKPOINTER_DB", ".copilot_checkpoints.sqlite")

_DEBUG_MODE = os.getenv("IM_COPILOT_DEBUG") == "1"


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
        deleted = cursor.rowcount
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        deleted += cursor.rowcount
        conn.commit()
        return deleted > 0
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def _user_can_access_thread(request: Request, thread_id: str) -> bool:
    user = getattr(request.state, "user", None)
    if not user:
        return True
    return user_session_store.has_session(user["open_id"], thread_id)


def _require_thread_access(request: Request, thread_id: str) -> None:
    if not _user_can_access_thread(request, thread_id):
        raise HTTPException(status_code=404, detail="Session not found")


def _result_response(result: dict, timing_data: dict | None = None) -> dict:
    response: dict = {
        "status": "complete",
        "summary": result.get("summary", ""),
        "artifacts": result.get("artifacts", {}),
        "plan": result.get("plan", []),
        "checks": result.get("checks", []),
    }
    if timing_data:
        response["timing"] = timing_data
    return response


def _interrupt_payload(result: dict, timing_data: dict | None = None) -> dict:
    interrupt_data = result["__interrupt__"][0]
    payload = {
        "gate": interrupt_data.value["gate"],
        "message": interrupt_data.value.get("message", ""),
    }
    for key in ("plan", "questions", "intent_type", "intent_params"):
        if key in interrupt_data.value:
            payload[key] = interrupt_data.value[key]
    response: dict = {"status": "interrupted", **payload}
    if timing_data:
        response["timing"] = timing_data
    return response


def _set_resume_status(thread_id: str, status: dict) -> None:
    with _resume_status_lock:
        _resume_status_cache[thread_id] = status


def _get_resume_status(thread_id: str) -> dict | None:
    with _resume_status_lock:
        return _resume_status_cache.get(thread_id)


def _clear_resume_status(thread_id: str) -> None:
    with _resume_status_lock:
        _resume_status_cache.pop(thread_id, None)


def _run_resume_background(thread_id: str, parsed_decision: dict | list, cp_type: str) -> None:
    if _DEBUG_MODE:
        start_collecting()
    try:
        with get_checkpointer(cp_type) as checkpointer:
            graph = build_pipeline(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke(Command(resume=parsed_decision), config=config)

        timing_data = stop_collecting().to_dict() if _DEBUG_MODE else None
        if "__interrupt__" in result:
            status = _interrupt_payload(result, timing_data)
        else:
            status = _result_response(result, timing_data)
        _set_resume_status(thread_id, status)
    except Exception as e:
        if _DEBUG_MODE:
            stop_collecting()
        _set_resume_status(thread_id, {"status": "error", "message": str(e)})


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
    ws_manager.bind_loop()
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


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if is_login_exempt(request.url.path):
        return await call_next(request)
    user = get_current_user(request)
    if user is None:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return RedirectResponse("/login", status_code=302)
    request.state.user = user
    return await call_next(request)


# SessionMiddleware must be added AFTER auth_middleware so it sits outermost
# (Starlette processes add_middleware in LIFO order)
_SESSION_SECRET = os.getenv("SESSION_SECRET", "im-copilot-dev-secret-change-me")
app.add_middleware(SessionMiddleware, secret_key=_SESSION_SECRET, max_age=7 * 24 * 3600)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = get_current_user(request)
    if user is not None:
        return RedirectResponse("/", status_code=302)
    error = request.query_params.get("error", "")
    error_msg = {
        "network": "网络错误，请重试",
        "auth": "授权失败",
        "userinfo": "获取用户信息失败",
        "state": "登录状态已失效，请重试",
    }.get(error, "")
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"login_url": login_redirect_url(request), "error": error_msg},
    )


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = request.state.user
    sessions = _sessions_for_user(user)
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"sessions": sessions, "thread_id": None, "debug": _DEBUG_MODE, "user": user},
    )


@app.get("/chat/{thread_id}", response_class=HTMLResponse)
async def chat_page(request: Request, thread_id: str):
    user = request.state.user
    _require_thread_access(request, thread_id)
    sessions = _sessions_for_user(user)
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"sessions": sessions, "thread_id": thread_id, "debug": _DEBUG_MODE, "user": user},
    )


def _sessions_for_user(user: dict | None) -> list[dict]:
    if not user:
        return _list_sessions()

    user_sessions = user_session_store.list_sessions(user["open_id"])
    all_sessions = {s["thread_id"]: s for s in _list_sessions()}
    return [
        {
            **all_sessions.get(s["thread_id"], {"thread_id": s["thread_id"], "latest_step": 0}),
            "source": s["source"],
        }
        for s in user_sessions
    ]


@app.get("/api/sessions")
async def api_list_sessions(request: Request):
    return {"sessions": _sessions_for_user(getattr(request.state, "user", None))}


@app.post("/api/sessions")
async def api_create_session(request: Request):
    thread_id = str(uuid.uuid4())
    user = getattr(request.state, "user", None)
    if user:
        user_session_store.record_session(user["open_id"], thread_id, "web")
    return {"thread_id": thread_id, "created": True}


@app.get("/api/sessions/{thread_id}")
async def api_get_session(request: Request, thread_id: str):
    _require_thread_access(request, thread_id)
    history = _get_session_history(thread_id)
    if not history:
        return {"thread_id": thread_id, "history": [], "state": None}

    # Get latest state
    latest = history[-1]["state"] if history else {}
    return {"thread_id": thread_id, "history": history, "state": latest}


@app.get("/api/sessions/{thread_id}/history")
async def api_get_session_history(request: Request, thread_id: str):
    _require_thread_access(request, thread_id)
    history = _get_session_history(thread_id)
    return {"thread_id": thread_id, "history": history}


@app.delete("/api/sessions/{thread_id}")
async def api_delete_session(request: Request, thread_id: str):
    _require_thread_access(request, thread_id)
    deleted = _delete_session(thread_id)
    async with _interrupt_cache_lock:
        _interrupt_cache.pop(thread_id, None)
    user = getattr(request.state, "user", None)
    if user:
        user_session_store.delete_session(user["open_id"], thread_id)
    return {"deleted": deleted}


@app.post("/api/sessions/{thread_id}/chat")
async def api_chat(request: Request, thread_id: str, message: str = Form(...)):
    from im_copilot.commands import parse_command, execute_command

    _require_thread_access(request, thread_id)
    _clear_resume_status(thread_id)
    user = getattr(request.state, "user", None)
    if user:
        user_session_store.record_session(user["open_id"], thread_id, "web")

    parsed = parse_command(message.strip())
    if parsed is not None:
        cmd_name, cmd_args = parsed
        cmd_result = execute_command(
            cmd_name,
            cmd_args,
            thread_id,
            thread_id,
            source="web",
            user_id=user["open_id"] if user else "",
        )
        if cmd_result.metadata.get("action") == "reset_thread":
            new_thread_id = str(uuid.uuid4())
            if user:
                user_session_store.record_session(user["open_id"], new_thread_id, "web")
            return JSONResponse({
                "status": "command",
                "command": "new",
                "message": cmd_result.response_text,
                "new_thread_id": new_thread_id,
            })
        return JSONResponse({
            "status": "command",
            "command": cmd_result.command,
            "message": cmd_result.response_text,
        })

    open_id = user["open_id"] if user else ""
    user_access_token = ""
    if open_id:
        user_access_token = token_store.get(open_id) or ""

    cp_type = os.getenv("CHECKPOINTER_TYPE", "sqlite")

    if _DEBUG_MODE:
        start_collecting()

    with get_checkpointer(cp_type) as checkpointer:
        graph = build_pipeline(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        initial_state: PipelineState = {
            "raw_message": message,
            "chat_id": thread_id,
            "message_id": str(uuid.uuid4()),
            "source": "web",
            "user_id": open_id,
            "user_access_token": user_access_token,
            "message_history": [{"role": "user", "content": message}],
            "errors": [],
            "checks": [],
            "reflection_iteration": 0,
        }

        try:
            result = graph.invoke(initial_state, config=config)
        except Exception as e:
            if _DEBUG_MODE:
                stop_collecting()
            raise HTTPException(status_code=500, detail=str(e))

    timing_data = stop_collecting().to_dict() if _DEBUG_MODE else None

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

        async with _interrupt_cache_lock:
            _interrupt_cache[thread_id] = payload
        response: dict = {"status": "interrupted", **payload}
        if timing_data:
            response["timing"] = timing_data
        return JSONResponse(response)

    # Complete
    async with _interrupt_cache_lock:
        _interrupt_cache.pop(thread_id, None)

    return JSONResponse(_result_response(result, timing_data))


@app.post("/api/sessions/{thread_id}/resume")
async def api_resume(
    request: Request,
    background_tasks: BackgroundTasks,
    thread_id: str,
    decision: str = Form(...),
):
    _require_thread_access(request, thread_id)
    async with _interrupt_cache_lock:
        if thread_id not in _interrupt_cache:
            raise HTTPException(status_code=400, detail="No pending interrupt for this session")

    parsed_decision = json.loads(decision)

    cp_type = os.getenv("CHECKPOINTER_TYPE", "sqlite")
    _set_resume_status(thread_id, {"status": "processing"})
    background_tasks.add_task(_run_resume_background, thread_id, parsed_decision, cp_type)
    return JSONResponse({"status": "processing"})


@app.get("/api/sessions/{thread_id}/status")
async def api_status(request: Request, thread_id: str):
    _require_thread_access(request, thread_id)
    resume_status = _get_resume_status(thread_id)
    if resume_status is not None:
        if resume_status.get("status") in {"complete", "error"}:
            _clear_resume_status(thread_id)
            async with _interrupt_cache_lock:
                _interrupt_cache.pop(thread_id, None)
        elif resume_status.get("status") == "interrupted":
            async with _interrupt_cache_lock:
                _interrupt_cache[thread_id] = {
                    key: value
                    for key, value in resume_status.items()
                    if key not in {"status", "timing"}
                }
        return JSONResponse(resume_status)

    async with _interrupt_cache_lock:
        cached = _interrupt_cache.get(thread_id)
    if cached is not None:
        return JSONResponse({
            "status": "interrupted",
            **cached,
        })
    return JSONResponse({"status": "idle"})


# --- Node LLM Configuration ---

KNOWN_NODES = ["intent", "planner", "doc", "whiteboard", "slide", "verify", "side_agent", "deliver"]


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    user = request.state.user
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={"nodes": KNOWN_NODES, "debug": _DEBUG_MODE, "user": user},
    )


@app.get("/api/config/nodes")
async def api_get_node_config():
    from im_copilot.llm import _load_node_config

    config = _load_node_config()
    masked = {}
    for node in KNOWN_NODES:
        node_cfg = config.get(node) or {}
        if not node_cfg:
            masked[node] = None
            continue
        entry = dict(node_cfg)
        if entry.get("api_key"):
            key = entry["api_key"]
            entry["api_key"] = f"****{key[-4:]}" if len(key) > 4 else "****"
        masked[node] = entry
    return JSONResponse({
        "nodes": masked,
        "defaults": {
            "base_url": os.getenv("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            "model": os.getenv("VOLC_MODEL", "ep-20260422180225-zllc4"),
        },
    })


@app.post("/api/config/nodes")
async def api_set_node_config(request: Request):
    import im_copilot.llm as llm_module

    body = await request.json()
    config_path = os.getenv("NODE_LLM_CONFIG", "node_llm_config.json")
    existing = llm_module._load_node_config()
    clean = {}
    for node in KNOWN_NODES:
        node_cfg = body.get(node)
        if not node_cfg:
            continue
        entry = {}
        existing_entry = existing.get(node) or {}
        for key in ("base_url", "model", "api_key"):
            val = (node_cfg.get(key) or "").strip()
            if val.startswith("****"):
                if key in existing_entry:
                    entry[key] = existing_entry[key]
            elif val:
                entry[key] = val
        if entry:
            clean[node] = entry

    with open(config_path, "w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    llm_module._config_cache = None
    llm_module._config_cache_ts = 0.0
    return JSONResponse({"saved": True})


# --- WebSocket for real-time Feishu message push ---

@app.websocket("/ws/{open_id}")
async def websocket_endpoint(websocket: WebSocket, open_id: str):
    session_open_id = websocket.session.get("open_id") if "session" in websocket.scope else None
    if not session_open_id or session_open_id != open_id:
        await websocket.close(code=1008)
        return

    await ws_manager.connect(websocket, open_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(websocket, open_id)
