from __future__ import annotations

import copy
import json
import os
import sqlite3
import time
from typing import Any, Literal

EventType = Literal[
    "user_message",
    "agent_stage",
    "tool_call",
    "artifact_created",
    "todo_detected",
    "todo_reminded",
    "calendar_event_created",
    "summary_created",
    "assistant_message",
    "todo_updated",
    "board_item_updated",
    "error",
]

EVENT_TYPES: set[str] = {
    "user_message",
    "agent_stage",
    "tool_call",
    "artifact_created",
    "todo_detected",
    "todo_reminded",
    "calendar_event_created",
    "summary_created",
    "assistant_message",
    "todo_updated",
    "board_item_updated",
    "error",
}

_DDL = """
CREATE TABLE IF NOT EXISTS agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_events_thread ON agent_events(thread_id, id);
CREATE INDEX IF NOT EXISTS idx_agent_events_type ON agent_events(event_type, created_at);
"""


def db_path() -> str:
    return os.getenv(
        "AGENT_EVENTS_DB",
        os.getenv("CHECKPOINTER_DB", ".copilot_checkpoints.sqlite"),
    )


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    return conn


def record_event(
    thread_id: str,
    source: str,
    event_type: EventType,
    payload: dict[str, Any] | None = None,
) -> int:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type: {event_type}")
    with _conn() as conn:
        cursor = conn.execute(
            "INSERT INTO agent_events (thread_id, source, event_type, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                thread_id,
                source,
                event_type,
                json.dumps(payload or {}, ensure_ascii=False),
                time.time(),
            ),
        )
        return int(cursor.lastrowid)


def list_events(thread_id: str) -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, thread_id, source, event_type, payload_json, created_at "
            "FROM agent_events WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()
    return [_row_to_event(row) for row in rows]


def list_threads() -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT thread_id, source, MAX(id) AS latest_step, MAX(created_at) AS updated_at "
            "FROM agent_events GROUP BY thread_id ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_thread(thread_id: str) -> bool:
    with _conn() as conn:
        cursor = conn.execute("DELETE FROM agent_events WHERE thread_id = ?", (thread_id,))
        return cursor.rowcount > 0


def iter_user_messages_for_chat(chat_id: str, since_ts: float) -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, thread_id, source, event_type, payload_json, created_at "
            "FROM agent_events "
            "WHERE event_type = 'user_message' AND created_at >= ? "
            "ORDER BY id ASC",
            (since_ts,),
        ).fetchall()
    events = [_row_to_event(row) for row in rows]
    return [
        event for event in events
        if event.get("payload", {}).get("chat_id") == chat_id
    ]


def recent_user_messages_for_chat(
    chat_id: str,
    *,
    before_event_id: int | None = None,
    limit: int,
    since_ts: float = 0.0,
) -> list[dict[str, Any]]:
    clauses = ["event_type = 'user_message'", "created_at >= ?"]
    params: list[Any] = [since_ts]
    if before_event_id is not None:
        clauses.append("id <= ?")
        params.append(before_event_id)
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, thread_id, source, event_type, payload_json, created_at "
            f"FROM agent_events WHERE {' AND '.join(clauses)} "
            "ORDER BY id ASC",
            params,
        ).fetchall()
    events = [_row_to_event(row) for row in rows]
    filtered = [
        event for event in events
        if event.get("payload", {}).get("chat_id") == chat_id
    ]
    return filtered[-limit:] if limit > 0 else filtered


def history_for_thread(thread_id: str) -> list[dict[str, Any]]:
    state: dict[str, Any] = {
        "artifacts": {},
        "errors": [],
        "checks": [],
        "message_history": [],
    }
    history: list[dict[str, Any]] = []

    for event in list_events(thread_id):
        payload = event["payload"]
        event_type = event["event_type"]
        if event_type == "user_message":
            text = str(payload.get("text") or payload.get("message") or "")
            state.update({
                "raw_message": text,
                "chat_id": payload.get("chat_id", ""),
                "message_id": payload.get("message_id", ""),
                "source": event["source"],
                "user_id": payload.get("user_id") or payload.get("source_open_id", ""),
            })
            state["message_history"] = state.get("message_history", []) + [
                {"role": "user", "content": text}
            ]
        elif event_type == "artifact_created":
            artifact = payload.get("artifact")
            if isinstance(artifact, dict):
                key = str(artifact.get("kind") or payload.get("kind") or "artifact")
                artifacts = dict(state.get("artifacts", {}))
                artifacts[key] = artifact
                state["artifacts"] = artifacts
        elif event_type == "assistant_message":
            summary = str(payload.get("summary") or payload.get("text") or "")
            state["summary"] = summary
            state["message_history"] = state.get("message_history", []) + [
                {"role": "assistant", "content": summary}
            ]
        elif event_type == "summary_created":
            if payload.get("summary"):
                state["summary"] = str(payload["summary"])
        elif event_type == "todo_detected":
            todos = list(state.get("todos", []))
            todos.append(payload)
            state["todos"] = todos
        elif event_type == "todo_updated":
            todos = list(state.get("todos", []))
            todo_id = payload.get("id")
            replaced = False
            for index, todo in enumerate(todos):
                if todo.get("id") == todo_id:
                    todos[index] = payload
                    replaced = True
                    break
            if not replaced:
                todos.append(payload)
            state["todos"] = todos
        elif event_type == "tool_call":
            state["last_tool"] = payload
        elif event_type == "error":
            errors = list(state.get("errors", []))
            errors.append(str(payload.get("error") or payload.get("message") or payload))
            state["errors"] = errors

        history.append({
            "step": event["id"],
            "node": _node_name(event_type, payload),
            "state": copy.deepcopy(state),
            "interrupt": None,
            "timestamp": event["created_at"],
        })
    return history


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    try:
        payload = json.loads(row["payload_json"] or "{}")
    except json.JSONDecodeError:
        payload = {}
    return {
        "id": row["id"],
        "thread_id": row["thread_id"],
        "source": row["source"],
        "event_type": row["event_type"],
        "payload": payload,
        "created_at": row["created_at"],
    }


def _node_name(event_type: str, payload: dict[str, Any]) -> str:
    if event_type == "user_message":
        return "input"
    if event_type == "artifact_created":
        return str(payload.get("kind") or payload.get("artifact", {}).get("kind") or "content")
    if event_type == "assistant_message":
        return "deliver"
    if event_type == "summary_created":
        return "summary"
    return event_type
