from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Literal

EventType = Literal[
    "user_message",
    "todo_detected",
    "todo_reminded",
    "error",
]

EVENT_TYPES: set[str] = {
    "user_message",
    "todo_detected",
    "todo_reminded",
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
