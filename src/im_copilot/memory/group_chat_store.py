from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass

_DDL = """
CREATE TABLE IF NOT EXISTS group_chats (
    chat_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'normal',
    external INTEGER NOT NULL DEFAULT 0,
    tenant_key TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    last_message_create_time_ms INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_group_chats_status
    ON group_chats(status, updated_at);
"""


@dataclass
class GroupChat:
    chat_id: str
    name: str
    status: str
    external: bool
    tenant_key: str
    source: str
    last_message_create_time_ms: int
    created_at: float
    updated_at: float


def _db_path() -> str:
    return os.getenv("GROUP_CHAT_DB", os.getenv("TODO_DB", os.getenv("AGENT_EVENTS_DB", ".copilot_checkpoints.sqlite")))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    return conn


class GroupChatStore:
    def upsert(
        self,
        *,
        chat_id: str,
        name: str = "",
        status: str = "normal",
        external: bool = False,
        tenant_key: str = "",
        source: str = "",
        initial_cursor_ms: int = 0,
    ) -> GroupChat | None:
        if not chat_id:
            return None
        now = time.time()
        with _conn() as conn:
            conn.execute(
                """INSERT INTO group_chats
                   (chat_id, name, status, external, tenant_key, source,
                    last_message_create_time_ms, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chat_id) DO UPDATE SET
                     name = COALESCE(NULLIF(excluded.name, ''), group_chats.name),
                     status = excluded.status,
                     external = excluded.external,
                     tenant_key = COALESCE(NULLIF(excluded.tenant_key, ''), group_chats.tenant_key),
                     source = excluded.source,
                     updated_at = excluded.updated_at""",
                (
                    chat_id,
                    name,
                    status,
                    1 if external else 0,
                    tenant_key,
                    source,
                    initial_cursor_ms,
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM group_chats WHERE chat_id = ?", (chat_id,)).fetchone()
        return _row_to_chat(row) if row else None

    def active(self) -> list[GroupChat]:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT * FROM group_chats WHERE status = 'normal' ORDER BY updated_at DESC, chat_id ASC"
            ).fetchall()
        return [_row_to_chat(row) for row in rows]

    def update_cursor(self, chat_id: str, last_message_create_time_ms: int) -> None:
        if not chat_id or last_message_create_time_ms <= 0:
            return
        now = time.time()
        with _conn() as conn:
            conn.execute(
                "UPDATE group_chats SET last_message_create_time_ms = MAX(last_message_create_time_ms, ?), "
                "updated_at = ? WHERE chat_id = ?",
                (last_message_create_time_ms, now, chat_id),
            )


def _row_to_chat(row: sqlite3.Row) -> GroupChat:
    return GroupChat(
        chat_id=row["chat_id"],
        name=row["name"],
        status=row["status"],
        external=bool(row["external"]),
        tenant_key=row["tenant_key"],
        source=row["source"],
        last_message_create_time_ms=int(row["last_message_create_time_ms"] or 0),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


group_chat_store = GroupChatStore()
