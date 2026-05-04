"""SQLite-backed store for mapping users to their conversation sessions."""

from __future__ import annotations

import logging
import os
import sqlite3
import time

logger = logging.getLogger(__name__)

_DB_PATH = os.environ.get("USER_SESSION_DB", ".user_sessions.sqlite")

_DDL = """
CREATE TABLE IF NOT EXISTS user_sessions (
    open_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    chat_id TEXT,
    source TEXT NOT NULL DEFAULT 'web',
    created_at REAL NOT NULL,
    display_name TEXT,
    PRIMARY KEY (open_id, thread_id)
);
CREATE INDEX IF NOT EXISTS idx_user_sessions_open_id ON user_sessions(open_id);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            c.execute(stmt)
    c.commit()
    return c


class UserSessionStore:

    def record_session(
        self,
        open_id: str,
        thread_id: str,
        source: str = "web",
        chat_id: str | None = None,
        display_name: str | None = None,
    ) -> None:
        if not open_id or not thread_id:
            return
        with _conn() as c:
            c.execute(
                """INSERT INTO user_sessions (open_id, thread_id, chat_id, source, created_at, display_name)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(open_id, thread_id) DO UPDATE SET
                     chat_id = COALESCE(excluded.chat_id, chat_id),
                     display_name = COALESCE(excluded.display_name, display_name)""",
                (open_id, thread_id, chat_id, source, time.time(), display_name),
            )

    def list_sessions(self, open_id: str) -> list[dict]:
        with _conn() as c:
            rows = c.execute(
                "SELECT thread_id, chat_id, source, created_at, display_name "
                "FROM user_sessions WHERE open_id = ? ORDER BY created_at DESC",
                (open_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def has_session(self, open_id: str, thread_id: str) -> bool:
        if not open_id or not thread_id:
            return False
        with _conn() as c:
            row = c.execute(
                "SELECT 1 FROM user_sessions WHERE open_id = ? AND thread_id = ?",
                (open_id, thread_id),
            ).fetchone()
        return row is not None

    def find_thread_by_chat(self, open_id: str, chat_id: str) -> str | None:
        with _conn() as c:
            row = c.execute(
                "SELECT thread_id FROM user_sessions WHERE open_id = ? AND chat_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (open_id, chat_id),
            ).fetchone()
        return row["thread_id"] if row else None

    def delete_session(self, open_id: str, thread_id: str) -> None:
        with _conn() as c:
            c.execute(
                "DELETE FROM user_sessions WHERE open_id = ? AND thread_id = ?",
                (open_id, thread_id),
            )


session_store = UserSessionStore()
