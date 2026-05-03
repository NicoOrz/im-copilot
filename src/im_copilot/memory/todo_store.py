from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

TodoStatus = Literal["pending", "reminded", "done", "deleted"]

_DDL = """
CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    source_open_id TEXT NOT NULL,
    assignee_open_id TEXT NOT NULL,
    title TEXT NOT NULL,
    action TEXT NOT NULL,
    due_at TEXT NOT NULL,
    remind_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    source_text TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_todos_message_assignee
    ON todos(message_id, assignee_open_id, title);
CREATE INDEX IF NOT EXISTS idx_todos_assignee_status
    ON todos(assignee_open_id, status, remind_at);
CREATE INDEX IF NOT EXISTS idx_todos_chat_status
    ON todos(chat_id, status, created_at);
"""


@dataclass
class TodoRecord:
    id: int
    chat_id: str
    message_id: str
    source_open_id: str
    assignee_open_id: str
    title: str
    action: str
    due_at: str
    remind_at: str
    status: TodoStatus
    source_text: str
    created_at: float
    updated_at: float


def _db_path() -> str:
    return os.getenv("TODO_DB", os.getenv("AGENT_EVENTS_DB", ".copilot_checkpoints.sqlite"))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    return conn


class TodoStore:
    def create(
        self,
        *,
        chat_id: str,
        message_id: str,
        source_open_id: str,
        assignee_open_id: str,
        title: str,
        action: str,
        due_at: str,
        remind_at: str,
        source_text: str,
    ) -> TodoRecord | None:
        now = time.time()
        with _conn() as conn:
            existing = conn.execute(
                """SELECT * FROM todos
                   WHERE assignee_open_id = ?
                     AND title = ?
                     AND due_at = ?
                     AND status IN ('pending', 'reminded')
                   ORDER BY id ASC
                   LIMIT 1""",
                (assignee_open_id, title, due_at),
            ).fetchone()
            if existing:
                return None
            cursor = conn.execute(
                """INSERT OR IGNORE INTO todos
                   (chat_id, message_id, source_open_id, assignee_open_id, title, action,
                    due_at, remind_at, status, source_text, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
                (
                    chat_id,
                    message_id,
                    source_open_id,
                    assignee_open_id,
                    title,
                    action,
                    due_at,
                    remind_at,
                    source_text,
                    now,
                    now,
                ),
            )
            if cursor.rowcount == 0:
                return None
            row = conn.execute(
                "SELECT * FROM todos WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return _row_to_todo(row) if row else None

    def list(
        self,
        *,
        assignee_open_id: str = "",
        chat_id: str = "",
        status: str = "pending",
    ) -> list[TodoRecord]:
        clauses = []
        params: list[str] = []
        if assignee_open_id:
            clauses.append("assignee_open_id = ?")
            params.append(assignee_open_id)
        if chat_id:
            clauses.append("chat_id = ?")
            params.append(chat_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with _conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM todos {where} ORDER BY remind_at ASC, id ASC",
                params,
            ).fetchall()
        return _dedupe_todos([_row_to_todo(row) for row in rows])

    def mark_done(self, todo_id: int, assignee_open_id: str = "") -> bool:
        return self._set_status(todo_id, "done", assignee_open_id, include_duplicates=True)

    def delete(self, todo_id: int, assignee_open_id: str = "") -> bool:
        return self._set_status(todo_id, "deleted", assignee_open_id, include_duplicates=True)

    def clear(self) -> int:
        with _conn() as conn:
            cursor = conn.execute("DELETE FROM todos")
            return max(cursor.rowcount, 0)

    def mark_reminded(self, todo_id: int) -> bool:
        return self._set_status(todo_id, "reminded", include_duplicates=True)

    def due_for_reminder(self, now: datetime) -> list[TodoRecord]:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT * FROM todos WHERE status = 'pending' AND remind_at <= ? "
                "ORDER BY remind_at ASC, id ASC",
                (now.isoformat(timespec="minutes"),),
            ).fetchall()
        return _dedupe_todos([_row_to_todo(row) for row in rows])

    def created_between(self, chat_id: str, start_ts: float, end_ts: float) -> list[TodoRecord]:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT * FROM todos WHERE chat_id = ? AND created_at >= ? AND created_at < ? "
                "ORDER BY id ASC",
                (chat_id, start_ts, end_ts),
            ).fetchall()
        return [_row_to_todo(row) for row in rows]

    def _set_status(
        self,
        todo_id: int,
        status: str,
        assignee_open_id: str = "",
        *,
        include_duplicates: bool = False,
    ) -> bool:
        now = time.time()
        clauses = ["id = ?"]
        params: list[object] = [todo_id]
        if assignee_open_id:
            clauses.append("assignee_open_id = ?")
            params.append(assignee_open_id)
        params = [status, now, *params]
        with _conn() as conn:
            target = conn.execute(
                f"SELECT * FROM todos WHERE {' AND '.join(clauses)}",
                params[2:],
            ).fetchone()
            if not target:
                return False
            if include_duplicates:
                cursor = conn.execute(
                    """UPDATE todos SET status = ?, updated_at = ?
                       WHERE assignee_open_id = ?
                         AND title = ?
                         AND due_at = ?
                         AND status IN ('pending', 'reminded')""",
                    (
                        status,
                        now,
                        target["assignee_open_id"],
                        target["title"],
                        target["due_at"],
                    ),
                )
                return cursor.rowcount > 0
            cursor = conn.execute(
                f"UPDATE todos SET status = ?, updated_at = ? WHERE {' AND '.join(clauses)}",
                params,
            )
            return cursor.rowcount > 0


def _row_to_todo(row: sqlite3.Row) -> TodoRecord:
    return TodoRecord(
        id=row["id"],
        chat_id=row["chat_id"],
        message_id=row["message_id"],
        source_open_id=row["source_open_id"],
        assignee_open_id=row["assignee_open_id"],
        title=row["title"],
        action=row["action"],
        due_at=row["due_at"],
        remind_at=row["remind_at"],
        status=row["status"],
        source_text=row["source_text"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _dedupe_todos(todos: list[TodoRecord]) -> list[TodoRecord]:
    seen: set[tuple[str, str, str]] = set()
    result: list[TodoRecord] = []
    for todo in todos:
        key = (todo.assignee_open_id, todo.title, todo.due_at)
        if key in seen:
            continue
        seen.add(key)
        result.append(todo)
    return result


todo_store = TodoStore()
