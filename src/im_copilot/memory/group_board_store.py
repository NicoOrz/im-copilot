from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Literal

BoardItemStatus = Literal["open", "pending_confirmation", "confirmed", "done", "deleted"]
BoardItemType = Literal["assignment", "meeting", "decision", "risk", "question", "resource"]

_DDL = """
CREATE TABLE IF NOT EXISTS group_board_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    title TEXT NOT NULL,
    owner_open_id TEXT NOT NULL DEFAULT '',
    owner_name TEXT NOT NULL DEFAULT '',
    due_at TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    source_open_id TEXT NOT NULL,
    source_text TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_group_board_message_type_title
    ON group_board_items(message_id, item_type, title, owner_open_id);
CREATE INDEX IF NOT EXISTS idx_group_board_chat_status
    ON group_board_items(chat_id, status, created_at);
"""


@dataclass
class GroupBoardItem:
    id: int
    chat_id: str
    message_id: str
    item_type: BoardItemType
    title: str
    owner_open_id: str
    owner_name: str
    due_at: str
    status: BoardItemStatus
    source_open_id: str
    source_text: str
    metadata_json: str
    created_at: float
    updated_at: float


def _db_path() -> str:
    return os.getenv("GROUP_BOARD_DB", os.getenv("TODO_DB", os.getenv("AGENT_EVENTS_DB", ".copilot_checkpoints.sqlite")))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    return conn


class GroupBoardStore:
    def create(
        self,
        *,
        chat_id: str,
        message_id: str,
        item_type: BoardItemType,
        title: str,
        source_open_id: str,
        source_text: str,
        owner_open_id: str = "",
        owner_name: str = "",
        due_at: str = "",
        status: BoardItemStatus = "open",
        metadata_json: str = "{}",
    ) -> GroupBoardItem | None:
        now = time.time()
        with _conn() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO group_board_items
                   (chat_id, message_id, item_type, title, owner_open_id, owner_name,
                    due_at, status, source_open_id, source_text, metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chat_id,
                    message_id,
                    item_type,
                    title,
                    owner_open_id,
                    owner_name,
                    due_at,
                    status,
                    source_open_id,
                    source_text,
                    metadata_json,
                    now,
                    now,
                ),
            )
            if cursor.rowcount == 0:
                return None
            row = conn.execute("SELECT * FROM group_board_items WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _row_to_item(row) if row else None

    def list(
        self,
        *,
        chat_id: str,
        status: str = "",
    ) -> list[GroupBoardItem]:
        clauses = []
        params: list[str] = []
        if chat_id:
            clauses.append("chat_id = ?")
            params.append(chat_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with _conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM group_board_items {where} ORDER BY created_at ASC, id ASC",
                params,
            ).fetchall()
        return [_row_to_item(row) for row in rows]

    def created_between(self, chat_id: str, start_ts: float, end_ts: float) -> list[GroupBoardItem]:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT * FROM group_board_items WHERE chat_id = ? AND created_at >= ? AND created_at < ? "
                "ORDER BY id ASC",
                (chat_id, start_ts, end_ts),
            ).fetchall()
        return [_row_to_item(row) for row in rows]


def _row_to_item(row: sqlite3.Row) -> GroupBoardItem:
    return GroupBoardItem(
        id=row["id"],
        chat_id=row["chat_id"],
        message_id=row["message_id"],
        item_type=row["item_type"],
        title=row["title"],
        owner_open_id=row["owner_open_id"],
        owner_name=row["owner_name"],
        due_at=row["due_at"],
        status=row["status"],
        source_open_id=row["source_open_id"],
        source_text=row["source_text"],
        metadata_json=row["metadata_json"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


group_board_store = GroupBoardStore()
