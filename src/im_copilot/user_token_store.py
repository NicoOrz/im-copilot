"""SQLite-backed store for per-user Feishu user_access_token."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DB_PATH = os.environ.get("USER_TOKEN_DB", ".user_tokens.sqlite")

_DDL = """
CREATE TABLE IF NOT EXISTS user_tokens (
    open_id TEXT PRIMARY KEY,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    expires_at INTEGER NOT NULL
)
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(_DDL)
    c.commit()
    return c


class UserTokenStore:
    """Thread-safe SQLite store for user_access_token per open_id."""

    def get(self, open_id: str) -> str | None:
        """Return a valid access_token, refreshing if needed. None if not found."""
        with _conn() as c:
            row = c.execute(
                "SELECT access_token, refresh_token, expires_at FROM user_tokens WHERE open_id = ?",
                (open_id,),
            ).fetchone()
        if row is None:
            return None
        # Refresh if expiring within 5 minutes
        if row["expires_at"] - time.time() < 300:
            return self._refresh(open_id, row["refresh_token"])
        return row["access_token"]

    def save(
        self,
        open_id: str,
        access_token: str,
        refresh_token: str | None,
        expires_in: int,
    ) -> None:
        expires_at = int(time.time()) + expires_in
        with _conn() as c:
            c.execute(
                """INSERT INTO user_tokens (open_id, access_token, refresh_token, expires_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(open_id) DO UPDATE SET
                     access_token = excluded.access_token,
                     refresh_token = COALESCE(excluded.refresh_token, refresh_token),
                     expires_at = excluded.expires_at""",
                (open_id, access_token, refresh_token, expires_at),
            )
        logger.info("Saved token for open_id=%s expires_at=%d", open_id, expires_at)

    def delete(self, open_id: str) -> bool:
        with _conn() as c:
            cursor = c.execute("DELETE FROM user_tokens WHERE open_id = ?", (open_id,))
            return cursor.rowcount > 0

    def _refresh(self, open_id: str, refresh_token: str | None) -> str | None:
        if not refresh_token:
            logger.warning("No refresh_token for open_id=%s, re-auth required", open_id)
            return None
        app_id = os.environ.get("FEISHU_APP_ID") or os.environ.get("LARK_APP_ID", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET") or os.environ.get("LARK_APP_SECRET", "")
        try:
            resp = httpx.post(
                "https://open.feishu.cn/open-apis/authen/v2/oauth/token",
                json={
                    "grant_type": "refresh_token",
                    "client_id": app_id,
                    "client_secret": app_secret,
                    "refresh_token": refresh_token,
                },
                timeout=10,
            )
            data: dict[str, Any] = resp.json()
        except Exception as e:
            logger.error("Token refresh failed for open_id=%s: %s", open_id, e)
            return None

        if data.get("code", -1) != 0:
            logger.error("Token refresh API error for open_id=%s: %s", open_id, data)
            return None

        new_uat = data["access_token"]
        new_rt = data.get("refresh_token", refresh_token)
        expires_in = data.get("expires_in", 7200)
        self.save(open_id, new_uat, new_rt, expires_in)
        return new_uat


# Module-level singleton
token_store = UserTokenStore()
