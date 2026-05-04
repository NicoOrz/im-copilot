"""WebSocket connection manager for real-time push to WebUI."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self) -> None:
        self._loop = asyncio.get_running_loop()

    async def connect(self, websocket: WebSocket, open_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.setdefault(open_id, []).append(websocket)
        logger.debug("WS connected: open_id=%s total=%d", open_id, len(self._connections.get(open_id, [])))

    async def disconnect(self, websocket: WebSocket, open_id: str) -> None:
        async with self._lock:
            conns = self._connections.get(open_id, [])
            if websocket in conns:
                conns.remove(websocket)
            if not conns:
                self._connections.pop(open_id, None)
        logger.debug("WS disconnected: open_id=%s", open_id)

    async def broadcast_to_user(self, open_id: str, data: dict[str, Any]) -> None:
        async with self._lock:
            conns = list(self._connections.get(open_id, []))
        message = json.dumps(data, ensure_ascii=False)
        stale: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_text(message)
            except Exception:
                stale.append(ws)
        if stale:
            async with self._lock:
                for ws in stale:
                    conns_list = self._connections.get(open_id, [])
                    if ws in conns_list:
                        conns_list.remove(ws)

    def broadcast_to_user_threadsafe(self, open_id: str, data: dict[str, Any]) -> None:
        if self._loop is None or self._loop.is_closed():
            logger.debug("WS broadcast skipped: no active server loop")
            return
        asyncio.run_coroutine_threadsafe(self.broadcast_to_user(open_id, data), self._loop)


ws_manager = ConnectionManager()
