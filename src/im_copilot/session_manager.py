"""Session manager for async LangGraph pipeline sessions in the Feishu bot.

Since Feishu is async, when the pipeline hits an ``interrupt()`` we save the
session state and resume later when the user clicks a card button.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class SessionManager:
    """Thread-safe singleton-like manager for LangGraph pipeline sessions."""

    _instance: SessionManager | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> SessionManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions: dict[str, dict[str, Any]] = {}
                    cls._instance._sessions_lock = threading.Lock()
        return cls._instance

    def create_session(
        self,
        thread_id: str,
        graph,
        config: dict,
        card_id: str | None = None,
        card_entity_id: str | None = None,
        chat_id: str | None = None,
    ) -> dict:
        """Save a new session and return its data.

        Args:
            thread_id: Unique identifier for the conversation thread.
            graph: A compiled LangGraph (``CompiledStateGraph``).
            config: LangGraph config, e.g. ``{"configurable": {"thread_id": ...}}``.
            card_id: Optional Feishu card message ID for later updates.
            card_entity_id: Optional Feishu card entity ID for streaming updates.

        Returns:
            The session dictionary.
        """
        session = {
            "thread_id": thread_id,
            "chat_id": chat_id or thread_id,
            "graph": graph,
            "config": config,
            "card_id": card_id,
            "card_message_id": card_id,
            "card_entity_id": card_entity_id,
            "last_interrupt": None,
            "sequence": 0,
            "created_at": time.time(),
        }
        with self._sessions_lock:
            self._sessions[thread_id] = session
        return session

    def get_session(self, thread_id: str) -> dict | None:
        """Retrieve an active session by *thread_id*."""
        with self._sessions_lock:
            return self._sessions.get(thread_id)

    def update_session(self, thread_id: str, **kwargs) -> dict | None:
        """Update fields of an existing session.

        Returns the updated session, or ``None`` if the session does not exist.
        """
        with self._sessions_lock:
            session = self._sessions.get(thread_id)
            if session is None:
                return None
            for key, value in kwargs.items():
                if key in session:
                    session[key] = value
            return session

    def resume_session(self, thread_id: str, decision: dict) -> dict:
        """Resume a paused graph with a user decision.

        Internally imports ``Command`` from ``langgraph.types`` to avoid circular
        imports at module load time.

        Args:
            thread_id: The session to resume.
            decision: Payload passed to ``Command(resume=decision)``.

        Returns:
            The result of ``graph.invoke(...)``.

        Raises:
            KeyError: If the session does not exist.
        """
        from langgraph.types import Command

        with self._sessions_lock:
            session = self._sessions.get(thread_id)
        if session is None:
            raise KeyError(f"Session not found for thread_id: {thread_id}")

        graph = session["graph"]
        config = session["config"]
        return graph.invoke(Command(resume=decision), config=config)

    def delete_session(self, thread_id: str) -> None:
        """Remove a session from the manager."""
        with self._sessions_lock:
            self._sessions.pop(thread_id, None)

    def list_sessions(self) -> list[dict]:
        """Return a shallow copy of all active sessions."""
        with self._sessions_lock:
            return list(self._sessions.values())


# Module-level singleton instance
session_manager = SessionManager()
