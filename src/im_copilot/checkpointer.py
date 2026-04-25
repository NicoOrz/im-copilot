import os
from contextlib import contextmanager

from langgraph.checkpoint.memory import InMemorySaver


@contextmanager
def get_checkpointer(checkpointer_type: str | None = None):
    """Factory for LangGraph checkpointers.

    Args:
        checkpointer_type: "memory" | "sqlite" | None (auto from env)

    Yields:
        A LangGraph checkpointer instance.
    """
    cp_type = checkpointer_type or os.getenv("CHECKPOINTER_TYPE", "memory")

    if cp_type == "sqlite":
        # Lazy import so the optional dependency is only required when used.
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = os.getenv("CHECKPOINTER_DB", ".copilot_checkpoints.sqlite")
        with SqliteSaver.from_conn_string(db_path) as saver:
            yield saver
    else:
        yield InMemorySaver()
