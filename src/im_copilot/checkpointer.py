import os

from langgraph.checkpoint.memory import InMemorySaver


def get_checkpointer(checkpointer_type: str | None = None):
    """Factory for LangGraph checkpointers.

    Args:
        checkpointer_type: "memory" | None (auto from env)

    Returns:
        A LangGraph checkpointer instance.
    """
    cp_type = checkpointer_type or os.getenv("CHECKPOINTER_TYPE", "memory")

    if cp_type == "memory":
        return InMemorySaver()

    raise ValueError(f"Unsupported checkpointer type: {cp_type}")
