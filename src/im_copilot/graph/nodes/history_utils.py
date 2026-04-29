from __future__ import annotations


def format_history(
    history: list[dict],
    recent_n: int = 6,
    summary_n: int = 10,
) -> str:
    if not history:
        return "（无历史对话）"

    total = len(history)
    parts: list[str] = []

    older_start = max(0, total - recent_n - summary_n)
    older_end = max(0, total - recent_n)
    for t in history[older_start:older_end]:
        content = t["content"][:50] + ("..." if len(t["content"]) > 50 else "")
        parts.append(f"[{t['role']}]: {content}")

    for t in history[max(0, total - recent_n):]:
        parts.append(f"{t['role']}: {t['content']}")

    return "\n".join(parts)
