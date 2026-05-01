from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_COMMAND_PATTERN = re.compile(r"^/(\w+)(?:\s+(.*))?$", re.DOTALL)


@dataclass
class CommandResult:
    handled: bool
    command: str
    response_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_command(text: str) -> tuple[str, str] | None:
    text = text.strip()
    match = _COMMAND_PATTERN.match(text)
    if not match:
        return None
    return match.group(1).lower(), (match.group(2) or "").strip()


def execute_command(
    command: str,
    args: str,
    chat_id: str,
    thread_id: str,
    source: str,
    user_id: str = "",
) -> CommandResult:
    handler = _COMMAND_REGISTRY.get(command)
    if handler is None:
        return CommandResult(
            handled=False,
            command=command,
            response_text=f"未知命令: /{command}。输入 /help 查看可用命令。",
        )
    return handler(args, chat_id, thread_id, source, user_id)


def _handle_new(args: str, chat_id: str, thread_id: str, source: str, user_id: str = "") -> CommandResult:
    return CommandResult(
        handled=True,
        command="new",
        response_text="已开始新对话。",
        metadata={"action": "reset_thread"},
    )


def _handle_help(args: str, chat_id: str, thread_id: str, source: str, user_id: str = "") -> CommandResult:
    help_text = (
        "可用命令：\n"
        "/new - 开始新对话（重置当前会话上下文）\n"
        "/todo - 查看待办\n"
        "/todo done <id> - 标记完成\n"
        "/todo delete <id> - 删除待办\n"
        "/todo summary today - 查看今日群聊任务摘要\n"
        "/todo sync - 从今日已收到群消息重新提取待办\n"
        "/help - 显示此帮助信息"
    )
    return CommandResult(
        handled=True,
        command="help",
        response_text=help_text,
    )


def _handle_todo(args: str, chat_id: str, thread_id: str, source: str, user_id: str = "") -> CommandResult:
    from im_copilot.memory.chat_sync import sync_today
    from im_copilot.memory.summary_worker import summary_today
    from im_copilot.memory.todo_store import todo_store

    parts = args.split()
    action = parts[0].lower() if parts else "list"

    if action == "done" and len(parts) >= 2:
        if not user_id:
            return CommandResult(True, "todo", "无法识别当前用户。")
        if not parts[1].isdigit():
            return CommandResult(True, "todo", "待办 ID 无效。")
        ok = todo_store.mark_done(int(parts[1]), user_id)
        return CommandResult(True, "todo", "已标记完成。" if ok else "未找到可操作的待办。")

    if action == "delete" and len(parts) >= 2:
        if not user_id:
            return CommandResult(True, "todo", "无法识别当前用户。")
        if not parts[1].isdigit():
            return CommandResult(True, "todo", "待办 ID 无效。")
        ok = todo_store.delete(int(parts[1]), user_id)
        return CommandResult(True, "todo", "已删除。" if ok else "未找到可操作的待办。")

    if action == "summary" and len(parts) >= 2 and parts[1].lower() == "today":
        return CommandResult(True, "todo", summary_today(chat_id))

    if action == "sync":
        count = sync_today(chat_id)
        return CommandResult(True, "todo", f"已从今日本地消息提取 {count} 个待办。")

    if not user_id:
        return CommandResult(True, "todo", "无法识别当前用户。")
    todos = todo_store.list(assignee_open_id=user_id, chat_id="" if source == "web" else chat_id)
    if not todos:
        return CommandResult(True, "todo", "暂无待办。")
    lines = ["待办："]
    for todo in todos:
        lines.append(f"{todo.id}. {todo.title}｜截止 {todo.due_at}")
    return CommandResult(True, "todo", "\n".join(lines))


_COMMAND_REGISTRY: dict[str, Any] = {
    "new": _handle_new,
    "help": _handle_help,
    "todo": _handle_todo,
}
