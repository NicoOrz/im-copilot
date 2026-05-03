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
        "/todo board - 查看群共享看板\n"
        "/todo done <序号> - 标记完成\n"
        "/todo delete <序号> - 删除待办\n"
        "/todo sync - 从今日已收到群消息重新提取待办\n"
        "/logout - 清除当前用户授权（单聊）\n"
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

    if action == "board":
        return CommandResult(True, "todo", summary_today(chat_id))

    if action == "sync":
        count = sync_today(chat_id)
        return CommandResult(True, "todo", f"已从今日本地消息提取 {count} 个待办。")

    if source == "feishu_group":
        return CommandResult(True, "todo", "个人待办仅在单聊中展示。群内可使用 /todo board 查看共享看板。")

    if action == "done" and len(parts) >= 2:
        if not user_id:
            return CommandResult(True, "todo", "无法识别当前用户。")
        if not parts[1].isdigit():
            return CommandResult(True, "todo", "待办序号无效。")
        todos = todo_store.list(assignee_open_id=user_id, chat_id="")
        todo_id = _resolve_todo_id(parts[1], todos)
        ok = todo_store.mark_done(todo_id, user_id)
        return CommandResult(True, "todo", "已标记完成。" if ok else "未找到可操作的待办。")

    if action == "delete" and len(parts) >= 2:
        if not user_id:
            return CommandResult(True, "todo", "无法识别当前用户。")
        if not parts[1].isdigit():
            return CommandResult(True, "todo", "待办序号无效。")
        todos = todo_store.list(assignee_open_id=user_id, chat_id="")
        todo_id = _resolve_todo_id(parts[1], todos)
        ok = todo_store.delete(todo_id, user_id)
        return CommandResult(True, "todo", "已删除。" if ok else "未找到可操作的待办。")

    if not user_id:
        return CommandResult(True, "todo", "无法识别当前用户。")
    todos = todo_store.list(assignee_open_id=user_id, chat_id="")
    if not todos:
        return CommandResult(True, "todo", "暂无待办。")
    lines = ["待办："]
    for index, todo in enumerate(todos, start=1):
        lines.append(f"{index}. {todo.title}｜截止 {todo.due_at}")
    return CommandResult(True, "todo", "\n".join(lines))


def _handle_logout(args: str, chat_id: str, thread_id: str, source: str, user_id: str = "") -> CommandResult:
    from im_copilot.user_token_store import token_store

    if source == "feishu_group":
        return CommandResult(True, "logout", "请在单聊中使用 /logout。")
    if not user_id:
        return CommandResult(True, "logout", "无法识别当前用户。")
    removed = token_store.delete(user_id)
    if removed:
        return CommandResult(True, "logout", "已清除当前授权。请重新发送消息并按提示授权。")
    return CommandResult(True, "logout", "当前没有可清除的授权。")


def _resolve_todo_id(value: str, todos: list[Any]) -> int:
    number = int(value)
    if 1 <= number <= len(todos):
        return int(todos[number - 1].id)
    return number


_COMMAND_REGISTRY: dict[str, Any] = {
    "new": _handle_new,
    "help": _handle_help,
    "todo": _handle_todo,
    "logout": _handle_logout,
}
