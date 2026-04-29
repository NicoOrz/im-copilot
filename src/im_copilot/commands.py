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
) -> CommandResult:
    handler = _COMMAND_REGISTRY.get(command)
    if handler is None:
        return CommandResult(
            handled=False,
            command=command,
            response_text=f"未知命令: /{command}。输入 /help 查看可用命令。",
        )
    return handler(args, chat_id, thread_id, source)


def _handle_new(args: str, chat_id: str, thread_id: str, source: str) -> CommandResult:
    return CommandResult(
        handled=True,
        command="new",
        response_text="已开始新对话。",
        metadata={"action": "reset_thread"},
    )


def _handle_help(args: str, chat_id: str, thread_id: str, source: str) -> CommandResult:
    help_text = (
        "可用命令：\n"
        "/new - 开始新对话（重置当前会话上下文）\n"
        "/help - 显示此帮助信息"
    )
    return CommandResult(
        handled=True,
        command="help",
        response_text=help_text,
    )


_COMMAND_REGISTRY: dict[str, Any] = {
    "new": _handle_new,
    "help": _handle_help,
}
