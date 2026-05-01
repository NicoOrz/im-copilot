from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

from im_copilot.deep_agent.tools import build_artifact_tools
from im_copilot.llm import get_llm_for_node

logger = logging.getLogger(__name__)

_LARK_DOC_ROOT = Path(
    os.getenv("IM_COPILOT_LARK_DOC_SKILL_ROOT", "/home/claude_worker/.agents/skills/lark-doc")
)

_DOC_TOOL_NAME = "create_doc_artifact"


class ForceDocArtifactToolMiddleware(AgentMiddleware):
    name = "ForceDocArtifactToolMiddleware"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if _has_doc_tool_call(request.messages):
            return handler(request)
        logger.info("force_doc_tool_choice enabled tool=%s", _DOC_TOOL_NAME)
        return handler(request.override(tool_choice=_DOC_TOOL_NAME))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if _has_doc_tool_call(request.messages):
            return await handler(request)
        logger.info("force_doc_tool_choice enabled tool=%s", _DOC_TOOL_NAME)
        return await handler(request.override(tool_choice=_DOC_TOOL_NAME))


def _has_doc_tool_call(messages: list[Any]) -> bool:
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == _DOC_TOOL_NAME:
            return True
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                if tool_call.get("name") == _DOC_TOOL_NAME:
                    return True
    return False


def build_doc_agent(
    *,
    thread_id: str,
    source: str,
    user_access_token: str = "",
    artifacts: dict[str, dict[str, Any]] | None = None,
    markdown: bool = False,
) -> Any:
    logger.info(
        "build_doc_agent start thread_id=%s source=%s markdown=%s has_user_token=%s skill_root=%s skill_root_exists=%s",
        thread_id,
        source,
        markdown,
        bool(user_access_token),
        _LARK_DOC_ROOT,
        _LARK_DOC_ROOT.exists(),
    )
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:
        logger.exception("build_doc_agent import failed thread_id=%s", thread_id)
        raise RuntimeError("deepagents dependency is not installed") from exc

    doc_tool = build_artifact_tools(
        thread_id=thread_id,
        source=source,
        user_access_token=user_access_token,
        artifacts=artifacts,
    )[0]

    kwargs: dict[str, Any] = {}
    try:
        from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

        logger.debug("build_doc_agent backend composite enabled thread_id=%s", thread_id)
        kwargs["backend"] = CompositeBackend(
            default=StateBackend(),
            routes={
                "/.agents/skills/": FilesystemBackend(
                    root_dir="/home/claude_worker/.agents/skills",
                    virtual_mode=True,
                )
            },
        )
    except Exception:
        logger.exception("build_doc_agent backend setup failed thread_id=%s", thread_id)
        pass

    agent = create_deep_agent(
        name="im-copilot-doc-agent",
        model=get_llm_for_node("doc"),
        tools=[doc_tool],
        system_prompt=_doc_system_prompt(markdown=markdown),
        middleware=[ForceDocArtifactToolMiddleware()],
        **kwargs,
    )
    logger.info("build_doc_agent complete thread_id=%s", thread_id)
    return agent


def build_doc_task_message(message: str, *, markdown: bool = False) -> str:
    doc_format = "Markdown" if markdown else "DocxXML"
    task_message = (
        "请根据用户原始请求生成完整飞书文档，并调用 create_doc_artifact 创建产物。\n"
        "原始 lark-doc 规则已经在 system prompt 中提供，直接生成内容并调用工具。\n"
        f"输出格式：{doc_format}\n\n"
        "用户原始请求：\n"
        f"{message}"
    )
    logger.debug(
        "build_doc_task_message markdown=%s raw_len=%s task_len=%s",
        markdown,
        len(message),
        len(task_message),
    )
    return task_message


def generate_doc_content(message: str) -> str:
    prompt = f"""{_doc_generation_prompt()}

用户原始请求：
{message}

任务：
- 生成完整 DocxXML 内容
- 忠实反映用户材料的关键信息，不添加无依据事实
- DocxXML 必须包含唯一 <title>
- 必须输出合法 XML 标签结构，标签本身不要转义
- 只输出文档内容，不要输出说明，不要使用代码块
"""
    content = get_llm_for_node("doc").invoke(prompt).content
    return _strip_code_fence(_content_to_text(content)).strip()


def _doc_generation_prompt() -> str:
    context = _lark_doc_context(markdown=False)
    logger.info("doc_generation_prompt built doc_format=xml context_len=%s", len(context))
    return f"""你是 IM Copilot 的文档内容生成器。始终用中文输出。

职责：
- 直接遵循下方原始 lark-doc skill 创建规则。
- 根据用户材料生成完整文档内容，不添加无依据事实。
- 固定使用 DocxXML，即使用户提到 Markdown 也输出 DocxXML。
- 当前代码会负责创建飞书文档；你只负责输出文档内容。

当前格式：DocxXML

以下是原始 lark-doc skill 与创建文档必需参考资料，已经预读给你：

{context}
"""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return str(content or "")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _doc_system_prompt(*, markdown: bool = False) -> str:
    context = _lark_doc_context(markdown=markdown)
    logger.info("doc_system_prompt built markdown=%s context_len=%s", markdown, len(context))
    return f"""你是 IM Copilot 的文档 Deep Agent。始终用中文回答用户。

职责：
- 直接遵循下方原始 lark-doc skill 创建规则。
- 根据用户材料生成完整文档内容，不添加无依据事实。
- 默认使用 DocxXML；用户明确要求 Markdown 时使用 Markdown。
- 生成内容后必须调用 create_doc_artifact。
- create_doc_artifact 是当前环境的飞书文档创建工具；不要直接调用 lark-cli。
- 工具调用完成后，只返回简洁结果，包含标题、状态和链接。
- 不得把准备动作作为最终回复；最终回复只允许出现在 create_doc_artifact 工具调用之后。
- 不要再次读取 skill 文件；下方资料就是本次任务所需的完整规则。

当前格式：{"Markdown" if markdown else "DocxXML"}

以下是原始 lark-doc skill 与创建文档必需参考资料，已经预读给你：

{context}
"""


def _lark_doc_context(*, markdown: bool = False) -> str:
    paths = [
        _LARK_DOC_ROOT / "SKILL.md",
        _LARK_DOC_ROOT / "references" / "style" / "lark-doc-create-workflow.md",
        _LARK_DOC_ROOT / "references" / "style" / "lark-doc-style.md",
        _LARK_DOC_ROOT / "references" / ("lark-doc-md.md" if markdown else "lark-doc-xml.md"),
    ]
    parts = []
    for path in paths:
        parts.append(f"## {path}\n{_read_text(path)}")
    context = "\n\n".join(parts)
    logger.info(
        "lark_doc_context loaded markdown=%s files=%s total_chars=%s",
        markdown,
        len(paths),
        len(context),
    )
    return context


def _read_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
        logger.debug("lark_doc_context file loaded path=%s chars=%s", path, len(text))
        return text
    except OSError as exc:
        logger.warning("lark_doc_context file read failed path=%s error=%s", path, exc)
        return f"无法读取 {path}: {exc}"
