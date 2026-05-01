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

_DOCXXML_CAPABILITIES = """DocxXML 能力使用规则：
- @人：结构化重点中出现 cite_users.user_id 时，必须复用 <cite type="user" user-id="..."></cite>；不知道用户 ID 时，用普通姓名文本，不要编造 user-id。
- @文档：结构化重点中出现 doc_id 时，必须复用 <cite type="doc" doc-id="..."></cite>。只有 URL 时优先使用 <a href="...">...</a> 或 <bookmark name="..." href="..."></bookmark>。
- 白板：结构化重点中有 whiteboards 时，必须在总结中保留一个白板；已有 token 可用 <whiteboard token="..."></whiteboard>，否则用 <whiteboard type="mermaid">...</whiteboard>。
- 思维导图：优先使用 Mermaid mindmap 语法放入 whiteboard；只在内容适合树状主题拆解时使用。
- 流程图/架构图：优先使用 Mermaid flowchart 语法放入 whiteboard；节点文字保持简短。
- 不要生成 <whiteboard token="...">，除非用户明确提供已有 whiteboard token；新建画板必须使用 type="mermaid"、type="plantuml" 或 type="blank"。
- 图片：结构化重点中出现 images.href 时，可以复用 <img href="..." caption="..."/>；没有可访问图片 URL 时不要编造 src、token、内部下载链接。
- 分栏：结构化重点中出现 grids 时，可复用 <grid><column>...</column></grid> 的分栏呈现方式，尤其适合“左文字、右图片/图示”。
- 链接：结构化重点或用户输入中出现 URL 时，必须放入“相关链接”。普通链接用 <a href="...">标题</a>；重要资源用 <bookmark name="标题" href="..."></bookmark>。
- 待办：结构化重点中出现 checkboxes 时，必须参考其 done 状态和 user_ids；文档内任务清单用 <checkbox done="false">...</checkbox>，不要使用 Markdown 复选框。
- 提醒时间：只有用户给出明确时间戳或可确定时间时，才使用 <time>。
- 复杂信息优先用 h1/h2、ul/ol、table、callout、blockquote、checkbox、whiteboard 组合表达，保持合法 XML。"""

_MEETING_MINUTES_TEMPLATE = """会议纪要模板（必须输出 DocxXML）：
- 适用条件：用户要求生成会议纪要、会议记录、会议总结、会后整理，或材料明显来自会议讨论。
- 禁止输出 Markdown：不得使用 # 标题、**加粗**、Markdown 表格、代码块或纯文本清单替代 XML 标签。
- 标题：<title>智能纪要：会议主题 日期</title>，日期缺失时使用“会议纪要：主题”。
- 标题后第一个 blockquote 必须包含 3 个独立段落：<p>会议主题：...</p><p>会议时间：...</p><p>参会人：...</p>；不要把三项合并到一个 p，不要用 br，不要给字段名加粗。
- 标题后第二个 blockquote 只包含一段固定文本：<p>智能会议纪要由 AI 生成，可能不准确，请谨慎甄别后使用</p>；不要加 emoji，不要加粗。
- 一级标题只允许这三个且顺序固定：<h1>总结</h1>、<h1>待办</h1>、<h1>相关链接</h1>。
- 总结标题后先写一个 p，概括会议目标、讨论范围、主要结论和后续方向。
- 总结正文使用多层 ul：一级 li 是主题模块，模块标题用 <b>...</b>；主题名不要加 emoji。二级 li 是议题，三级 li 写事实、决策、分歧、风险、时间节点和负责人。
- 总结不得使用表格替代多层列表；每个主题模块控制在 2 到 5 个要点，避免短句堆叠。
- 如果结构化重点中有 whiteboards，必须在总结标题后保留一个白板；如果材料能提取关键结构图或流程关系，可以生成 <whiteboard type="mermaid">...</whiteboard>；不能确认结构时不要生成新 whiteboard。
- 不要生成 img，除非结构化重点或用户材料中给出可用图片 URL。
- 待办：只使用 <checkbox done="false">...</checkbox>；每项写“任务名：具体事项（负责人：姓名或负责人未提及）”。已完成事项可用 done="true"。
- 相关链接：用 ul 嵌套展示链接来源和链接项；链接必须来自用户材料、用户输入中的 URL 或引用文档。没有链接时写 <p>未提及相关链接</p>。
- 全文保持客观纪要风格，不添加无依据事实。"""


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
        "如用户要求生成会议纪要，必须使用 system prompt 中的会议纪要模板。\n"
        "如用户原始请求中包含“生成约束上下文”，其中的结构化字段 JSON 和生成要求必须落实到文档 XML。\n"
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
- 如果是会议纪要，必须使用会议纪要模板，且输出 DocxXML，不得输出 Markdown
- 如果用户原始请求包含“生成约束上下文”，必须把其中的 cite_users、whiteboards、images、checkboxes、links、grids 转化为对应 DocxXML 标签
- 若 links 或用户输入 URL 非空，相关链接不得写“未提及相关链接”
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
- 如用户要求生成会议纪要，必须使用下方会议纪要模板。
- 用户消息中如果出现“生成约束上下文”，它不是普通参考资料；其中“结构化字段 JSON”和“生成要求”是本次输出的硬性输入。
- 生成会议纪要时，参会人、待办负责人、白板、图片、分栏和相关链接优先来自结构化字段 JSON。
- 输出前检查：结构化字段 JSON 中非空的 cite_users、whiteboards、checkboxes、links、grids，应在最终 DocxXML 中出现对应 <cite>、<whiteboard>、<checkbox>、<a>/<bookmark>、<grid> 标签；没有可用图片 URL 时才省略 <img>。

当前格式：DocxXML

{_DOCXXML_CAPABILITIES}

{_MEETING_MINUTES_TEMPLATE}

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
- 默认使用 DocxXML；会议纪要始终使用 DocxXML。
- 仅当用户明确要求 Markdown 且不是会议纪要时，才使用 Markdown。
- 生成内容后必须调用 create_doc_artifact。
- create_doc_artifact 是当前环境的飞书文档创建工具；不要直接调用 lark-cli。
- 工具调用完成后，只返回简洁结果，包含标题、状态和链接。
- 不得把准备动作作为最终回复；最终回复只允许出现在 create_doc_artifact 工具调用之后。
- 不要再次读取 skill 文件；下方资料就是本次任务所需的完整规则。
- 如用户要求生成会议纪要，必须使用下方会议纪要模板。
- 用户消息中如果出现“生成约束上下文”，它不是普通参考资料；其中“结构化字段 JSON”和“生成要求”是本次输出的硬性输入。
- 生成会议纪要时，参会人、待办负责人、白板、图片、分栏和相关链接优先来自结构化字段 JSON。
- 输出前检查：结构化字段 JSON 中非空的 cite_users、whiteboards、checkboxes、links、grids，应在最终 DocxXML 中出现对应 <cite>、<whiteboard>、<checkbox>、<a>/<bookmark>、<grid> 标签；没有可用图片 URL 时才省略 <img>。

当前格式：{"Markdown" if markdown else "DocxXML"}

{_DOCXXML_CAPABILITIES}

{_MEETING_MINUTES_TEMPLATE}

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
