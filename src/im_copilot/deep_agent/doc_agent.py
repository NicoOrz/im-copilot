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

_DOCXXML_CAPABILITIES = """DocxXML 规则：
- 常用标签：title、p、h1-h3、ul、ol、li、table、thead、tbody、tr、th、td、blockquote、callout、checkbox、grid、column、whiteboard、img、a、bookmark、cite、hr、b、code、span。
- XML 标签本身不要转义；仅转义文本内的 <、>、&。
- cite_users.user_id 非空时用 <cite type="user" user-id="..."></cite>，未知 ID 才用普通姓名。
- whiteboards.token 非空时可用 <whiteboard token="..."></whiteboard>；新图示用 <whiteboard type="mermaid">...</whiteboard>。
- images 仅能复用已有 href/src，不要编造图片地址。
- links 非空时使用 <a href="...">...</a> 或 <bookmark name="..." href="..."></bookmark>。
- checkboxes 非空时用 <checkbox done="true|false">...</checkbox>。"""

_MEETING_MINUTES_TEMPLATE = """会议纪要模板（必须输出 DocxXML）：
- 纯文本会议材料优先使用表格化结构：title、概要 callout、基本信息表、已达成共识表、待定事项表、后续 Action。
- 标题：<title>会议纪要：会议主题</title> 或 <title>智能纪要：会议主题 日期</title>。
- 开头：<callout background-color="rgb(240,244,255)" border-color="rgb(130,167,252)" emoji="📝"><p><b>会议概要：</b>时间、参与人、主题、核心结论、待定问题。</p></callout>
- <h1>基本信息</h1> 后用两列表格，列名为“项目 / 内容”，行包含时间、参与人、主题。
- <h1>已达成共识</h1> 后用两列表格，列名为“决议项 / 结论”；没有明确共识时写“未提及”。
- <h1>待定事项</h1> 后可加黄色 callout 提醒分歧，再用三列表格，列名为“事项 / 分歧点 / 涉及方”；没有待定事项时写“未提及”。
- <h1>后续 Action</h1> 后只用 <checkbox done="false">负责人：任务</checkbox>，已完成事项用 done="true"。
- 仅当材料或结构化重点中有链接时，才追加 <h1>相关链接</h1>。"""

_PRD_TEMPLATE = """PRD 模板（用户要求 PRD、产品需求文档、需求方案、需求说明时必须输出 DocxXML）：
- 用户要求 PRD 时优先使用本模板，即使原始材料来自会议、聊天记录或已有文档。
- 标题：<title>PRD：项目或需求名称</title>。
- 一级标题与顺序固定：<h1>文档版本记录</h1>、<h1>一、 项目背景</h1>、<h1>二、 项目目标与评估</h1>、<h1>三、 需求描述</h1>。
- 文档版本记录使用四列表格，列名固定为”文档版本 / 更新日期 / 撰写人 / 说明”；缺失信息写”待完善”。
- 项目背景要说明项目动因、需求来源、当前业务问题、相关数据或调研结论、各利益相关方收益；仅使用材料中可依据的信息。
- 项目目标与评估要写清产品业务目标、项目价值评估方式、可量化指标、指标计算方式；缺失指标写”待完善”，不要编造数值。
- 需求描述必须包含两个二级标题：<h2>1、需求概述</h2>、<h2>2、需求详述</h2>。
- 需求概述用一个段落说明需求要做什么、解决什么业务问题、面向哪些用户或场景。
- 需求详述按材料组织业务流程、交互或视觉逻辑、原型信息、AB 实验方案、埋点需求、历史数据处理、新旧系统兼容、后端模块逻辑和业务规则；没有材料的部分不要展开。
- 不要把模板指导语写入正文；未知信息统一写”待完善”。"""


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
    task_message = (
        "请根据用户原始请求生成完整飞书文档，并调用 create_doc_artifact 创建产物。\n"
        "DocxXML 规则已经在 system prompt 中提供，直接生成内容并调用工具。\n"
        "如用户要求生成 PRD 或产品需求文档，必须使用 system prompt 中的 PRD 模板。\n"
        "如用户要求生成会议纪要，必须使用 system prompt 中的会议纪要模板。\n"
        "如用户原始请求中包含“生成约束上下文”，其中的结构化字段 JSON 和生成要求必须落实到文档 XML。\n"
        "输出格式：DocxXML\n\n"
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


def generate_doc_content(message: str, *, existing_content: str = "") -> str:
    update_section = (
        f"\n以下是现有文档内容（DocxXML），按用户要求修改，保留无需变更的部分：\n{existing_content}"
        if existing_content
        else ""
    )
    prompt = f"""{_doc_generation_prompt()}

用户原始请求：
{message}

任务：
- 生成完整 DocxXML 内容
- 忠实反映用户材料的关键信息，不添加无依据事实
- DocxXML 必须包含唯一 <title>，标题必须来自用户主题或材料中的核心事项
- 禁止使用 Untitled、无标题、默认标题或空标题
- 必须输出合法 XML 标签结构，标签本身不要转义
- 如果是 PRD 或产品需求文档，必须使用 PRD 模板，且输出 DocxXML，不得输出 Markdown
- 如果是会议纪要，必须使用会议纪要模板，且输出 DocxXML，不得输出 Markdown
- 如果用户原始请求包含”生成约束上下文”，必须把其中的 cite_users、whiteboards、images、checkboxes、links、grids 转化为对应 DocxXML 标签
- 若 links 或用户输入 URL 非空，相关链接不得写”未提及相关链接”
- 只输出文档内容，不要输出说明，不要使用代码块{update_section}
"""
    content = get_llm_for_node("doc").invoke(prompt).content
    return _strip_code_fence(_content_to_text(content)).strip()


def _doc_generation_prompt() -> str:
    context = _lark_doc_context(markdown=False)
    logger.info("doc_generation_prompt built doc_format=xml context_len=%s", len(context))
    return f"""你是 IM Copilot 的文档内容生成器。始终用中文输出。

职责：
- 根据用户材料生成完整文档内容，不添加无依据事实。
- 固定使用 DocxXML，即使用户提到 Markdown 也输出 DocxXML。
- 当前代码会负责创建飞书文档；你只负责输出文档内容。
- 文档必须有明确标题，禁止输出 Untitled、无标题、默认标题或空标题。
- 如用户要求生成 PRD、产品需求文档、需求方案或需求说明，必须使用下方 PRD 模板。
- 如用户要求生成会议纪要，必须使用下方会议纪要模板。
- 如果用户要求从会议材料生成 PRD，使用 PRD 模板。
- 用户消息中如果出现“生成约束上下文”，它不是普通参考资料；其中“结构化字段 JSON”和“生成要求”是本次输出的硬性输入。
- 生成会议纪要时，参会人、待办负责人、白板、图片、分栏和相关链接优先来自结构化字段 JSON。
- 输出前检查：结构化字段 JSON 中非空的 cite_users、whiteboards、checkboxes、links、grids，应在最终 DocxXML 中出现对应 <cite>、<whiteboard>、<checkbox>、<a>/<bookmark>、<grid> 标签；没有可用图片 URL 时才省略 <img>。

当前格式：DocxXML

{_DOCXXML_CAPABILITIES}

{_MEETING_MINUTES_TEMPLATE}

{_PRD_TEMPLATE}

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
- 根据用户材料生成完整文档内容，不添加无依据事实。
- 固定使用 DocxXML，即使用户提到 Markdown 也输出 DocxXML。
- 生成内容后必须调用 create_doc_artifact。
- create_doc_artifact 是当前环境的飞书文档创建工具；不要直接调用 lark-cli。
- 工具调用完成后，只返回简洁结果，包含标题、状态和链接。
- 不得把准备动作作为最终回复；最终回复只允许出现在 create_doc_artifact 工具调用之后。
- 如用户要求生成 PRD、产品需求文档、需求方案或需求说明，必须使用下方 PRD 模板。
- 如用户要求生成会议纪要，必须使用下方会议纪要模板。
- 如果用户要求从会议材料生成 PRD，使用 PRD 模板。
- 用户消息中如果出现“生成约束上下文”，它不是普通参考资料；其中“结构化字段 JSON”和“生成要求”是本次输出的硬性输入。
- 生成会议纪要时，参会人、待办负责人、白板、图片、分栏和相关链接优先来自结构化字段 JSON。
- 输出前检查：结构化字段 JSON 中非空的 cite_users、whiteboards、checkboxes、links、grids，应在最终 DocxXML 中出现对应 <cite>、<whiteboard>、<checkbox>、<a>/<bookmark>、<grid> 标签；没有可用图片 URL 时才省略 <img>。

当前格式：DocxXML

{_DOCXXML_CAPABILITIES}

{_MEETING_MINUTES_TEMPLATE}

{_PRD_TEMPLATE}

{context}
"""


def _lark_doc_context(*, markdown: bool = False) -> str:
    context = (
        "输出要求：只输出合法 DocxXML；每篇只含一个 <title>；"
        "标题必须来自用户主题或材料中的核心事项，禁止使用 Untitled、无标题、默认标题或空标题；"
        "不要输出 Markdown 标题、代码块或 Markdown 表格。"
    )
    logger.info("lark_doc_context loaded compact markdown=%s total_chars=%s", markdown, len(context))
    return context


def _read_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
        logger.debug("lark_doc_context file loaded path=%s chars=%s", path, len(text))
        return text
    except OSError as exc:
        logger.warning("lark_doc_context file read failed path=%s error=%s", path, exc)
        return f"无法读取 {path}: {exc}"
