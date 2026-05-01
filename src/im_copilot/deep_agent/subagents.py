from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_subagents(
    artifact_tools: list[Callable[..., Any]],
    *,
    skills: list[str] | None = None,
) -> list[dict[str, Any]]:
    doc_tool, whiteboard_tool, slide_tool = artifact_tools
    return [
        {
            "name": "doc_agent",
            "description": "Generate complete document content and create Feishu document artifacts.",
            "system_prompt": (
                "你负责生成完整文档产物。必须读取并遵循原始 lark-doc skill："
                "`/.agents/skills/lark-doc/SKILL.md`。读取其引用资料时使用 "
                "`/.agents/skills/lark-doc/...` 形式的完整虚拟路径。"
                "如果调用方已经提供 lark-doc 上下文，直接遵循该上下文完成文档。"
                "必须先生成完整内容，再调用 create_doc_artifact，不直接调用 lark-cli。"
                "内容要忠实覆盖用户材料，不添加无依据事实。"
            ),
            "tools": [doc_tool],
            "skills": skills or [],
        },
        {
            "name": "whiteboard_agent",
            "description": "Generate complete Mermaid diagrams and create Feishu whiteboard artifacts.",
            "system_prompt": (
                "你负责生成完整白板产物。必须输出完整 Mermaid 内容，再调用 create_whiteboard_artifact。"
                "节点文字使用中文，结构清晰。"
            ),
            "tools": [whiteboard_tool],
        },
        {
            "name": "slide_agent",
            "description": "Generate complete slide XML and create Feishu slide artifacts.",
            "system_prompt": (
                "你负责生成完整 PPT 产物。必须生成 JSON 字符串数组形式的完整 slide XML，"
                "再调用 create_slide_artifact。"
            ),
            "tools": [slide_tool],
        },
        {
            "name": "verifier_agent",
            "description": "Validate whether generated artifacts are complete and usable.",
            "system_prompt": (
                "你负责校验产物完整性和创建结果。发现缺失内容、空预览或创建失败时，"
                "返回明确问题和需要修正的产物类型。"
            ),
            "tools": [],
        },
        {
            "name": "secretary_agent",
            "description": "Extract explicit group-chat TODO items only.",
            "system_prompt": (
                "你只识别显性任务：负责人、动作或交付物、可解析时间、来源消息均明确时才输出任务。"
                "不要记录用户习惯。"
            ),
            "tools": [],
        },
    ]
