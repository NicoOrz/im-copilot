from __future__ import annotations

from im_copilot.skills.base import SkillSpec
from im_copilot.skills.lark_doc import create as create_lark_doc
from im_copilot.skills.lark_slide import create as create_lark_slide
from im_copilot.skills.lark_whiteboard import create as create_lark_whiteboard

_SKILLS: dict[str, SkillSpec] = {
    "lark_doc.create": SkillSpec(
        name="lark_doc.create",
        description="创建飞书文档，默认使用 DocxXML，用户明确要求 Markdown 时使用 Markdown。",
        plan_step="doc",
        handler=create_lark_doc,
    ),
    "lark_whiteboard.create": SkillSpec(
        name="lark_whiteboard.create",
        description="创建独立飞书白板，优先生成 Mermaid，并写入文档内的白板资源。",
        plan_step="whiteboard",
        handler=create_lark_whiteboard,
    ),
    "lark_slide.create": SkillSpec(
        name="lark_slide.create",
        description="创建飞书 PPT，默认 5-8 页，使用 slides +create 写入完整 slide XML。",
        plan_step="slide",
        handler=create_lark_slide,
    ),
}


def get_skill(name: str) -> SkillSpec:
    return _SKILLS[name]


def list_skills() -> list[SkillSpec]:
    return list(_SKILLS.values())


def planner_capability_text() -> str:
    return "\n".join(
        f"- {spec.plan_step}: {spec.name} - {spec.description}"
        for spec in list_skills()
    )
