from __future__ import annotations

import copy
import json
import os
import time
from typing import Any

_CONFIG_TTL = 5.0
_config_cache: dict[str, Any] | None = None
_config_cache_ts = 0.0

DEFAULT_SKILL_CONFIG: dict[str, Any] = {
    "skills": {
        "lark_doc": {
            "system_prompt": "你是飞书 DocxXML 文档创建助手。",
            "style_rules": "结构清晰，标题层级明确，正文信息完整。",
            "block_rules": "默认使用 title、h1、p、ul、callout、table、checkbox、grid、whiteboard、bookmark。sheet、task、img、source、time 只有在用户提供必要 token、URL、文件或时间信息时使用；缺失信息时改为普通文本说明。",
            "title_rules": "标题简洁，保留用户主题。",
            "forbidden_text": "不要输出代码块标记，不要添加额外解释。",
        },
        "lark_whiteboard": {
            "system_prompt": "你是飞书白板可视化创建助手。",
            "diagram_rules": "简单流程、思维导图、时序图、类图优先 Mermaid；节点文字简洁。",
            "title_rules": "标题简洁，保留用户主题。",
            "forbidden_text": "不要输出代码块标记，不要添加额外解释。",
        },
        "lark_slide": {
            "system_prompt": "你是飞书幻灯片 XML 创建助手。",
            "style_rules": "信息密度适中，版式清晰，颜色统一。",
            "page_rules": "默认 5-8 页；每页必须是完整 slide XML，内容位于 data 内；只使用文本、基础形状、背景、列表、表格类表达。",
            "title_rules": "标题简洁，保留用户主题。",
            "forbidden_text": "不要输出代码块标记，不要添加额外解释。",
        },
    }
}

SKILL_FIELDS: dict[str, list[str]] = {
    "lark_doc": ["system_prompt", "style_rules", "block_rules", "title_rules", "forbidden_text"],
    "lark_whiteboard": ["system_prompt", "diagram_rules", "title_rules", "forbidden_text"],
    "lark_slide": ["system_prompt", "style_rules", "page_rules", "title_rules", "forbidden_text"],
}


def _config_path() -> str:
    return os.getenv("SKILL_CONFIG", "skill_config.json")


def _merge_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_SKILL_CONFIG)
    raw_skills = raw.get("skills") if isinstance(raw, dict) else {}
    if not isinstance(raw_skills, dict):
        return merged
    for skill_name, fields in SKILL_FIELDS.items():
        raw_skill = raw_skills.get(skill_name)
        if not isinstance(raw_skill, dict):
            continue
        for field in fields:
            value = raw_skill.get(field)
            if isinstance(value, str):
                merged["skills"][skill_name][field] = value
    return merged


def load_skill_config() -> dict[str, Any]:
    global _config_cache, _config_cache_ts
    now = time.time()
    if _config_cache is not None and (now - _config_cache_ts) < _CONFIG_TTL:
        return copy.deepcopy(_config_cache)

    path = _config_path()
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}

    _config_cache = _merge_defaults(raw)
    _config_cache_ts = now
    return copy.deepcopy(_config_cache)


def get_skill_config(skill_name: str) -> dict[str, str]:
    config = load_skill_config()
    skill_config = config.get("skills", {}).get(skill_name, {})
    return dict(skill_config) if isinstance(skill_config, dict) else {}


def save_skill_config(config: dict[str, Any]) -> None:
    global _config_cache, _config_cache_ts
    clean: dict[str, Any] = {"skills": {}}
    raw_skills = config.get("skills") if isinstance(config, dict) else {}
    if isinstance(raw_skills, dict):
        for skill_name, fields in SKILL_FIELDS.items():
            raw_skill = raw_skills.get(skill_name)
            if not isinstance(raw_skill, dict):
                continue
            entry = {}
            for field in fields:
                value = raw_skill.get(field)
                if isinstance(value, str) and value.strip():
                    entry[field] = value.strip()
            if entry:
                clean["skills"][skill_name] = entry

    with open(_config_path(), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    _config_cache = None
    _config_cache_ts = 0.0
