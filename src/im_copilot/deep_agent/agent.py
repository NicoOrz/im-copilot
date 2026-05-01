from __future__ import annotations

import os
from typing import Any

from im_copilot.deep_agent.subagents import build_subagents
from im_copilot.deep_agent.tools import build_artifact_tools
from im_copilot.llm import get_llm_for_node

SYSTEM_PROMPT = "你是 IM Copilot 的 Deep Agents 主 Agent。遵循已加载 memory、skills 和 subagents。"


def build_agent(
    *,
    thread_id: str,
    source: str,
    user_access_token: str = "",
    artifacts: dict[str, dict[str, Any]] | None = None,
) -> Any:
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:
        raise RuntimeError("deepagents dependency is not installed") from exc

    artifact_tools = build_artifact_tools(
        thread_id=thread_id,
        source=source,
        user_access_token=user_access_token,
        artifacts=artifacts,
    )
    kwargs: dict[str, Any] = {}
    try:
        from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

        kwargs["backend"] = CompositeBackend(
            default=StateBackend(),
            routes={
                "/.agents/skills/": FilesystemBackend(
                    root_dir="/home/claude_worker/.agents/skills",
                    virtual_mode=True,
                ),
                "/.agents/memory/": FilesystemBackend(
                    root_dir="/home/claude_worker/im_copilot/.agents/memory",
                    virtual_mode=True,
                )
            },
        )
    except Exception:
        pass

    return create_deep_agent(
        name="im-copilot",
        model=get_llm_for_node("deep_agent"),
        tools=artifact_tools,
        system_prompt=SYSTEM_PROMPT,
        subagents=build_subagents(artifact_tools, skills=_skill_sources()),
        skills=_skill_sources(),
        memory=_memory_sources(),
        **kwargs,
    )


def _skill_sources() -> list[str]:
    configured = os.getenv("IM_COPILOT_DEEP_AGENT_SKILLS", "").strip()
    if configured:
        return [path for path in configured.split(os.pathsep) if path]
    return ["/.agents/skills"]


def _memory_sources() -> list[str]:
    configured = os.getenv("IM_COPILOT_DEEP_AGENT_MEMORY", "").strip()
    if configured:
        return [path for path in configured.split(os.pathsep) if path]
    return ["/.agents/memory/AGENTS.md"]
