from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict


class SkillArtifact(TypedDict, total=False):
    kind: str
    title: str
    status: Literal["draft", "created", "error"]
    preview: str
    token: str
    url: str


SkillHandler = Callable[[Mapping[str, Any]], SkillArtifact]


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    plan_step: str
    handler: SkillHandler
