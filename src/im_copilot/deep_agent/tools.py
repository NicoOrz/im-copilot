from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

from im_copilot.deep_agent.events import record_event
from im_copilot.skills.lark_doc import create_doc_from_content
from im_copilot.skills.lark_slide import create_slide_from_xml
from im_copilot.skills.lark_whiteboard import create_whiteboard_from_mermaid

ArtifactDict = dict[str, Any]
logger = logging.getLogger(__name__)


def build_artifact_tools(
    *,
    thread_id: str,
    source: str,
    user_access_token: str = "",
    artifacts: dict[str, ArtifactDict] | None = None,
) -> list[Callable[..., ArtifactDict]]:
    artifact_store = artifacts if artifacts is not None else {}

    @tool("create_doc_artifact")
    def create_doc_artifact(title: str, content: str, doc_format: str = "xml") -> ArtifactDict:
        """Create a Feishu document artifact from complete DocxXML content."""
        doc_format = "xml"
        logger.info(
            "create_doc_artifact start thread_id=%s source=%s title=%r doc_format=%s content_len=%s has_user_token=%s",
            thread_id,
            source,
            title,
            doc_format,
            len(content or ""),
            bool(user_access_token),
        )
        record_event(thread_id, source, "tool_call", {"tool": "create_doc_artifact", "title": title})
        artifact = create_doc_from_content(
            title=title,
            content=content,
            user_access_token=user_access_token,
            doc_format=doc_format,
        )
        artifact_store["doc"] = dict(artifact)
        logger.info(
            "create_doc_artifact result thread_id=%s status=%s token=%s url=%s preview_len=%s",
            thread_id,
            artifact.get("status"),
            artifact.get("token"),
            artifact.get("url"),
            len(str(artifact.get("preview") or "")),
        )
        record_event(thread_id, source, "artifact_created", {"kind": "doc", "artifact": dict(artifact)})
        if _artifact_failed(artifact, user_access_token):
            logger.warning("create_doc_artifact failed thread_id=%s artifact=%s", thread_id, artifact)
            record_event(thread_id, source, "error", {"error": "doc artifact creation failed", "artifact": dict(artifact)})
        return dict(artifact)

    @tool("create_whiteboard_artifact")
    def create_whiteboard_artifact(title: str, mermaid: str) -> ArtifactDict:
        """Create a Feishu whiteboard artifact from complete Mermaid content."""
        logger.info(
            "create_whiteboard_artifact start thread_id=%s source=%s title=%r mermaid_len=%s has_user_token=%s",
            thread_id,
            source,
            title,
            len(mermaid or ""),
            bool(user_access_token),
        )
        record_event(thread_id, source, "tool_call", {"tool": "create_whiteboard_artifact", "title": title})
        artifact = create_whiteboard_from_mermaid(
            title=title,
            mermaid=mermaid,
            user_access_token=user_access_token,
        )
        artifact_store["whiteboard"] = dict(artifact)
        logger.info(
            "create_whiteboard_artifact result thread_id=%s status=%s token=%s url=%s preview_len=%s",
            thread_id,
            artifact.get("status"),
            artifact.get("token"),
            artifact.get("url"),
            len(str(artifact.get("preview") or "")),
        )
        record_event(thread_id, source, "artifact_created", {"kind": "whiteboard", "artifact": dict(artifact)})
        if _artifact_failed(artifact, user_access_token):
            logger.warning("create_whiteboard_artifact failed thread_id=%s artifact=%s", thread_id, artifact)
            record_event(thread_id, source, "error", {"error": "whiteboard artifact creation failed", "artifact": dict(artifact)})
        return dict(artifact)

    @tool("create_slide_artifact")
    def create_slide_artifact(title: str, slides_xml: str) -> ArtifactDict:
        """Create a Feishu slide artifact from complete slide XML content."""
        logger.info(
            "create_slide_artifact start thread_id=%s source=%s title=%r slides_xml_len=%s has_user_token=%s",
            thread_id,
            source,
            title,
            len(slides_xml or ""),
            bool(user_access_token),
        )
        record_event(thread_id, source, "tool_call", {"tool": "create_slide_artifact", "title": title})
        artifact = create_slide_from_xml(
            title=title,
            slides_xml=slides_xml,
            user_access_token=user_access_token,
        )
        artifact_store["slide"] = dict(artifact)
        logger.info(
            "create_slide_artifact result thread_id=%s status=%s token=%s url=%s preview_len=%s",
            thread_id,
            artifact.get("status"),
            artifact.get("token"),
            artifact.get("url"),
            len(str(artifact.get("preview") or "")),
        )
        record_event(thread_id, source, "artifact_created", {"kind": "slide", "artifact": dict(artifact)})
        if _artifact_failed(artifact, user_access_token):
            logger.warning("create_slide_artifact failed thread_id=%s artifact=%s", thread_id, artifact)
            record_event(thread_id, source, "error", {"error": "slide artifact creation failed", "artifact": dict(artifact)})
        return dict(artifact)

    return [create_doc_artifact, create_whiteboard_artifact, create_slide_artifact]


def _artifact_failed(artifact: ArtifactDict, user_access_token: str) -> bool:
    return artifact.get("status") == "error" or (
        bool(user_access_token) and artifact.get("status") != "created"
    )
