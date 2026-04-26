"""Factory for Feishu (Lark) Card JSON 2.0 templates used by the IM Copilot Agent.

All cards are built for CardKit V2 with streaming_mode and update_multi enabled.
"""

from __future__ import annotations


def _base_card() -> dict:
    """Return the common card envelope with streaming flags."""
    return {
        "config": {
            "streaming_mode": True,
            "update_multi": True,
        },
        "card_link": {},
        "header": {},
        "elements": [],
    }


def create_progress_card(title: str = "正在处理...") -> dict:
    """Return an initial progress card.

    Args:
        title: The main title shown in the progress markdown element.

    Returns:
        A Feishu CardKit V2 dict with a markdown element (element_id="progress_md")
        and a div element (element_id="status_div").
    """
    card = _base_card()
    card["header"] = {
        "title": {
            "tag": "plain_text",
            "content": "处理进度",
        },
        "template": "blue",
    }
    card["elements"] = [
        {
            "tag": "markdown",
            "element_id": "progress_md",
            "content": f"**{title}**",
        },
        {
            "tag": "div",
            "element_id": "status_div",
            "text": {
                "tag": "plain_text",
                "content": "准备开始...",
            },
        },
    ]
    return card


def create_approval_card(
    plan: list[str],
    intent_type: str,
    intent_params: dict,
) -> dict:
    """Return an interactive approval card.

    Args:
        plan: List of planned steps to display.
        intent_type: The classified intent type.
        intent_params: Extracted intent parameters.

    Returns:
        A Feishu CardKit V2 dict with markdown content and approve/reject buttons.
    """
    plan_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    params_text = "\n".join(f"- **{k}**: {v}" for k, v in intent_params.items())

    content = (
        f"**意图类型:** {intent_type}\n\n"
        f"**执行计划:**\n{plan_text}\n\n"
        f"**参数:**\n{params_text}"
    )

    card = _base_card()
    card["header"] = {
        "title": {
            "tag": "plain_text",
            "content": "需要您的确认",
        },
        "template": "orange",
    }
    card["elements"] = [
        {
            "tag": "markdown",
            "element_id": "approval_md",
            "content": content,
        },
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {
                        "tag": "plain_text",
                        "content": "✅ 同意",
                    },
                    "type": "primary",
                    "value": {"action": "approve"},
                },
                {
                    "tag": "button",
                    "text": {
                        "tag": "plain_text",
                        "content": "❌ 拒绝",
                    },
                    "type": "danger",
                    "value": {"action": "reject"},
                },
            ],
        },
    ]
    return card


def create_clarification_card(questions: list[str]) -> dict:
    """Return an interactive clarification card.

    Args:
        questions: List of questions that need user clarification.

    Returns:
        A Feishu CardKit V2 dict with markdown content and buttons for each answer.
    """
    questions_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    content = f"**为了更准确地帮助您，请补充以下信息：**\n\n{questions_text}"

    actions = []
    for i, q in enumerate(questions):
        actions.append(
            {
                "tag": "button",
                "text": {
                    "tag": "plain_text",
                    "content": f"回答: {q[:20]}..." if len(q) > 20 else f"回答: {q}",
                },
                "type": "default",
                "value": {
                    "action": "clarify",
                    "question_index": i,
                    "question": q,
                },
            }
        )

    card = _base_card()
    card["header"] = {
        "title": {
            "tag": "plain_text",
            "content": "需要补充信息",
        },
        "template": "wathet",
    }
    card["elements"] = [
        {
            "tag": "markdown",
            "element_id": "clarification_md",
            "content": content,
        },
        {
            "tag": "action",
            "actions": actions,
        },
    ]
    return card


def create_result_card(
    summary: str,
    artifacts: dict,
    doc_links: list[dict] | None = None,
) -> dict:
    """Return a final result card.

    Args:
        summary: Text summary of the result.
        artifacts: Dictionary of generated artifacts metadata.
        doc_links: Optional list of document link dicts, each with at least
            ``title`` and ``url`` keys.

    Returns:
        A Feishu CardKit V2 dict with summary markdown and document links.
    """
    artifacts_text = "\n".join(
        f"- **{k}**: {v}" for k, v in artifacts.items()
    )
    content = f"**处理结果:**\n\n{summary}\n\n**产物:**\n{artifacts_text}"

    elements: list[dict] = [
        {
            "tag": "markdown",
            "element_id": "result_md",
            "content": content,
        }
    ]

    if doc_links:
        links_md = "\n".join(
            f"- [{link['title']}]({link['url']})" for link in doc_links
        )
        elements.append(
            {
                "tag": "markdown",
                "element_id": "doc_links_md",
                "content": f"**相关文档:**\n\n{links_md}",
            }
        )

    card = _base_card()
    card["header"] = {
        "title": {
            "tag": "plain_text",
            "content": "处理完成",
        },
        "template": "green",
    }
    card["elements"] = elements
    return card


def update_progress_payload(
    card_id: str,
    step: str,
    detail: str,
    sequence: int,
) -> dict:
    """Return the payload for ``update_card_stream`` to update the progress card.

    Args:
        card_id: The ID of the card to update.
        step: The current step name.
        detail: Detailed description of the current step.
        sequence: Monotonically increasing sequence number for streaming.

    Returns:
        A dict suitable for passing to the Feishu card stream update API.
    """
    return {
        "card_id": card_id,
        "sequence": sequence,
        "card": {
            "config": {
                "streaming_mode": True,
                "update_multi": True,
            },
            "elements": [
                {
                    "tag": "markdown",
                    "element_id": "progress_md",
                    "content": f"**{step}**",
                },
                {
                    "tag": "div",
                    "element_id": "status_div",
                    "text": {
                        "tag": "plain_text",
                        "content": detail,
                    },
                },
            ],
        },
    }
