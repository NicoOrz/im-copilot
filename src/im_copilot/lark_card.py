"""Factory for Feishu (Lark) Card JSON 2.0 templates used by the IM Copilot Agent.

All cards are built for CardKit V2 with streaming_mode and update_multi enabled.
"""

from __future__ import annotations


def _base_card() -> dict:
    """Return the common card envelope with streaming flags."""
    return {
        "schema": "2.0",
        "config": {
            "streaming_mode": True,
            "update_multi": True,
        },
        "header": {},
        "body": {
            "direction": "vertical",
            "padding": "12px 12px 12px 12px",
            "elements": [],
        },
    }


def create_streaming_card(title: str = "思考中...") -> dict:
    """Return a streaming card for LLM output.

    Args:
        title: The header title.

    Returns:
        A Feishu CardKit V2 dict with a markdown element (element_id="stream_md")
        for streaming text updates.
    """
    card = _base_card()
    card["header"] = {
        "title": {
            "tag": "plain_text",
            "content": title,
        },
        "template": "blue",
        "padding": "12px 12px 12px 12px",
    }
    card["body"]["elements"] = [
        {
            "tag": "markdown",
            "element_id": "stream_md",
            "content": "",
            "text_align": "left",
            "text_size": "normal_v2",
            "margin": "0px 0px 0px 0px",
        },
    ]
    return card


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
        "padding": "12px 12px 12px 12px",
    }
    card["body"]["elements"] = [
        {
            "tag": "markdown",
            "element_id": "progress_md",
            "content": f"**{title}**",
            "text_align": "left",
            "text_size": "normal_v2",
            "margin": "0px 0px 0px 0px",
        },
        {
            "tag": "div",
            "element_id": "status_div",
            "text": {
                "tag": "plain_text",
                "content": "准备开始...",
            },
            "margin": "0px 0px 0px 0px",
        },
    ]
    return card


def create_approval_card(
    plan: list[str],
    intent_type: str,
    intent_params: dict,
    thread_id: str | None = None,
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
        "padding": "12px 12px 12px 12px",
    }
    card["body"]["elements"] = [
        {
            "tag": "markdown",
            "element_id": "approval_md",
            "content": content,
            "text_align": "left",
            "text_size": "normal_v2",
            "margin": "0px 0px 0px 0px",
        },
        {
            "tag": "button",
            "element_id": "approve_btn",
            "text": {
                "tag": "plain_text",
                "content": "✅ 同意",
            },
            "type": "primary",
            "behaviors": [
                {"type": "callback", "value": {"action": "approve", "thread_id": thread_id}}
            ],
            "margin": "0px 0px 0px 0px",
        },
        {
            "tag": "button",
            "element_id": "reject_btn",
            "text": {
                "tag": "plain_text",
                "content": "❌ 拒绝",
            },
            "type": "danger",
            "behaviors": [
                {"type": "callback", "value": {"action": "reject", "thread_id": thread_id}}
            ],
            "margin": "8px 0px 0px 0px",
        },
    ]
    return card


def create_clarification_card(questions: list[str], thread_id: str | None = None) -> dict:
    """Return an interactive clarification card.

    Args:
        questions: List of questions that need user clarification.

    Returns:
        A Feishu CardKit V2 dict with markdown content and buttons for each answer.
    """
    questions_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    content = f"**为了更准确地帮助您，请补充以下信息：**\n\n{questions_text}"

    form_elements = []
    for i, q in enumerate(questions):
        form_elements.append(
            {
                "tag": "input",
                "element_id": f"answer_{i}",
                "name": f"answer_{i}",
                "required": False,
                "input_type": "multiline_text",
                "rows": 2,
                "auto_resize": True,
                "placeholder": {
                    "tag": "plain_text",
                    "content": "请输入回答",
                },
                "label": {
                    "tag": "plain_text",
                    "content": f"问题 {i + 1}",
                },
                "margin": "8px 0px 0px 0px",
            }
        )
    form_elements.append(
        {
            "tag": "button",
            "name": "submit_clarification",
            "text": {
                "tag": "plain_text",
                "content": "提交回答",
            },
            "type": "primary",
            "form_action_type": "submit",
            "behaviors": [
                {"type": "callback", "value": {"action": "clarify", "thread_id": thread_id}}
            ],
            "margin": "8px 0px 0px 0px",
        }
    )

    card = _base_card()
    card["header"] = {
        "title": {
            "tag": "plain_text",
            "content": "需要补充信息",
        },
        "template": "wathet",
        "padding": "12px 12px 12px 12px",
    }
    card["body"]["elements"] = [
        {
            "tag": "markdown",
            "element_id": "clarification_md",
            "content": content,
            "text_align": "left",
            "text_size": "normal_v2",
            "margin": "0px 0px 0px 0px",
        },
        {
            "tag": "form",
            "element_id": "clarification_form",
            "name": "clarification_form",
            "elements": form_elements,
            "margin": "8px 0px 0px 0px",
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
    content = f"**处理结果:**\n\n{summary}"
    if artifacts:
        artifacts_text = "\n".join(
            f"- **{k}**: {v}" for k, v in artifacts.items()
        )
        content = f"{content}\n\n**产物:**\n{artifacts_text}"

    elements: list[dict] = [
        {
            "tag": "markdown",
            "element_id": "result_md",
            "content": content,
            "text_align": "left",
            "text_size": "normal_v2",
            "margin": "0px 0px 0px 0px",
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
                "text_align": "left",
                "text_size": "normal_v2",
                "margin": "0px 0px 0px 0px",
            }
        )

    card = _base_card()
    card["header"] = {
        "title": {
            "tag": "plain_text",
            "content": "处理完成",
        },
        "template": "green",
        "padding": "12px 12px 12px 12px",
    }
    card["body"]["elements"] = elements
    return card
