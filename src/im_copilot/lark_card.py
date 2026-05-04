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


def create_command_response_card(text: str, title: str = "命令结果") -> dict:
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
            "element_id": "command_result_md",
            "content": text,
            "text_align": "left",
            "text_size": "normal_v2",
            "margin": "0px 0px 0px 0px",
        }
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


def create_meeting_confirmation_card(
    *,
    board_item_id: int,
    title: str,
    start: str,
    end: str,
    source_text: str,
    attendee_ids: list[str],
) -> dict:
    time_text = f"{start} ~ {end}" if start and end else "时间待确认"
    attendees_text = "、".join(_at_tag(item) for item in attendee_ids) if attendee_ids else "仅创建者"
    source_excerpt = source_text.strip() or "来自群聊会议候选"

    card = _base_card()
    card["header"] = {
        "title": {
            "tag": "plain_text",
            "content": "是否创建飞书会议",
        },
        "template": "orange",
        "padding": "12px 12px 12px 12px",
    }
    card["body"]["elements"] = [
        {
            "tag": "markdown",
            "element_id": "meeting_confirm_md",
            "content": (
                f"**会议事项**\n{title}\n\n"
                f"**时间**\n{time_text}\n\n"
                f"**参与人**\n{attendees_text}\n\n"
                f"**依据**\n{source_excerpt}"
            ),
            "text_align": "left",
            "text_size": "normal_v2",
            "margin": "0px 0px 0px 0px",
        },
        {
            "tag": "button",
            "element_id": "create_meeting_btn",
            "text": {
                "tag": "plain_text",
                "content": "创建日程",
            },
            "type": "primary",
            "behaviors": [
                {
                    "type": "callback",
                    "value": {
                        "action": "create_group_meeting_event",
                        "board_item_id": board_item_id,
                    },
                }
            ],
            "margin": "8px 0px 0px 0px",
        },
        {
            "tag": "button",
            "element_id": "ignore_meeting_btn",
            "text": {
                "tag": "plain_text",
                "content": "忽略",
            },
            "type": "default",
            "behaviors": [
                {
                    "type": "callback",
                    "value": {
                        "action": "ignore_group_meeting_event",
                        "board_item_id": board_item_id,
                    },
                }
            ],
            "margin": "8px 0px 0px 0px",
        },
    ]
    return card


def _at_tag(open_id: str) -> str:
    return f"<at id={open_id}></at>"


def create_todo_confirm_card(
    todo_id: int,
    title: str,
    action_phrase: str,
    due_at: str,
    source_text: str,
    source_open_id: str,
    assignee_name: str = "",
) -> dict:
    """Return an interactive todo confirmation card.

    Args:
        todo_id: The unique identifier for the todo.
        title: The todo title.
        action_phrase: The todo action phrase.
        due_at: The deadline in ISO format (e.g., "2026-05-08T18:00").
        source_text: The original source message text.
        source_open_id: The open_id of the user who created the todo.
        assignee_name: Optional name of the person assigned to the todo.

    Returns:
        A Feishu CardKit V2 dict with markdown content and confirm/reject buttons.
    """
    assignee_line = f"**负责人：** {assignee_name}\n" if assignee_name else ""
    source_excerpt = source_text[:100] + "…" if len(source_text) > 100 else source_text
    content = (
        f"**事项：** {title}\n"
        f"**动作：** {action_phrase}\n"
        f"**截止时间：** {due_at}\n"
        f"{assignee_line}"
        f"**来源消息：** {source_excerpt}"
    )
    card = _base_card()
    card["header"] = {
        "title": {"tag": "plain_text", "content": "待办确认"},
        "template": "yellow",
        "padding": "12px 12px 12px 12px",
    }
    card["body"]["elements"] = [
        {
            "tag": "markdown",
            "element_id": "todo_confirm_md",
            "content": content,
            "text_align": "left",
            "text_size": "normal_v2",
            "margin": "0px 0px 0px 0px",
        },
        {
            "tag": "button",
            "element_id": "confirm_todo_btn",
            "text": {"tag": "plain_text", "content": "确认"},
            "type": "primary",
            "behaviors": [
                {"type": "callback", "value": {"action": "confirm_todo", "todo_id": todo_id}}
            ],
            "margin": "8px 0px 0px 0px",
        },
        {
            "tag": "button",
            "element_id": "reject_todo_btn",
            "text": {"tag": "plain_text", "content": "忽略"},
            "type": "default",
            "behaviors": [
                {"type": "callback", "value": {"action": "reject_todo", "todo_id": todo_id}}
            ],
            "margin": "4px 0px 0px 0px",
        },
    ]
    return card
