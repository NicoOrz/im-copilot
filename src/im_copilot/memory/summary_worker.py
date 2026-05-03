from __future__ import annotations

import json
import logging
import re
from datetime import datetime, time as dt_time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from im_copilot.deep_agent.events import iter_user_messages_for_chat
from im_copilot.llm import invoke_structured
from im_copilot.memory.group_board_store import GroupBoardItem, group_board_store

logger = logging.getLogger(__name__)
_TZ = ZoneInfo("Asia/Hong_Kong")


class SummaryAction(BaseModel):
    assignee_open_id: str = ""
    assignee_name: str = ""
    text: str = ""


class SummaryTopic(BaseModel):
    scope: str = "noise"
    title: str = ""
    progress: str = ""
    details: str = ""
    actions: list[SummaryAction] = Field(default_factory=list)


class SummaryOutput(BaseModel):
    topics: list[SummaryTopic] = Field(default_factory=list)


class MeetingMetadataPatch(BaseModel):
    item_id: int = 0
    topic: str = ""
    start: str = ""
    end: str = ""
    location: str = ""
    recipients: list[str] = Field(default_factory=list)
    reason: str = ""


class MeetingMetadataPatchOutput(BaseModel):
    patches: list[MeetingMetadataPatch] = Field(default_factory=list)


def summary_today(chat_id: str) -> str:
    today = datetime.now(_TZ).date()
    start = datetime.combine(today, dt_time.min, tzinfo=_TZ).timestamp()
    end = datetime.combine(today + timedelta(days=1), dt_time.min, tzinfo=_TZ).timestamp()
    board_items = [
        item for item in group_board_store.created_between(chat_id, start, end)
        if item.status != "deleted"
    ]
    messages = iter_user_messages_for_chat(chat_id, start)
    if not board_items and not messages:
        return "聊天概要：暂无公开群事项。"

    board_items = _refresh_meeting_metadata(board_items, messages)
    lines = ["聊天概要：", _summarize_messages(messages)]
    _extend_section(lines, "公开任务：", [_format_assignment(item) for item in board_items if item.item_type == "assignment"])
    _extend_section(
        lines,
        "待确认会议：",
        [_format_meeting(item) for item in board_items if item.item_type == "meeting" and item.status == "pending_confirmation"],
    )
    _extend_section(lines, "决议：", [_format_simple(item) for item in board_items if item.item_type == "decision"])
    _extend_section(lines, "风险：", [_format_simple(item) for item in board_items if item.item_type == "risk"])
    _extend_section(lines, "待定问题：", [_format_simple(item) for item in board_items if item.item_type == "question"])
    _extend_section(lines, "资源链接：", [_format_resource(item) for item in board_items if item.item_type == "resource"])
    _extend_section(lines, "明日关注：", [_format_tomorrow(item) for item in board_items if _is_tomorrow(item)])
    return "\n".join(lines)


def _summarize_messages(messages: list[dict]) -> str:
    message_rows = []
    mention_map = _mention_map(messages)
    mentions = _mention_records(messages)
    for event in messages:
        payload = event.get("payload", {})
        text = _clean_message_text(str(payload.get("text") or ""))
        if text:
            sender_open_id = str(payload.get("source_open_id") or payload.get("user_id") or "").strip()
            message_rows.append(
                {
                    "sender_open_id": sender_open_id,
                    "sender": _at_tag(sender_open_id) if sender_open_id else "",
                    "text": _apply_at_tags(text, mention_map)[:300],
                    "mentions": payload.get("mentions") or [],
                }
            )
    if not message_rows:
        return "暂无公开群事项。"
    prompt = (
        "请基于以下原始群聊消息生成今日聊天概要。\n"
        "目标：按话题归并公开团队信息，让用户快速知道今天群里发生了什么、当前进展和后续动作。\n"
        "输出必须是结构化字段，不要输出 Markdown 字符串。\n"
        "筛选规则：只保留公开团队事项、会议、任务、通知、决议、风险、待定问题、团队资料；"
        "排除私人自我提醒、寒暄、无意义数字、天气闲聊、请求机器人总结历史记录、个人阅读感想和普通链接分享。"
        "链接只有被明确作为团队资料、任务材料、通知附件或决策依据时才纳入。\n"
        "组织规则：同一话题下的时间、地点、负责人、结论、后续动作必须归并到同一话题；"
        "不要把同一会议或同一任务拆成多段；一行能表达清楚的信息保持在同一行。\n"
        "会议规则：聊天概要只写会议进展和背景，会议的具体时间、地点、确认人员不要写入 actions；"
        "这些信息由看板会议区展示。\n"
        "行动建议规则：actions 只写需要后续执行的动作，不要写已经结构化为会议候选的信息。\n"
        "结构字段：topics[].scope 必须是 team、personal 或 noise，只有公开团队话题使用 team；"
        "topics[].title 写话题；topics[].progress 写当前状态；topics[].details 写事实细节；"
        "topics[].actions 写后续动作，assignee_open_id 只能来自 mentions 或消息 sender_open_id，text 只写动作内容。\n"
        "人员规则：消息中的“我”指该条消息的 sender_open_id；能确定人员时必须使用 open_id；无法确定人员时不要编造。\n"
        "输出规则：不要写与群消息无关的信息；不要解释筛选过程；总长度控制在 600 字以内。\n\n"
        f"mentions：{json.dumps(mentions, ensure_ascii=False)}\n"
        f"messages：{json.dumps(message_rows, ensure_ascii=False)}"
    )
    try:
        output = invoke_structured("group_summary", SummaryOutput, prompt, timeout=30, max_retries=1)
    except Exception as exc:
        logger.warning("group summary LLM failed: %s", exc)
        return "智能摘要生成失败，请稍后重试。"
    summary = _format_summary_output(output, mention_map)
    return summary or "暂无公开群事项。"


def _clean_message_text(text: str) -> str:
    cleaned = re.sub(r"@_user_\d+", "", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _refresh_meeting_metadata(items: list[GroupBoardItem], messages: list[dict]) -> list[GroupBoardItem]:
    meetings = [
        item for item in items
        if item.item_type == "meeting" and item.status == "pending_confirmation"
    ]
    if not meetings or not messages:
        return items

    message_rows = _message_rows_for_metadata(messages)
    if not message_rows:
        return items

    prompt = (
        "请基于今日原始群聊消息，校准已有待确认会议的结构化字段。\n"
        "输出必须是结构化字段，不要输出 Markdown 字符串。\n"
        "只处理 existing_meetings 中已有会议，不创建新会议。\n"
        "判断依据必须来自 messages 原文；同一会议有多条消息时，较新的时间、地点、主题、参与人信息优先。\n"
        "地点规则：如果消息明确写出会议地点、会议室、楼层或房间名，必须写入 location；无法确定才留空。\n"
        "时间规则：start/end 使用 ISO 8601；只知道开始时间且不知道结束时间时，end 留空。\n"
        "人员规则：recipients 只能使用 messages 中的 sender_open_id 或 mentions.open_id，不能编造。\n"
        "保留规则：无法从原文确认的新字段留空，代码会保留旧值。\n\n"
        f"existing_meetings：{json.dumps(_meeting_payload(meetings), ensure_ascii=False)}\n"
        f"messages：{json.dumps(message_rows, ensure_ascii=False)}"
    )
    try:
        output = invoke_structured(
            "group_board_meeting_metadata",
            MeetingMetadataPatchOutput,
            prompt,
            timeout=30,
            max_retries=1,
        )
    except Exception as exc:
        logger.warning("meeting metadata refresh LLM failed: %s", exc)
        return items

    by_id = {item.id: item for item in meetings}
    refreshed: dict[int, GroupBoardItem] = {}
    for patch in output.patches:
        item = by_id.get(patch.item_id)
        if not item:
            continue
        metadata = _json_dict(item.metadata_json)
        changed = False
        for key in ("topic", "location"):
            value = _clean_line(str(getattr(patch, key, "") or ""))
            if value and value != str(metadata.get(key) or ""):
                metadata[key] = value
                changed = True
        start_value = _iso_or_empty(str(patch.start or ""))
        end_value = _iso_or_empty(str(patch.end or ""))
        if start_value and end_value and start_value == end_value:
            end_value = ""
        if start_value and start_value != str(metadata.get("start") or ""):
            metadata["start"] = start_value
            changed = True
        if end_value and end_value != str(metadata.get("end") or ""):
            metadata["end"] = end_value
            changed = True
        recipients = _json_list(patch.recipients)
        if recipients:
            old = _json_list(metadata.get("recipients"))
            merged = _dedupe([*old, *recipients])
            if merged != old:
                metadata["recipients"] = merged
                changed = True
        if not changed:
            continue
        updated = group_board_store.update_item(
            item.id,
            title=(metadata.get("topic") or item.title),
            due_at=str(metadata.get("start") or item.due_at or ""),
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
        if updated:
            refreshed[item.id] = updated

    if not refreshed:
        return items
    return [refreshed.get(item.id, item) for item in items]


def _message_rows_for_metadata(messages: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mention_map = _mention_map(messages)
    for event in messages:
        payload = event.get("payload", {})
        text = _clean_message_text(str(payload.get("text") or ""))
        if not text:
            continue
        rows.append(
            {
                "sender_open_id": str(payload.get("source_open_id") or payload.get("user_id") or "").strip(),
                "text": _apply_at_tags(text, mention_map)[:300],
                "mentions": payload.get("mentions") or [],
            }
        )
    return rows


def _meeting_payload(meetings: list[GroupBoardItem]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.id,
            "title": item.title,
            "due_at": item.due_at,
            "source_text": item.source_text,
            "metadata": _json_dict(item.metadata_json),
        }
        for item in meetings
    ]


def _extend_section(lines: list[str], title: str, items: list[str]) -> None:
    visible = [item for item in items if item]
    if not visible:
        return
    lines.append(title)
    lines.extend(visible)


def _format_assignment(item: GroupBoardItem) -> str:
    owner = _display_person(item.owner_name, item.owner_open_id)
    fields = [f"- {_clean_message_text(item.title)}"]
    if owner != "未指定":
        fields.append(f"负责人：{owner}")
    if item.due_at:
        fields.append(f"时间：{_format_time_value(item.due_at)}")
    fields.append(f"状态：{_status_label(item.status)}")
    return _join_card_lines(fields)


def _format_meeting(item: GroupBoardItem) -> str:
    metadata = _json_dict(item.metadata_json)
    topic = _clean_message_text(str(metadata.get("topic") or item.title or "会议"))
    start = str(metadata.get("start") or item.due_at or "")
    end = str(metadata.get("end") or "")
    location = str(metadata.get("location") or "").strip() or "未指定"
    recipients = _recipient_names(metadata)
    owner = _display_person(item.owner_name, item.owner_open_id)
    time_text = _format_time_range(start, end)
    recipient_text = "、".join(recipients)
    owner_text = f"｜负责人：{owner}" if owner != "未指定" else ""
    fields = [f"- {topic}", f"时间：{time_text}", f"地点：{location}"]
    if owner_text:
        fields.append(owner_text.removeprefix("｜"))
    if recipient_text:
        fields.append(f"待确认人员：{recipient_text}")
    fields.append(f"状态：{_status_label(item.status)}")
    return _join_card_lines(fields)


def _format_simple(item: GroupBoardItem) -> str:
    owner = _display_person(item.owner_name, item.owner_open_id, empty="")
    due = item.due_at
    suffix = []
    if owner:
        suffix.append(f"负责人：{owner}")
    if due:
        suffix.append(f"时间：{_format_time_value(due)}")
    suffix.append(f"状态：{_status_label(item.status)}")
    lines = [f"- {_clean_message_text(item.title)}"]
    lines.extend(suffix)
    return _join_card_lines(lines)


def _format_resource(item: GroupBoardItem) -> str:
    metadata = _json_dict(item.metadata_json)
    url = str(metadata.get("url") or metadata.get("link") or "").strip()
    title = _clean_message_text(str(metadata.get("title") or item.title or url))
    if url and url not in title:
        return f"- {title}\n  链接：{url}"
    return f"- {title or url}"


def _format_tomorrow(item: GroupBoardItem) -> str:
    due = item.due_at or _json_time(item.metadata_json)
    return _join_card_lines([
        f"- {_clean_message_text(item.title)}",
        f"时间：{_format_time_value(due)}",
        f"状态：{_status_label(item.status)}",
    ])


def _format_summary_output(output: SummaryOutput, mention_map: dict[str, str]) -> str:
    blocks: list[str] = []
    for topic in output.topics:
        if topic.scope != "team":
            continue
        title = _clean_line(topic.title)
        if not title:
            continue
        lines = [f"话题：{_apply_at_tags(title, mention_map)}"]
        progress = _clean_line(topic.progress)
        if progress:
            lines.append(f"  进展：{_apply_at_tags(progress, mention_map)}")
        details = _clean_line(topic.details)
        if details:
            lines.append(f"  详细内容：{_apply_at_tags(details, mention_map)}")
        action_lines = _format_actions(topic.actions, mention_map)
        if action_lines:
            lines.append("  行动建议：")
            lines.extend(f"    - {line}" for line in action_lines)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _join_card_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    first, *rest = lines
    if not rest:
        return first
    return "\n".join([first, *(f"  - {line}" for line in rest)])


def _format_actions(actions: list[SummaryAction], mention_map: dict[str, str]) -> list[str]:
    result: list[str] = []
    for action in actions:
        text = _apply_at_tags(_clean_line(action.text), mention_map)
        if not text:
            continue
        actor = _display_person(action.assignee_name, action.assignee_open_id, empty="")
        result.append(f"{actor} {text}".strip())
    return result


def _clean_line(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "")
    cleaned = re.sub(r"[ \t]*([，。；：])", r"\1", cleaned)
    return cleaned.strip(" \n；;，,。")


def _display_person(name: str, open_id: str = "", *, empty: str = "未指定") -> str:
    raw_open_id = str(open_id or "").strip()
    if raw_open_id:
        return _at_tag(raw_open_id)
    value = str(name or "").strip()
    if value:
        return value.lstrip("@")
    return empty


def _recipient_names(metadata: dict[str, Any]) -> list[str]:
    recipients = _json_list(metadata.get("recipients"))
    if recipients:
        return [_at_tag(open_id) for open_id in recipients]
    names = _json_list(metadata.get("recipient_names"))
    if names:
        return [name.lstrip("@") for name in names if name]
    users = metadata.get("mentioned_users")
    if isinstance(users, dict):
        resolved = [
            _display_person(str(users.get(open_id) or ""), open_id, empty="")
            for open_id in _json_list(metadata.get("recipients"))
        ]
        return [item for item in resolved if item]
    return []


def _mention_map(messages: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for event in messages:
        for mention in event.get("payload", {}).get("mentions") or []:
            if not isinstance(mention, dict):
                continue
            open_id = str(mention.get("open_id") or "").strip()
            if not open_id:
                continue
            for raw in (mention.get("key"), mention.get("name")):
                name = str(raw or "").strip()
                if name:
                    result[name] = open_id
                    result[name.lstrip("@")] = open_id
                    result[f"@{name.lstrip('@')}"] = open_id
    return result


def _mention_records(messages: list[dict]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for event in messages:
        for mention in event.get("payload", {}).get("mentions") or []:
            if not isinstance(mention, dict):
                continue
            open_id = str(mention.get("open_id") or "").strip()
            if not open_id or open_id in seen:
                continue
            seen.add(open_id)
            records.append(
                {
                    "open_id": open_id,
                    "name": str(mention.get("name") or "").strip(),
                    "key": str(mention.get("key") or "").strip(),
                }
            )
    return records


def _apply_at_tags(text: str, mention_map: dict[str, str]) -> str:
    result = text
    for label, open_id in sorted(mention_map.items(), key=lambda item: len(item[0]), reverse=True):
        if not label.startswith("@"):
            continue
        result = result.replace(label, _at_tag(open_id))
    return result


def _at_tag(open_id: str) -> str:
    return f"<at id={open_id}></at>"


def _format_time_range(start: str, end: str) -> str:
    start_text = _format_time_value(start)
    end_dt = _parse_time(end)
    if start_text == "未指定":
        return "未指定"
    if not end_dt:
        return start_text
    start_dt = _parse_time(start)
    end_time = end_dt.strftime("%H:%M")
    if start_dt and start_dt.date() != end_dt.date():
        return f"{start_text} ~ {_format_time_value(end)}"
    return f"{start_text}-{end_time}"


def _format_time_value(value: str) -> str:
    parsed = _parse_time(value)
    if not parsed:
        return value or "未指定"
    now = datetime.now(_TZ)
    clock = parsed.strftime("%H:%M")
    if parsed.date() == now.date():
        return f"今晚 {clock}" if parsed.hour >= 18 else f"今天 {clock}"
    if parsed.date() == now.date() + timedelta(days=1):
        return f"明晚 {clock}" if parsed.hour >= 18 else f"明天 {clock}"
    return parsed.strftime("%Y-%m-%d %H:%M")


def _parse_time(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_TZ)
    return parsed.astimezone(_TZ)


def _is_tomorrow(item: GroupBoardItem) -> bool:
    due = item.due_at or _json_time(item.metadata_json)
    parsed = _parse_time(due)
    if not parsed:
        return False
    tomorrow = datetime.now(_TZ).date() + timedelta(days=1)
    return parsed.date() == tomorrow


def _json_time(metadata_json: str) -> str:
    metadata = _json_dict(metadata_json)
    return str(metadata.get("start") or metadata.get("due_at") or "")


def _json_dict(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _json_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result


def _iso_or_empty(value: str) -> str:
    parsed = _parse_time(value)
    return parsed.isoformat(timespec="minutes") if parsed else ""


def _status_label(status: str) -> str:
    labels = {
        "open": "进行中",
        "pending_confirmation": "待确认",
        "confirmed": "已确认",
        "done": "已完成",
        "deleted": "已忽略",
    }
    return labels.get(status, status or "未知")
