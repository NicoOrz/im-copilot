from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from im_copilot.deep_agent.events import recent_user_messages_for_chat, record_event
from im_copilot.llm import invoke_structured
from im_copilot.memory.group_board_store import GroupBoardItem, group_board_store
from im_copilot.memory.todo_extractor import WindowMessage, assemble_window

logger = logging.getLogger(__name__)
_TZ = ZoneInfo("Asia/Hong_Kong")
_MIN_CONFIDENCE = 0.65


class BoardCandidate(BaseModel):
    links_to_existing_id: str = ""
    item_type: Literal["assignment", "meeting", "decision", "risk", "question", "resource"]
    title: str = ""
    status: Literal["open", "pending_confirmation"] = "open"
    owner_open_id: str = ""
    owner_name: str = ""
    due_at: str = ""
    start_at: str = ""
    end_at: str = ""
    recipients: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class BoardExtractionOutput(BaseModel):
    items: list[BoardCandidate] = Field(default_factory=list)


@dataclass
class BoardExtractionResult:
    items: list[GroupBoardItem]
    confirmation_recipients: list[str]


def extract_and_store_group_board_items(
    text: str,
    *,
    chat_id: str,
    message_id: str,
    source_open_id: str,
    source: str = "feishu",
    mentions: list[dict] | None = None,
    now: datetime | None = None,
) -> BoardExtractionResult:
    clean = _clean_text(text)
    if not clean or not chat_id or not message_id:
        return BoardExtractionResult([], [])

    mentions = mentions or []
    created: list[GroupBoardItem] = []
    confirmation_recipients: list[str] = []
    current_time = now or datetime.now(_TZ)
    window = _recent_context(chat_id, message_id, current_time)
    allowed_open_ids = _window_open_ids(window, mentions)
    if not allowed_open_ids:
        return BoardExtractionResult([], [])
    name_by_open_id = _window_name_by_open_id(window, mentions)
    allowed_names = {name for name in name_by_open_id.values() if name}
    existing_items = _existing_open_items(chat_id)

    try:
        extracted = _extract_with_llm(
            clean,
            chat_id=chat_id,
            source_open_id=source_open_id,
            mentions=mentions,
            allowed_open_ids=allowed_open_ids,
            name_by_open_id=name_by_open_id,
            now=current_time,
            window=window,
            existing_items=existing_items,
        )
    except Exception as exc:
        record_event(
            chat_id,
            source,
            "error",
            {
                "error": "group board extraction failed",
                "message_id": message_id,
                "source_open_id": source_open_id,
                "detail": str(exc),
            },
        )
        return BoardExtractionResult([], [])

    existing_by_id = {str(item.id): item for item in existing_items}
    for candidate in extracted.items:
        owner_open_id = candidate.owner_open_id.strip()
        if owner_open_id not in allowed_open_ids:
            logger.warning("group board extraction dropped unknown owner: owner=%s", owner_open_id)
            continue

        link_id = candidate.links_to_existing_id.strip()
        merge_target = None
        if link_id:
            merge_target = existing_by_id.get(link_id)
            if merge_target is None:
                logger.warning("group board extraction ignored unknown existing id: id=%s", link_id)
                link_id = ""
        if not link_id and candidate.confidence < _MIN_CONFIDENCE:
            continue

        item_type = candidate.item_type
        owner_name = _owner_name(candidate, name_by_open_id, allowed_names)
        mention_names = _mention_names(window, mentions)
        start_at = _parse_iso_time(candidate.start_at)
        end_at = _parse_iso_time(candidate.end_at)
        due_at = _parse_iso_time(candidate.due_at)
        if item_type == "meeting" and start_at and not end_at:
            end_at = start_at + timedelta(minutes=30)
        if item_type == "meeting" and not due_at:
            due_at = start_at

        metadata = _metadata(candidate)
        metadata["mentioned_users"] = mention_names
        metadata.setdefault("public_scope", "team")
        recipients = [item for item in _dedupe(candidate.recipients) if item in allowed_open_ids]
        if item_type == "meeting":
            if not recipients and source_open_id in allowed_open_ids:
                recipients = [source_open_id]
            recipient_names = [
                mention_names[open_id]
                for open_id in recipients
                if open_id in mention_names
            ]
            metadata.update(
                {
                    "recipients": recipients,
                    "recipient_names": recipient_names,
                    "start": _iso_minutes(start_at),
                    "end": _iso_minutes(end_at),
                    "location": str(metadata.get("location") or ""),
                    "topic": str(metadata.get("topic") or candidate.title or _title(clean)),
                }
            )

        if merge_target is not None:
            merged_metadata = _merge_metadata(_json_dict(merge_target.metadata_json), metadata)
            item = group_board_store.update_item(
                merge_target.id,
                title=(candidate.title or merge_target.title or _title(clean))[:120],
                owner_open_id=owner_open_id or merge_target.owner_open_id,
                owner_name=owner_name or merge_target.owner_name,
                due_at=_iso_minutes(due_at) or merge_target.due_at,
                status="pending_confirmation" if item_type == "meeting" else candidate.status,
                source_text=_join_source_text(merge_target.source_text, clean),
                metadata_json=_json_dumps(merged_metadata),
            )
        else:
            item = group_board_store.create(
                chat_id=chat_id,
                message_id=message_id,
                item_type=item_type,
                title=(candidate.title or _title(clean))[:120],
                owner_open_id=owner_open_id,
                owner_name=owner_name,
                due_at=_iso_minutes(due_at),
                status="pending_confirmation" if item_type == "meeting" else candidate.status,
                source_open_id=source_open_id,
                source_text=clean,
                metadata_json=_json_dumps(metadata),
            )
        if not item:
            continue

        created.append(item)
        event_kind = "meeting_candidate" if item_type == "meeting" else f"group_{item_type}"
        event_type = "board_item_updated" if merge_target is not None else "todo_detected"
        record_event(chat_id, source, event_type, _event_payload(item, event_kind, candidate))

        if item_type == "meeting" and merge_target is None:
            confirmation_recipients.extend(recipients)

    return BoardExtractionResult(created, _dedupe(confirmation_recipients))


def _extract_with_llm(
    text: str,
    *,
    chat_id: str,
    source_open_id: str,
    mentions: list[dict],
    allowed_open_ids: set[str],
    name_by_open_id: dict[str, str],
    now: datetime,
    window: list[WindowMessage],
    existing_items: list[GroupBoardItem],
) -> BoardExtractionOutput:
    prompt = (
        "你是群聊团队看板识别器。输入包含触发消息所在的对话窗口、当前 chat 内未关闭看板事项、窗口内 open_id/name 白名单。\n"
        "可选类型：assignment, meeting, decision, risk, question, resource。\n"
        "只为触发消息或它直接修改、补充的内容输出 0..N 个 item。\n"
        "如果 item 实际上是在补充或修正 existing_items 里某条，必须设置 links_to_existing_id 为对应 id，不要新建。\n"
        "挂靠覆盖 meeting、assignment、decision、risk、question、resource；会议地点、时间、参会人变更也在同一次输出中判断。\n"
        "owner_open_id 必须出自窗口内可选 open_id 集合，集合包含消息发送者和 mentions 中的 open_id。\n"
        "owner_name 必须出自窗口内已知 name 表；无法确定则留空。\n"
        "recipients 的 open_id 也只能出自窗口内可选 open_id 集合。\n"
        "会议的地点必须写入 metadata.location，不能留“未指定”；会议主题必须写入 metadata.topic，不能留空。\n"
        "时间请用 ISO 8601，无法确定则留空。会议 end_at 缺失可留空。\n"
        "metadata 必须包含 public_scope，团队公开事项为 team，个人事项不要输出。\n"
        "个人自我提醒、寒暄、纯数字、闲聊、个人阅读感想、单纯链接分享输出空列表。\n"
        "resource 仅限团队需复用、决策依据、后续处理的资料。\n"
        "信息不足以判断时输出空列表，不要硬凑 title、owner 或 metadata。\n"
        "输出 items 数组；每个 item 对应一个候选。\n\n"
        f"当前时间：{now.isoformat(timespec='minutes')}\n"
        f"chat_id：{chat_id}\n"
        f"source_open_id：{source_open_id}\n"
        f"窗口内可选 open_id：{json.dumps(sorted(allowed_open_ids), ensure_ascii=False)}\n"
        f"窗口内已知 name：{json.dumps(name_by_open_id, ensure_ascii=False)}\n"
        f"当前消息 mentions：{json.dumps(_mention_lines(mentions), ensure_ascii=False)}\n"
        f"existing_items：\n{_existing_items_text(existing_items)}\n\n"
        f"对话窗口：\n{_window_prompt_text(window)}\n\n"
        f"触发消息：{text}"
    )
    return invoke_structured("group_board_extractor", BoardExtractionOutput, prompt, timeout=60, max_retries=1)


def _parse_iso_time(value: str) -> datetime | None:
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


def _iso_minutes(value: datetime | None) -> str:
    return value.isoformat(timespec="minutes") if value else ""


def _metadata(candidate: BoardCandidate) -> dict[str, Any]:
    value = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    data = dict(value)
    data["confidence"] = candidate.confidence
    data["reason"] = candidate.reason
    return data


def _recent_context(chat_id: str, message_id: str, now: datetime) -> list[WindowMessage]:
    trigger_event_id = _trigger_event_id(chat_id, message_id)
    return assemble_window(chat_id, trigger_event_id, now=now)


def _trigger_event_id(chat_id: str, message_id: str) -> int | None:
    events = recent_user_messages_for_chat(chat_id, limit=10_000, since_ts=0.0)
    for event in reversed(events):
        payload = event.get("payload", {})
        if str(payload.get("message_id") or event.get("id") or "") == message_id:
            return int(event.get("id") or 0)
    return None


def _existing_open_items(chat_id: str) -> list[GroupBoardItem]:
    return [
        item for item in group_board_store.list(chat_id=chat_id)
        if item.status in {"open", "pending_confirmation"}
    ][-20:]


def _merge_metadata(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = dict(old)
    for key, value in new.items():
        if value in ("", None, [], {}):
            continue
        if isinstance(value, list):
            merged[key] = _dedupe([*merged.get(key, []), *value]) if isinstance(merged.get(key), list) else value
        elif isinstance(value, dict):
            current = merged.get(key)
            merged[key] = {**current, **value} if isinstance(current, dict) else value
        else:
            merged[key] = value
    return merged


def _join_source_text(old: str, new: str) -> str:
    old_text = old.strip()
    new_text = new.strip()
    if not old_text:
        return new_text
    if not new_text or new_text in old_text:
        return old_text
    return f"{old_text}\n{new_text}"


def _mention_lines(mentions: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "key": str(item.get("key") or ""),
            "name": str(item.get("name") or item.get("text") or ""),
            "open_id": _mention_open_id(item),
        }
        for item in mentions
        if isinstance(item, dict)
    ]


def _window_open_ids(window: list[WindowMessage], mentions: list[dict]) -> set[str]:
    open_ids = {message.open_id.strip() for message in window if message.open_id.strip()}
    for mention in _all_mentions(window, mentions):
        open_id = _mention_open_id(mention)
        if open_id:
            open_ids.add(open_id)
    return open_ids


def _window_name_by_open_id(window: list[WindowMessage], mentions: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for message in window:
        if message.open_id.strip() and message.name.strip():
            result[message.open_id.strip()] = message.name.strip().lstrip("@")
    for item in _all_mentions(window, mentions):
        open_id = _mention_open_id(item)
        name = str(item.get("name") or item.get("text") or item.get("key") or "").strip()
        if open_id and name:
            result[open_id] = name.lstrip("@")
    return result


def _mention_names(window: list[WindowMessage], mentions: list[dict]) -> dict[str, str]:
    return _window_name_by_open_id(window, mentions)


def _all_mentions(window: list[WindowMessage], mentions: list[dict]) -> list[dict]:
    result: list[dict] = [item for item in mentions if isinstance(item, dict)]
    for message in window:
        for mention in message.mentions:
            if isinstance(mention, dict):
                result.append(mention)
    return result


def _mention_open_id(item: dict) -> str:
    open_id = str(item.get("open_id") or "").strip()
    identity = item.get("id")
    if not open_id and isinstance(identity, dict):
        open_id = str(identity.get("open_id") or "").strip()
    return open_id


def _owner_name(candidate: BoardCandidate, name_by_open_id: dict[str, str], allowed_names: set[str]) -> str:
    name = candidate.owner_name.strip().lstrip("@")
    if name:
        return name if name in allowed_names else ""
    return name_by_open_id.get(candidate.owner_open_id.strip(), "")


def _window_prompt_text(window: list[WindowMessage]) -> str:
    lines = []
    for index, message in enumerate(window, start=1):
        marker = " ← TRIGGER" if message.is_trigger else ""
        lines.append(
            f"[{index}] open_id={message.open_id} name={message.name or '-'} "
            f"ts={_iso_from_ts(message.ts)} {message.text}{marker}"
        )
    return "\n".join(lines)


def _existing_items_text(items: list[GroupBoardItem]) -> str:
    if not items:
        return "(空)"
    return "\n".join(
        f'id={item.id} type={item.item_type} status={item.status} '
        f'owner={item.owner_open_id} due_at={item.due_at} '
        f'title="{_quote(item.title)}" source="{_quote(item.source_text)}" '
        f"metadata={json.dumps(_json_dict(item.metadata_json), ensure_ascii=False)}"
        for item in items
    )


def _iso_from_ts(value: float) -> str:
    if value <= 0:
        return ""
    return datetime.fromtimestamp(value, _TZ).isoformat(timespec="minutes")


def _quote(value: str) -> str:
    return str(value or "").replace('"', '\\"')


def _json_dumps(value: dict[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        cleaned = {str(k): str(v) for k, v in value.items()}
        return json.dumps(cleaned, ensure_ascii=False)


def _json_dict(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _title(text: str) -> str:
    return text.strip(" ，,。")[:80] or "群聊事项"


def _dedupe(values) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result


def _event_payload(item: GroupBoardItem, kind: str, candidate: BoardCandidate) -> dict:
    return {
        "kind": kind,
        "id": item.id,
        "chat_id": item.chat_id,
        "message_id": item.message_id,
        "item_type": item.item_type,
        "title": item.title,
        "owner_open_id": item.owner_open_id,
        "owner_name": item.owner_name,
        "due_at": item.due_at,
        "status": item.status,
        "source_open_id": item.source_open_id,
        "source_text": item.source_text,
        "confidence": candidate.confidence,
        "reason": candidate.reason,
    }
