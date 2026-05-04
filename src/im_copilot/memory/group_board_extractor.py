from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from im_copilot.deep_agent.events import iter_user_messages_for_chat, record_event
from im_copilot.llm import invoke_structured
from im_copilot.memory.group_board_store import GroupBoardItem, group_board_store
from im_copilot.memory.todo_store import todo_store

_TZ = ZoneInfo("Asia/Hong_Kong")
_MIN_CONFIDENCE = 0.65


class BoardCandidate(BaseModel):
    merge_item_id: int = 0
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
    should_create_personal_todo: bool = False
    personal_todo_assignee_open_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class BoardExtractionOutput(BaseModel):
    items: list[BoardCandidate] = Field(default_factory=list)


class MergeDecision(BaseModel):
    merge_item_id: int = 0
    reason: str = ""


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
    context = _recent_context(chat_id, current_time)
    existing_items = _existing_open_items(chat_id)
    known_users = _build_known_users(source_open_id, mentions, context)

    try:
        extracted = _extract_with_llm(
            clean,
            chat_id=chat_id,
            source_open_id=source_open_id,
            mentions=mentions,
            known_users=known_users,
            now=current_time,
            context=context,
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

    for candidate in extracted.items:
        if candidate.confidence < _MIN_CONFIDENCE:
            continue

        item_type = candidate.item_type
        mention_names = _mention_names(mentions)
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
        recipients = _dedupe(candidate.recipients)
        if item_type == "meeting":
            if not recipients and source_open_id:
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

        merge_target = None
        if item_type == "meeting":
            merge_target = _infer_meeting_merge_target(candidate, existing_items, clean, current_time)
        else:
            merge_target = _merge_target(candidate, existing_items, chat_id, start_at)
        if merge_target is not None:
            merged_metadata = _merge_metadata(_json_dict(merge_target.metadata_json), metadata)
            item = group_board_store.update_item(
                merge_target.id,
                title=(candidate.title or merge_target.title or _title(clean))[:120],
                owner_open_id=candidate.owner_open_id.strip() or merge_target.owner_open_id,
                owner_name=_owner_name(candidate, mention_names) or merge_target.owner_name,
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
                owner_open_id=candidate.owner_open_id.strip(),
                owner_name=_owner_name(candidate, mention_names),
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
        record_event(chat_id, source, "todo_detected", _event_payload(item, event_kind, candidate))

        if item_type == "meeting" and merge_target is None:
            confirmation_recipients.extend(recipients)
            continue

        if item_type != "assignment":
            continue

        assignee = candidate.personal_todo_assignee_open_id.strip() or candidate.owner_open_id.strip()
        if candidate.should_create_personal_todo and assignee and due_at:
            personal = todo_store.create(
                chat_id=chat_id,
                message_id=message_id,
                source_open_id=source_open_id,
                assignee_open_id=assignee,
                title=(candidate.title or _title(clean))[:80],
                action=(candidate.title or _title(clean))[:80],
                due_at=due_at.isoformat(timespec="minutes"),
                remind_at=_remind_at(due_at).isoformat(timespec="minutes"),
                source_text=clean,
                status="pending",
            )
            if personal:
                record_event(
                    chat_id,
                    source,
                    "todo_detected",
                    {
                        "id": personal.id,
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "source_open_id": source_open_id,
                        "assignee_open_id": personal.assignee_open_id,
                        "title": personal.title,
                        "action": personal.action,
                        "due_at": personal.due_at,
                        "remind_at": personal.remind_at,
                        "status": personal.status,
                        "source_text": personal.source_text,
                        "confidence": candidate.confidence,
                        "needs_confirmation": False,
                    },
                )

    return BoardExtractionResult(created, _dedupe(confirmation_recipients))


def _extract_with_llm(
    text: str,
    *,
    chat_id: str,
    source_open_id: str,
    mentions: list[dict],
    known_users: dict[str, str],
    now: datetime,
    context: list[dict[str, Any]],
    existing_items: list[GroupBoardItem],
) -> BoardExtractionOutput:
    mention_lines = []
    for item in mentions:
        mention_lines.append(
            {
                "key": str(item.get("key") or ""),
                "name": str(item.get("name") or ""),
                "open_id": str(item.get("open_id") or ""),
            }
        )
    prompt = (
        "你是群聊团队看板识别器。只识别团队公开事项，输出严格结构化结果。\n"
        "可选类型：assignment, meeting, decision, risk, question, resource。\n"
        "个人自我提醒、寒暄、纯数字、闲聊输出空列表。\n"
        "个人阅读感想、普通链接分享、请求机器人帮忙总结历史记录输出空列表。\n"
        "resource 只用于团队明确需要复用、决策或后续处理的资料。\n"
        "会议地点、时间、主持人变更属于公开会议事项。\n"
        "会议的地点必须写入 metadata.location；会议主题必须写入 metadata.topic。\n"
        "如果当前消息或近期上下文提供了会议地点，不要留空或写未指定。\n"
        "如果当前消息是在补充会议地点、时间、主持人或参与人，优先合并到已有 meeting。\n"
        "时间请用 ISO 8601，无法确定则留空。会议 end_at 缺失可留空。\n"
        "负责人和 recipients 的 open_id 只能来自 mentions、source_open_id 或 known_users；"
        "如果消息中提到的人名在 known_users 中有对应 open_id，必须使用该 open_id；"
        "owner_name 使用 mentions 或 known_users 中的 name。\n"
        "metadata 必须包含 public_scope，团队公开事项为 team，个人事项不要输出。\n"
        "如果当前消息是在补充或修正已有看板项，设置 merge_item_id 为 existing_items 中对应 id。\n"
        "如果近期上下文里多条消息描述同一个公开事项，把时间、地点、负责人等合并到同一个 item。\n"
        "会议 recipients 只写需要看到候选卡片的人；若发送者明显是参与人，可包含 source_open_id。\n"
        "只有公开任务且负责人和时间明确时，should_create_personal_todo 才为 true。\n\n"
        f"当前时间：{now.isoformat(timespec='minutes')}\n"
        f"chat_id：{chat_id}\n"
        f"source_open_id：{source_open_id}\n"
        f"mentions：{json.dumps(mention_lines, ensure_ascii=False)}\n"
        f"known_users（名字→open_id，可用于解析消息中直接写出的人名）："
        f"{json.dumps(known_users, ensure_ascii=False)}\n"
        f"existing_items：{json.dumps(_existing_payload(existing_items), ensure_ascii=False)}\n"
        f"近期上下文：{json.dumps(context, ensure_ascii=False)}\n"
        f"当前消息：{text}"
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


def _remind_at(due_at: datetime) -> datetime:
    return due_at - timedelta(minutes=10)


def _metadata(candidate: BoardCandidate) -> dict[str, Any]:
    value = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    data = dict(value)
    data["confidence"] = candidate.confidence
    data["reason"] = candidate.reason
    return data


def _recent_context(chat_id: str, now: datetime) -> list[dict[str, Any]]:
    start = (now - timedelta(hours=6)).timestamp()
    events = iter_user_messages_for_chat(chat_id, start)[-20:]
    context: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload", {})
        context.append(
            {
                "message_id": str(payload.get("message_id") or event.get("id") or ""),
                "text": str(payload.get("text") or "")[:300],
                "source_open_id": str(payload.get("source_open_id") or payload.get("user_id") or ""),
                "mentions": payload.get("mentions") or [],
            }
        )
    return context


def _existing_open_items(chat_id: str) -> list[GroupBoardItem]:
    return [
        item for item in group_board_store.list(chat_id=chat_id)
        if item.status in {"open", "pending_confirmation"}
    ][-20:]


def _existing_payload(items: list[GroupBoardItem]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "item_type": item.item_type,
            "title": item.title,
            "due_at": item.due_at,
            "status": item.status,
            "owner_open_id": item.owner_open_id,
            "source_text": item.source_text,
            "metadata": _json_dict(item.metadata_json),
        }
        for item in items
    ]


def _merge_target(
    candidate: BoardCandidate,
    existing_items: list[GroupBoardItem],
    chat_id: str,
    candidate_start: datetime | None = None,
) -> GroupBoardItem | None:
    if candidate.merge_item_id <= 0:
        return None
    for item in existing_items:
        if item.id != candidate.merge_item_id or item.chat_id != chat_id or item.item_type != candidate.item_type:
            continue
        if candidate.item_type == "meeting" and not _same_meeting_day(item, candidate_start):
            return None
        return item
    return None


def _infer_meeting_merge_target(
    candidate: BoardCandidate,
    existing_items: list[GroupBoardItem],
    text: str,
    now: datetime,
) -> GroupBoardItem | None:
    meetings = [item for item in existing_items if item.item_type == "meeting" and item.status == "pending_confirmation"]
    if not meetings:
        return None
    prompt = (
        "判断当前会议候选是否是在补充或修正 existing_items 中的某个会议。\n"
        "必须根据语义、上下文、时间、主题、地点、负责人等综合判断；不要使用关键词规则。\n"
        "只有当前消息与已有会议描述的是同一事项时才能合并。\n"
        "如果当前消息独立描述了新的参会对象、主题、业务事项或人员，即使时间接近，也必须返回 0。\n"
        "地点、时间、主持人、参与人变更只有在语义上属于同一会议时才合并。\n"
        "如果是补充或修正已有会议，返回对应 merge_item_id；否则返回 0。\n\n"
        f"当前时间：{now.isoformat(timespec='minutes')}\n"
        f"当前消息：{text}\n"
        f"会议候选：{json.dumps(_model_data(candidate), ensure_ascii=False)}\n"
        f"existing_items：{json.dumps(_existing_payload(meetings), ensure_ascii=False)}"
    )
    try:
        decision = invoke_structured("group_board_merge", MergeDecision, prompt, timeout=20, max_retries=1)
    except Exception:
        return None
    return _merge_target(
        BoardCandidate(
            merge_item_id=decision.merge_item_id,
            item_type="meeting",
            start_at=candidate.start_at,
        ),
        meetings,
        meetings[0].chat_id,
        _parse_iso_time(candidate.start_at),
    )


def _model_data(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _same_meeting_day(item: GroupBoardItem, candidate_start: datetime | None) -> bool:
    if candidate_start is None:
        return True
    metadata = _json_dict(item.metadata_json)
    item_start = _parse_iso_time(str(metadata.get("start") or item.due_at or ""))
    if item_start is None:
        return True
    return item_start.date() == candidate_start.date()


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


def _mention_names(mentions: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in mentions:
        open_id = str(item.get("open_id") or "").strip()
        name = str(item.get("name") or item.get("key") or "").strip()
        if open_id and name:
            result[open_id] = name.lstrip("@")
    return result


def _owner_name(candidate: BoardCandidate, mention_names: dict[str, str]) -> str:
    name = candidate.owner_name.strip().lstrip("@")
    if name:
        return name
    return mention_names.get(candidate.owner_open_id.strip(), "")


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


def _build_known_users(
    source_open_id: str,
    mentions: list[dict],
    context: list[dict[str, Any]],
) -> dict[str, str]:
    """Build name -> open_id map from current mentions and context message mentions."""
    result: dict[str, str] = {}
    all_mentions: list[dict] = list(mentions)
    for ctx_msg in context:
        for mention in ctx_msg.get("mentions") or []:
            if isinstance(mention, dict):
                all_mentions.append(mention)
    for mention in all_mentions:
        open_id = str(mention.get("open_id") or "").strip()
        name = str(mention.get("name") or mention.get("key") or "").strip().lstrip("@")
        if open_id and name:
            result[name] = open_id
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
