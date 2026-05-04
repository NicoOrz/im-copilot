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
from im_copilot.memory.todo_store import TodoRecord, todo_store

logger = logging.getLogger(__name__)
_TZ = ZoneInfo("Asia/Hong_Kong")
_MIN_CONFIDENCE = 0.65
_WINDOW_MINUTES = 30
_WINDOW_MIN_MESSAGES = 50
_OPEN_STATUSES = ("pending", "reminded", "awaiting_confirmation")


@dataclass
class WindowMessage:
    message_id: str
    open_id: str
    name: str
    text: str
    ts: float
    is_trigger: bool
    mentions: tuple[dict[str, str], ...] = ()


@dataclass
class ExistingTodoBrief:
    id: int
    assignee_open_id: str
    title: str
    action_phrase: str
    due_at: str
    status: str


class TodoExtractionItem(BaseModel):
    is_todo: bool = False
    links_to_existing_id: str = ""
    assignee_open_id: str = ""
    title: str = ""
    action_phrase: str = ""
    due_at: str = ""
    remind_at: str = ""
    confidence: float = 0.0
    needs_confirmation: bool = False
    scope: Literal["personal", "team", "none"] = "none"
    reasoning: str = ""


class TodoExtractionOutput(BaseModel):
    items: list[TodoExtractionItem] = Field(default_factory=list)


@dataclass
class TodoDraft:
    assignee_open_id: str
    title: str
    action_phrase: str
    due_at: datetime
    remind_at: datetime
    source_text: str
    confidence: float = 1.0
    needs_confirmation: bool = False


@dataclass
class TodoUpdate:
    existing_id: int
    title: str
    action_phrase: str
    due_at: datetime | None
    remind_at: datetime | None
    source_text: str
    confidence: float = 1.0
    needs_confirmation: bool = False


def assemble_window(
    chat_id: str,
    trigger_event_id: int | None,
    now: datetime | None = None,
) -> list[WindowMessage]:
    current_time = now or datetime.now(_TZ)
    recent_since = (current_time - timedelta(minutes=_WINDOW_MINUTES)).timestamp()
    recent = recent_user_messages_for_chat(
        chat_id,
        before_event_id=trigger_event_id,
        limit=10_000,
        since_ts=recent_since,
    )
    if len(recent) > _WINDOW_MIN_MESSAGES:
        return _events_to_window(recent, trigger_event_id)
    events = recent_user_messages_for_chat(
        chat_id,
        before_event_id=trigger_event_id,
        limit=_WINDOW_MIN_MESSAGES,
        since_ts=0.0,
    )
    return _events_to_window(events, trigger_event_id)


def assemble_window_from_events(
    events: list[dict[str, Any]],
    trigger_event_id: int | None = None,
) -> list[WindowMessage]:
    filtered = [
        event for event in sorted(events, key=lambda item: int(item.get("id") or 0))
        if event.get("event_type") == "user_message"
    ]
    if trigger_event_id is not None:
        filtered = [
            event for event in filtered
            if int(event.get("id") or 0) <= trigger_event_id
        ]
    if not filtered:
        return []
    trigger = filtered[-1]
    trigger_ts = float(trigger.get("created_at") or 0.0)
    recent_since = trigger_ts - _WINDOW_MINUTES * 60
    recent = [
        event for event in filtered
        if float(event.get("created_at") or 0.0) >= recent_since
    ]
    selected = recent if len(recent) > _WINDOW_MIN_MESSAGES else filtered[-_WINDOW_MIN_MESSAGES:]
    return _events_to_window(selected, int(selected[-1].get("id") or 0))


def load_open_todos_brief(chat_id: str) -> list[ExistingTodoBrief]:
    records: list[TodoRecord] = []
    for status in _OPEN_STATUSES:
        records.extend(todo_store.list(chat_id=chat_id, status=status))
    return [
        ExistingTodoBrief(
            id=record.id,
            assignee_open_id=record.assignee_open_id,
            title=record.title,
            action_phrase=record.action_phrase,
            due_at=record.due_at,
            status=record.status,
        )
        for record in sorted(records, key=lambda item: item.id)
    ]


def extract_todos_from_window(
    window: list[WindowMessage],
    *,
    existing_open_todos: list[ExistingTodoBrief],
    is_bot_request: bool = False,
    now: datetime | None = None,
) -> list[TodoDraft | TodoUpdate]:
    window = _clean_window(window)
    if not window:
        return []
    open_ids = _window_open_ids(window)
    if not open_ids:
        return []
    current_time = now or datetime.now(_TZ)
    try:
        output = _invoke_window_llm(
            window,
            existing_open_todos=existing_open_todos,
            open_ids=open_ids,
            is_bot_request=is_bot_request,
            now=current_time,
        )
    except Exception:
        logger.exception("todo window extraction failed")
        return []

    existing_by_id = {str(todo.id): todo for todo in existing_open_todos}
    source_text = _source_text(window)
    results: list[TodoDraft | TodoUpdate] = []
    for item in output.items:
        if not item.is_todo or item.scope == "none":
            continue
        assignee = item.assignee_open_id.strip()
        if assignee not in open_ids:
            logger.warning("todo extraction dropped unknown assignee: assignee=%s", assignee)
            continue
        link_id = item.links_to_existing_id.strip()
        if link_id and link_id not in existing_by_id:
            logger.warning("todo extraction ignored unknown existing id: id=%s", link_id)
            link_id = ""
        if not link_id and item.confidence < _MIN_CONFIDENCE:
            continue

        due_at = _parse_iso_time(item.due_at)
        remind_at = _parse_iso_time(item.remind_at) or _default_remind_at(due_at)
        title = item.title.strip()
        action_phrase = item.action_phrase.strip()
        if link_id:
            results.append(
                TodoUpdate(
                    existing_id=int(link_id),
                    title=title,
                    action_phrase=action_phrase,
                    due_at=due_at,
                    remind_at=remind_at,
                    source_text=source_text,
                    confidence=item.confidence,
                    needs_confirmation=item.needs_confirmation,
                )
            )
            continue

        if not assignee or not title or due_at is None or remind_at is None:
            continue
        results.append(
            TodoDraft(
                assignee_open_id=assignee,
                title=title[:80],
                action_phrase=action_phrase[:80],
                due_at=due_at,
                remind_at=remind_at,
                source_text=source_text,
                confidence=item.confidence,
                needs_confirmation=item.needs_confirmation,
            )
        )
    return results


def extract_and_store_todos_from_window(
    *,
    chat_id: str,
    message_id: str,
    source_open_id: str,
    window: list[WindowMessage],
    existing_open_todos: list[ExistingTodoBrief],
    source: str = "feishu",
    is_bot_request: bool = False,
    now: datetime | None = None,
) -> list[TodoRecord]:
    records: list[TodoRecord] = []
    items = extract_todos_from_window(
        window,
        existing_open_todos=existing_open_todos,
        is_bot_request=is_bot_request,
        now=now,
    )
    for item in items:
        if isinstance(item, TodoUpdate):
            record = todo_store.update_fields(
                item.existing_id,
                title=item.title or None,
                action_phrase=item.action_phrase or None,
                due_at=_iso_minutes(item.due_at) if item.due_at else None,
                remind_at=_iso_minutes(item.remind_at) if item.remind_at else None,
            )
            if record is None:
                continue
            record_event(
                chat_id,
                source,
                "todo_updated",
                _event_payload(record, message_id, source_open_id, item.source_text, item.confidence, item.needs_confirmation),
            )
            records.append(record)
            continue

        status = "awaiting_confirmation" if item.needs_confirmation else "pending"
        record = todo_store.create(
            chat_id=chat_id,
            message_id=message_id,
            source_open_id=source_open_id,
            assignee_open_id=item.assignee_open_id,
            title=item.title,
            action_phrase=item.action_phrase,
            due_at=item.due_at.isoformat(timespec="minutes"),
            remind_at=item.remind_at.isoformat(timespec="minutes"),
            source_text=item.source_text,
            status=status,
        )
        if record is None:
            continue
        record_event(
            chat_id,
            source,
            "todo_detected",
            _event_payload(record, message_id, source_open_id, item.source_text, item.confidence, item.needs_confirmation),
        )
        records.append(record)
    return records


def _invoke_window_llm(
    window: list[WindowMessage],
    *,
    existing_open_todos: list[ExistingTodoBrief],
    open_ids: set[str],
    is_bot_request: bool,
    now: datetime,
) -> TodoExtractionOutput:
    prompt = (
        "你是群聊待办抽取器，输入是一段对话窗口和当前 chat 内未完成待办列表。\n"
        "只为触发消息或它直接修改、确认的内容输出新待办或更新已有待办。\n"
        "assignee_open_id 必须出自窗口内可选 open_id 集合，集合包含消息发送者和 mentions 中的 open_id。\n"
        "如果触发消息里同时给出了被指派人、指派语义动词、截止时间或可推断的截止三要素，"
        "必须输出 is_todo=true，title 为该任务的具体动作短语；不要因为细节不全输出 false。"
        "指派语义包括但不限于动词式祈使（“补”、“出”、“准备”、“提交”等被语义识别为请求行动的句式），"
        "但识别只能凭语义，不要靠关键词列表。"
        "反例：触发消息只是闲聊、表态、回复“收到”/“好的”/“可以”且没有引入新可执行内容，输出 is_todo=false。\n"
        "选 assignee 的优先级，从高到低："
        "1. 如果触发消息内含 @ mention，且消息语义是把任务交给被提及人或讨论被提及人需做的事，"
        "assignee 必须是被提及人。"
        "2. 如果触发消息是在确认/调整某个先前承诺（典型：“那就明早10点前”、“OK”、“行”），"
        "回到窗口里找出真正的承诺者，assignee 必须是承诺者，不能是当前说话人。"
        "3. 如果触发消息是说话人对自己的承诺/计划（典型：“我来准备...”），assignee = 说话人。"
        "4. 任何情况下 assignee_open_id 必须出自窗口 open_id 白名单。"
        "反例（必须避免）：消息是“@A 那就明早10点前”，assignee 选成说话人 B 是错的；"
        "正确：assignee=A，并 link 到 A 之前承诺的那条 todo。\n"
        "title 必须是具体动作短语，包含动词和宾语；禁止输出占位语句，例如“X的待办”“完成相关任务”。\n"
        "action_phrase 是该待办对应的动作短语（动词+宾语），用于在卡片或提醒里清晰展示要做什么。"
        "范例：“提交 Q2 数据口径”、“补一页效率数据到 PPT”、“确认隐私权限边界”。"
        "绝对不要输出 create / update / new 这种操作动词，那不是动作描述。\n"
        "links_to_existing_id 仅当触发消息和某条 existing todo 是严格语义上同一件事"
        "（同一个负责人 + 同一个动作 + 不同的截止时间或细节微调）时才使用，"
        "作用是更新该 todo 的 due_at / 细节。"
        "以下情况禁止使用 links_to_existing_id：触发消息引入了一个新的子任务（即便主题相关），"
        "例如 existing 17=\"准备方案和PPT\" / 触发=\"赵磊补一页效率数据\" 时应新建 todo for 赵磊，"
        "不要改 17；触发消息是不同负责人对相关主题的独立工作；触发消息只是讨论、表态、评论，"
        "不是对既有承诺的具体修改。"
        "反例（必须避免）：把所有“汇报准备”相关讨论都 link 到同一条 master todo 让其 title 滚雪球。\n"
        "如果窗口内信息不足以判断动作内容，输出 is_todo=false，不要硬凑 title 或 action_phrase。\n"
        "如果 existing_open_todos 里没有对应事项，但窗口足以判断动作内容和承诺人，可以新建并归给真正承诺人。\n"
        "如果触发消息只是寒暄、表态、确认时间，且对应承诺已在 existing_open_todos 里，"
        "输出 is_todo=true 并设置 links_to_existing_id，用输出字段表达需要变更的时间或内容。\n"
        "如果 is_bot_request=true，表示用户在要求机器人生成文档、PPT、画板、总结或执行工具；"
        "这类请求由 Agent 执行，不创建个人待办，除非明确为某个真实成员独立创建待办。\n"
        "输出 items 数组；每个 item 对应一个新待办、一个已有待办更新，或一个明确丢弃判断。\n\n"
        f"当前时间：{now.isoformat(timespec='minutes')}\n"
        f"is_bot_request：{is_bot_request}\n"
        f"窗口内可选 open_id：{json.dumps(sorted(open_ids), ensure_ascii=False)}\n"
        f"窗口 mentions：{json.dumps(_window_mentions_payload(window), ensure_ascii=False)}\n"
        f"existing_open_todos：\n{_existing_todos_text(existing_open_todos)}\n\n"
        f"对话窗口：\n{_window_prompt_text(window)}"
    )
    return invoke_structured("todo_extractor", TodoExtractionOutput, prompt, timeout=60, max_retries=1)


def _events_to_window(
    events: list[dict[str, Any]],
    trigger_event_id: int | None,
) -> list[WindowMessage]:
    if not events:
        return []
    name_by_open_id: dict[str, str] = {}
    event_mentions: list[tuple[dict[str, str], ...]] = []
    for event in events:
        mentions = _mention_items(event.get("payload", {}).get("mentions") or [])
        event_mentions.append(mentions)
        for mention in mentions:
            open_id = mention.get("open_id", "")
            name = mention.get("name", "")
            if open_id and name:
                name_by_open_id[open_id] = name

    if trigger_event_id is not None:
        events = [
            event for event in events
            if int(event.get("id") or 0) <= trigger_event_id
        ]
        event_mentions = event_mentions[:len(events)]
    if not events:
        return []

    messages: list[WindowMessage] = []
    for index, event in enumerate(events):
        payload = event.get("payload", {})
        open_id = str(payload.get("source_open_id") or payload.get("user_id") or payload.get("speaker_open_id") or "").strip()
        messages.append(
            WindowMessage(
                message_id=str(payload.get("message_id") or event.get("id") or ""),
                open_id=open_id,
                name=name_by_open_id.get(open_id, ""),
                text=_clean_text(str(payload.get("text") or payload.get("message") or "")),
                ts=float(event.get("created_at") or 0.0),
                is_trigger=index == len(events) - 1,
                mentions=event_mentions[index] if index < len(event_mentions) else (),
            )
        )
    return messages


def _mention_items(raw_mentions: object) -> tuple[dict[str, str], ...]:
    if not isinstance(raw_mentions, list):
        return ()
    result: list[dict[str, str]] = []
    for item in raw_mentions:
        if not isinstance(item, dict):
            continue
        identity = item.get("id")
        open_id = str(item.get("open_id") or "").strip()
        if not open_id and isinstance(identity, dict):
            open_id = str(identity.get("open_id") or "").strip()
        name = str(item.get("name") or item.get("text") or "").strip()
        key = str(item.get("key") or "").strip()
        if open_id:
            result.append({"open_id": open_id, "name": name, "key": key})
    return tuple(result)


def _clean_window(window: list[WindowMessage]) -> list[WindowMessage]:
    cleaned: list[WindowMessage] = []
    for index, message in enumerate(window):
        text = _clean_text(message.text)
        open_id = message.open_id.strip()
        if not text or not open_id:
            continue
        cleaned.append(
            WindowMessage(
                message_id=message.message_id,
                open_id=open_id,
                name=message.name.strip(),
                text=text,
                ts=message.ts,
                is_trigger=False,
                mentions=tuple(message.mentions or ()),
            )
        )
    if not cleaned:
        return []
    return [
        WindowMessage(
            message_id=message.message_id,
            open_id=message.open_id,
            name=message.name,
            text=message.text,
            ts=message.ts,
            is_trigger=index == len(cleaned) - 1,
            mentions=message.mentions,
        )
        for index, message in enumerate(cleaned)
    ]


def _window_open_ids(window: list[WindowMessage]) -> set[str]:
    open_ids = {message.open_id.strip() for message in window if message.open_id.strip()}
    for message in window:
        for mention in message.mentions:
            open_id = str(mention.get("open_id") or "").strip()
            if open_id:
                open_ids.add(open_id)
    return open_ids


def _window_prompt_text(window: list[WindowMessage]) -> str:
    lines = []
    for index, message in enumerate(window, start=1):
        marker = " ← TRIGGER" if message.is_trigger else ""
        lines.append(
            f"[{index}] open_id={message.open_id} name={message.name or '-'} "
            f"ts={_iso_from_ts(message.ts)} {message.text}{marker}"
        )
    return "\n".join(lines)


def _window_mentions_payload(window: list[WindowMessage]) -> list[dict[str, str]]:
    seen: set[str] = set()
    mentions: list[dict[str, str]] = []
    for message in window:
        for mention in message.mentions:
            open_id = str(mention.get("open_id") or "").strip()
            if not open_id or open_id in seen:
                continue
            seen.add(open_id)
            mentions.append(
                {
                    "open_id": open_id,
                    "name": str(mention.get("name") or "").strip(),
                    "key": str(mention.get("key") or "").strip(),
                }
            )
    return mentions


def _existing_todos_text(todos: list[ExistingTodoBrief]) -> str:
    if not todos:
        return "(空)"
    return "\n".join(
        f'id={todo.id} assignee={todo.assignee_open_id} status={todo.status} '
        f'due_at={todo.due_at} title="{_quote(todo.title)}" action_phrase="{_quote(todo.action_phrase)}"'
        for todo in todos
    )


def _source_text(window: list[WindowMessage]) -> str:
    return "\n".join(
        f"{_iso_from_ts(message.ts)} {message.name or message.open_id}: {message.text}"
        f"{' ← TRIGGER' if message.is_trigger else ''}"
        for message in window
    )


def _event_payload(
    record: TodoRecord,
    message_id: str,
    source_open_id: str,
    source_text: str,
    confidence: float,
    needs_confirmation: bool,
) -> dict[str, Any]:
    return {
        "id": record.id,
        "chat_id": record.chat_id,
        "message_id": message_id,
        "source_open_id": source_open_id,
        "assignee_open_id": record.assignee_open_id,
        "title": record.title,
        "action_phrase": record.action_phrase,
        "due_at": record.due_at,
        "remind_at": record.remind_at,
        "status": record.status,
        "source_text": source_text,
        "confidence": confidence,
        "needs_confirmation": needs_confirmation,
    }


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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


def _default_remind_at(due_at: datetime | None) -> datetime | None:
    if due_at is None:
        return None
    return due_at - timedelta(minutes=10)


def _iso_minutes(value: datetime | None) -> str:
    return value.isoformat(timespec="minutes") if value else ""


def _iso_from_ts(value: float) -> str:
    if value <= 0:
        return ""
    return datetime.fromtimestamp(value, _TZ).isoformat(timespec="minutes")


def _quote(value: str) -> str:
    return str(value or "").replace('"', '\\"')
