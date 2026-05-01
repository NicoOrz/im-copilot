---
name: lark-im-bot-debugging
description: Use when debugging Feishu/Lark IM bot message handling, WebSocket event retries, card action callbacks, card JSON 2.0 errors, LangGraph interrupts, duplicate replies, stale message replays, or sqlite checkpointer resume failures.
---

# Lark IM Bot Debugging

## Core Principle

Treat Feishu/Lark IM bots as an async event system with strict callback deadlines. Do not run slow LLM, LangGraph, or card update work inside SDK event callbacks. First confirm whether the event reaches local code, then isolate card JSON, callback subscription, deduplication, and LangGraph persistence separately.

## Required Sources

- Use `mcp__lark_open_doc_search__` for current Feishu/Lark card and IM callback docs.
- Use codebase retrieval before editing message handlers, card builders, session managers, or graph resume logic.
- Prefer official Open Platform docs over memory for error codes and Card JSON 2.0 schema.

## Failure Map

| Symptom | Likely Cause | Fix Pattern |
|---|---|---|
| Same message replies twice | Feishu retry after callback exceeds 3s or in-memory dedup lost on restart | Return quickly; process in background; persist processed `message_id` |
| Old message processed after restart | Dedup stored only in memory; Feishu redelivered unacknowledged event | Store `message_id` in local DB with TTL |
| WebSocket `ping_timeout` | Handler blocks SDK loop with LLM/graph work | Spawn daemon worker and return from callback |
| `processor not found: im.message.message_read_v1` | App subscribed to read event but handler not registered | Register empty `message_read_v1` handler or unsubscribe event |
| `card action 200340` | Card callback not configured, wrong callback style, or callback exceeds 3s | Verify `card.action.trigger` subscription, use `behaviors.callback`, return toast immediately |
| `unsupported tag action` | Card JSON 2.0 still uses JSON 1.0 `tag: action` container | Put `button` elements directly in `body.elements` |
| Empty final card `处理完成` | Stream loop kept only last update, losing `summary` | Accumulate update dicts into final state |
| Click approval says session expired | Session deleted after interrupt | On interrupt, update card then return; do not finalize or delete session |
| `Cannot operate on a closed database` | Saved graph uses checkpointer whose context manager closed | Rebuild graph with fresh sqlite checkpointer on resume |
| Non-chat card creation failure goes silent | Fallback invokes interrupting graph without session/card | Send error text or create a non-streaming interactive card path |

## Event Callback Pattern

Use this shape for `im.message.receive_v1`:

```python
def on_message_receive(data, lark_bot):
    text, chat_id, message_id = parse_event(data)
    if not persistent_dedup_mark(message_id):
        return
    threading.Thread(
        target=process_message,
        args=(text, chat_id, message_id, lark_bot),
        daemon=True,
    ).start()
```

Rules:
- The SDK callback should only parse, validate, dedup, and start background work.
- Do not call LLM, `graph.invoke`, `graph.stream`, or OpenAPI card updates directly in the callback.
- Use `message.message_id` as idempotency key. Memory-only dedup is insufficient.

## Card Action Callback Pattern

For `card.action.trigger`:

```python
def on_card_action(data, lark_bot):
    action_value = data.event.action.value or {}
    thread_id = data.event.context.open_chat_id
    decision = build_decision(action_value)
    threading.Thread(
        target=resume_card_action,
        args=(thread_id, decision, lark_bot),
        daemon=True,
    ).start()
    return toast_response("已收到您的反馈")
```

Rules:
- Return a valid response within 3 seconds.
- If the log does not show the handler entry, check Developer Console callback subscription first.
- Subscribe to **new** `card.action.trigger` via long connection. Remove legacy `card.action.trigger_v1` and historical bot card request URLs unless intentionally supported.
- Generate a new card after changing JSON; old cards keep old behavior.

## Card JSON 2.0 Rules

- Always declare `"schema": "2.0"` when using CardKit 2.0.
- Do not use `{"tag": "action"}`. JSON 2.0 no longer supports it.
- Put buttons directly in `body.elements` or supported containers.
- For server callback behavior, configure:

```json
{
  "tag": "button",
  "text": {"tag": "plain_text", "content": "同意"},
  "type": "primary",
  "behaviors": [
    {"type": "callback", "value": {"action": "approve"}}
  ]
}
```

- `value` alone may appear in examples, but `behaviors.callback` is the explicit request callback event pattern.

## LangGraph Interrupt Rules

When streaming a graph that may interrupt:

```python
for step in graph.stream(initial_state, config=config, stream_mode="updates"):
    if "__interrupt__" in step:
        save_session(thread_id, config, card_message_id, interrupt_info)
        update_card_for_interrupt(...)
        return
    merge_update_into_final_state(step)

finalize_card(final_state)
delete_session(thread_id)
```

Rules:
- Do not `break` on interrupt and continue to finalization.
- Do not delete session after interrupt.
- Preserve enough session metadata for the click handler: `thread_id`, `config`, `card_message_id`, `last_interrupt`.
- Avoid storing a compiled graph tied to a soon-closed sqlite checkpointer; rebuild graph with a fresh checkpointer on resume.

## SQLite Checkpointer Pattern

If initial processing uses:

```python
with get_checkpointer("sqlite") as checkpointer:
    graph = build_pipeline(checkpointer=checkpointer)
```

then any graph saved outside the `with` block is unsafe. Resume with:

```python
with get_checkpointer("sqlite") as checkpointer:
    graph = build_pipeline(checkpointer=checkpointer)
    result = graph.invoke(Command(resume=decision), config=config)
```

Store `config`, not the closed graph, as the durable resume handle.

## Verification Checklist

Before reporting completion:

- Compile touched files with `python -m compileall`.
- Trigger a new incoming message after restart; old cards/messages are not valid verification artifacts.
- Confirm no duplicate reply for the same `message_id` after restarting the bot.
- Confirm callback entry log appears for card clicks.
- Confirm approval interrupt card remains visible until click, and session is not deleted before click.
- Confirm final card uses accumulated graph state, including `summary`.
