# Group History Polling Design

## Goal

Capture ordinary group messages that Feishu does not push through `im.message.receive_v1` when the bot is not mentioned.

## Design

The bot keeps the existing WebSocket path for single chats, mentions, commands, and card callbacks. A new background worker discovers group chats where the bot is present, then polls recent history through `GET /open-apis/im/v1/messages`.

The group list has two sources. On startup, the worker calls `GET /open-apis/im/v1/chats` and stores every normal group. At runtime, the WebSocket handler listens for `im.chat.member.bot.added_v1` and records the new `chat_id`.

Each polled message uses `message_id` as the idempotency key. User text messages are stored as local `user_message` events and then passed through the existing personal todo extractor and group board extractor. The worker does not send public group replies.

## Configuration

- `LARK_GROUP_HISTORY_ENABLED=1` enables polling.
- `LARK_GROUP_HISTORY_POLL_SECONDS` controls interval, default `60`.
- `LARK_GROUP_HISTORY_LOOKBACK_SECONDS` controls the first read window, default `300`.

## Permissions

Tenant scopes:

- `im:chat:read`
- `im:message`
- `im:message.group_msg`
- `im:message:send_as_bot`
- `im:message.reactions:write_only`
- `im:chat.members:read`

Event subscriptions:

- `im.message.receive_v1`
- `card.action.trigger`
- `im.chat.member.bot.added_v1`
