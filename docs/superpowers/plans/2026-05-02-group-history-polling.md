# Group History Polling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a history polling path so non-mentioned group messages can feed existing todo and group board extraction.

**Architecture:** Keep WebSocket for real-time interactive paths. Add a sqlite-backed group chat store, LarkBot methods for chat discovery and history reads, and a background worker that polls known groups and reuses the existing extractors.

**Tech Stack:** Python, sqlite3, lark-oapi SDK, existing Feishu bot wrapper.

---

## File Structure

- Create `src/im_copilot/memory/group_chat_store.py`
  - Persist known bot group chats and last message cursor time.

- Create `src/im_copilot/memory/group_history_worker.py`
  - Discover bot groups, poll message history, dedupe by `message_id`, and run existing extractors.

- Modify `src/im_copilot/lark_bot.py`
  - Add methods to list bot chats and list chat messages.

- Modify `src/im_copilot/lark_handlers.py`
  - Register `im.chat.member.bot.added_v1`.
  - Record newly joined groups in the group chat store.

- Modify `src/im_copilot/web/app.py`
  - Start the group history worker when enabled.

## Tasks

- [x] Add design spec.
- [x] Add implementation plan.
- [x] Add group chat store.
- [x] Add LarkBot chat and message history methods.
- [x] Add group history worker.
- [x] Register bot-added event.
- [x] Start worker from app startup.
- [x] Compile changed files.
- [x] Run existing unittest suite.
