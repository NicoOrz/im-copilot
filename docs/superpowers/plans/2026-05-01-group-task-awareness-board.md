# Group Task Awareness Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MVP support for group task awareness, public group board summaries, and meeting review confirmation prompts.

**Architecture:** Reuse the existing group-message passive memory path, todo store, reminder worker, and `/todo` command. Add a small group board store and extractor so public assignments and meeting intents are separated from private personal todos.

**Tech Stack:** Python, sqlite3, existing Feishu bot wrapper, existing command and memory modules.

---

## File Structure

- Create `src/im_copilot/memory/group_board_store.py`
  - Persist group board items in the same sqlite database family as todos.
  - Store item type, title, owner, status, source message, and optional meeting fields.

- Create `src/im_copilot/memory/group_board_extractor.py`
  - Classify group messages into public assignments and meeting/review candidates.
  - Store board items and optional personal todo copies for public assignments.

- Modify `src/im_copilot/memory/todo_extractor.py`
  - Add confidence to todo drafts.
  - Keep high-confidence personal todo behavior.
  - Provide a helper that can be reused for public assignment personal copies.

- Modify `src/im_copilot/memory/summary_worker.py`
  - Show group board items in `/todo summary today`.
  - Exclude ordinary private personal commitments from group summary.

- Modify `src/im_copilot/memory/chat_sync.py`
  - Re-extract group board items during `/todo sync`.

- Modify `src/im_copilot/commands.py`
  - Keep `/todo` personal list private in group chats.
  - Add `/todo board` as alias for group summary.

- Modify `src/im_copilot/lark_handlers.py`
  - In passive group path, run group board extraction after private todo extraction.
  - Send meeting/review confirmation prompt to related users by single chat message.

## Task 1: Group Board Store

**Files:**
- Create: `src/im_copilot/memory/group_board_store.py`

- [x] Define `GroupBoardItem` dataclass.
- [x] Create sqlite table `group_board_items`.
- [x] Add `GroupBoardStore.create`.
- [x] Add `GroupBoardStore.created_between`.
- [x] Add `GroupBoardStore.list`.
- [x] Verify with `python -m py_compile src/im_copilot/memory/group_board_store.py`.

## Task 2: Todo Draft Confidence and Public Assignment Helper

**Files:**
- Modify: `src/im_copilot/memory/todo_extractor.py`

- [x] Add `confidence` and `needs_confirmation` to `TodoDraft`.
- [x] Keep current high-confidence extraction semantics.
- [x] Add `store_todo_draft` helper so group board extractor can create personal copies.
- [x] Verify with `python -m py_compile src/im_copilot/memory/todo_extractor.py`.

## Task 3: Group Board Extractor

**Files:**
- Create: `src/im_copilot/memory/group_board_extractor.py`

- [x] Add public assignment detection for “负责/跟进/完成/整理/提交/发送/更新/确认” with explicit mention or assignee name.
- [x] Add meeting/review detection for “会议/评审/同步/讨论/约”.
- [x] Store public assignments in group board.
- [x] Store meeting candidates in group board with status `pending_confirmation`.
- [x] Create personal todo copy for public assignments with explicit assignee and time.
- [x] Verify with `python -m py_compile src/im_copilot/memory/group_board_extractor.py`.

## Task 4: Summary and Sync Commands

**Files:**
- Modify: `src/im_copilot/memory/summary_worker.py`
- Modify: `src/im_copilot/memory/chat_sync.py`
- Modify: `src/im_copilot/commands.py`

- [x] Update summary output to show chat messages and group board items.
- [x] Keep private todos out of group summaries.
- [x] Update sync to run both private todo and group board extraction.
- [x] Add `/todo board` command.
- [x] In group chat `/todo`, return guidance instead of listing private todos.
- [x] Verify with `python -m py_compile src/im_copilot/memory/summary_worker.py src/im_copilot/memory/chat_sync.py src/im_copilot/commands.py`.

## Task 5: Message Handler Wiring

**Files:**
- Modify: `src/im_copilot/lark_handlers.py`

- [x] Import group board extraction only inside `_process_message`.
- [x] Run board extraction for passive group messages.
- [x] Send private confirmation text for meeting candidates to mentioned related users.
- [x] Keep group chat silent for personal todos.
- [x] Verify with `python -m py_compile src/im_copilot/lark_handlers.py`.

## Task 6: Final Verification

**Files:**
- Existing test suite only.

- [x] Run `uv run python -m unittest`.
- [x] Run `git diff --stat`.
- [x] Confirm no unrelated files are changed.
