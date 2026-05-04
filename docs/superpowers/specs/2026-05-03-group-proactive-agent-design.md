# Group Proactive Agent Design

**Date:** 2026-05-03
**Branch:** feature/group-task-awareness-board

## Context

群看板（Group Board）已能从群聊消息中自动提取任务、会议、决议、风险等事项，并支持按需查询今日看板。本次设计在此基础上增加两个主动能力：

1. **EOD 自动总结**：每天下班前自动向活跃群发送全天看板总结 + 明日待办预告，无需用户主动 @bot 查询。
2. **Proactive Worker**：在群聊沉默一段时间后，LLM 判断讨论是否有值得主动介入的场景（总结结论、澄清盲点、推荐下一步），若有则主动发群消息。
3. **产物建议卡片**：EOD 总结或看板查询时，若会议材料成熟，附上卡片按钮供用户一键触发生成文稿、画板或 PPT。

子系统 B（看板事项跨天沉淀）本次不做。

---

## 子系统 A：EOD 自动总结

### 目标

每天在配置时间（默认 18:00）向所有活跃群发送：
- 全天看板总结（复用现有 `summary_today()`）
- 明日待办预告（查询 `due_at` 在明天的 open 事项）

### 新文件

`src/im_copilot/memory/eod_summary_worker.py`

### 核心逻辑

```
每分钟检查：
  当前时间 >= EOD_TIME 且 今天尚未发送过
    → 对所有活跃群调用 send_eod_summary(chat_id)
    → 记录今日已发送
```

**防重复：** 用 `dict[str, date]` 记录每个群上次发送日期，同一天不重复发送。

**明日预告：** 查询 `group_board_store.list(chat_id)` 中 `due_at` 在明天且 `status=open` 的事项，格式化后追加到 `summary_today()` 输出末尾。

**配置：**
- `LARK_EOD_SUMMARY_TIME`：格式 `HH:MM`，默认 `18:00`
- `LARK_EOD_SUMMARY_ENABLED`：`1` 启用，默认关闭

**启动：** `lark_handlers.py` 的 `build_event_handler()` 中，与 `reminder_worker` 并列启动。

### 复用

- `summary_today(chat_id)` → `src/im_copilot/memory/summary_worker.py`
- `group_board_store.list(chat_id)` → `src/im_copilot/memory/group_board_store.py`
- `group_chat_store.active()` → `src/im_copilot/memory/group_chat_store.py`
- `lark_bot.send_text(chat_id, text)` → `src/im_copilot/lark_bot.py`

---

## 子系统 C：Proactive Worker（沉默窗口触发）

### 目标

群聊沉默 N 分钟后，LLM 判断最近一段讨论是否有值得主动介入的场景，若有则直接发群消息。

**触发场景（LLM 判断）：**
- `summarize`：讨论有明确结论但无人总结
- `clarify`：关键信息缺失、存在模糊假设或盲点、讨论无法推进
- `recommend`：讨论结束后有明显的下一步行动未被认领

### 新文件

`src/im_copilot/memory/proactive_worker.py`

### 状态结构

每个群维护内存状态（进程内，不持久化）：

```python
@dataclass
class ChatProactiveState:
    last_message_ts: float = 0.0   # 最后一条消息时间戳
    triggered: bool = False         # 本次沉默窗口是否已触发
```

### 触发逻辑

```
轮询间隔：60 秒

对每个活跃群：
  if now - last_message_ts >= SILENCE_WINDOW and not triggered:
    取最近 2 小时内、最多 60 条群消息（过滤 bot 消息）
    调用 LLM 判断 ProactiveDecision
    if decision.should_act:
      lark_bot.send_text(chat_id, decision.message)
    triggered = True

当 group_history_worker 检测到新消息时：
  重置 triggered = False
  更新 last_message_ts
```

### LLM 输出结构

```python
class ProactiveDecision(BaseModel):
    should_act: bool
    action_type: Literal["clarify", "summarize", "recommend", "none"]
    message: str    # 直接发送的消息，should_act=False 时为空
    reason: str
```

**Prompt 约束：**
- 只有在非常有必要时才返回 `should_act=True`，默认倾向 `none`
- `clarify`：讨论中存在关键假设未验证、信息缺失、或明显盲点
- `summarize`：讨论有清晰结论但没有人做过总结
- `recommend`：讨论结束后有明显的下一步行动未被认领
- 消息语气自然，不要像机器人播报，像一个参与讨论的同事
- 消息长度控制在 100 字以内

### 状态更新接口

`proactive_worker` 暴露一个函数供 `group_history_worker` 调用：

```python
def notify_new_messages(chat_id: str, latest_ts: float) -> None:
    """group_history_worker 每次轮询到新消息时调用。"""
```

**注意：** 仅由 `group_history_worker` 调用，`lark_handlers._sync_recent_chat_context` 不调用此函数，避免重复触发。

### 配置

- `LARK_PROACTIVE_ENABLED`：`1` 启用，默认关闭
- `LARK_PROACTIVE_SILENCE_MINUTES`：沉默窗口分钟数，默认 `10`
- `LARK_PROACTIVE_CONTEXT_HOURS`：消息上下文时间窗口（小时），默认 `2`
- `LARK_PROACTIVE_CONTEXT_LIMIT`：消息上下文条数上限，默认 `60`

### 启动

`lark_handlers.py` 的 `build_event_handler()` 中启动，与 `reminder_worker` 并列。

---

## 子系统 D：会议材料成熟时建议生成产物

### 目标

EOD 总结或看板查询时，若检测到有足够的会议材料（已确认会议 + 相关讨论），在消息末尾附上卡片按钮，用户点击后触发 Agent 生成对应产物。

### 实现位置

`eod_summary_worker.py` 的 `send_eod_summary()` 末尾，以及 `summary_today()` 的调用方（`lark_handlers.py`）。

### 判断条件

满足以下任一条件时展示建议卡片：
- 当天有 `status=confirmed` 的会议事项
- 当天有 `item_type=decision` 且 `source_text` 超过 200 字的事项

### 卡片结构

使用现有 `lark_card.py` 的 `_base_card()` 模式，包含：
- 标题："会议材料已就绪，建议生成产物"
- 三个按钮：`生成文稿` / `生成画板` / `生成 PPT`
- 对应 `artifact_type`：`doc` / `whiteboard` / `slide`
- 按钮 action 携带 `chat_id` 和 `artifact_type`，触发 `run_agent` 调用

### 卡片交互处理

在 `lark_handlers.py` 的 `on_card_action` 中新增 action 类型 `generate_artifact`，读取 `chat_id` 和 `artifact_type`，调用 `run_agent`。

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/im_copilot/memory/eod_summary_worker.py` | 新增 | EOD 定时总结 worker |
| `src/im_copilot/memory/proactive_worker.py` | 新增 | 主动行为 worker |
| `src/im_copilot/memory/summary_worker.py` | 修改 | 新增明日预告段落生成函数 |
| `src/im_copilot/memory/group_history_worker.py` | 修改 | 新消息时调用 `proactive_worker.notify_new_messages` |
| `src/im_copilot/lark_handlers.py` | 修改 | 启动两个新 worker；新增卡片 action 处理 |
| `src/im_copilot/lark_card.py` | 修改 | 新增产物建议卡片生成函数 |

---

## 验证方式

1. **EOD Worker**：设置 `LARK_EOD_SUMMARY_ENABLED=1`，将 `LARK_EOD_SUMMARY_TIME` 设为当前时间 +2 分钟，观察群消息是否收到总结；重启后确认不重复发送。
2. **Proactive Worker**：设置 `LARK_PROACTIVE_ENABLED=1`，`LARK_PROACTIVE_SILENCE_MINUTES=1`，在测试群发几条有讨论内容的消息后等待 1 分钟，观察是否收到主动消息；发一条新消息后再等待，确认重置逻辑正常。
3. **产物建议卡片**：在测试群确认一个会议事项，触发看板查询，确认卡片出现；分别点击"生成文稿"、"生成画板"、"生成 PPT"确认 Agent 被触发并返回对应产物链接。
