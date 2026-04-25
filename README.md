# IM Copilot

基于 LangGraph 的智能办公助手，支持多轮对话、人机协作（HITL）和会话管理。

## 功能特性

- **意图识别**：自动识别用户意图（创建文档/白板/PPT、聊天、多任务）
- **任务规划**：LLM 驱动的执行计划生成，支持澄清和审批
- **内容生成**：文档、白板、PPT 的 AI 生成
- **质量验证**：双重验证机制（主验证 + 并行 Side Agent）
- **人机协作**：计划审批和澄清问题的中断-恢复机制
- **会话管理**：基于 SQLite 的持久化，支持多会话切换
- **Web UI**：FastAPI + Jinja2 的交互式网页界面
- **CLI**：命令行入口，支持中断恢复

## 架构

```
用户输入 → 意图识别 → 任务规划 → [澄清/审批] → 内容生成 → 验证 → 交付
                              ↑___________↓
```

核心节点：
- `intent`：意图分类
- `planner`：任务规划（含澄清分支）
- `clarification`：HITL 澄清问题
- `plan_approval`：HITL 计划审批
- `doc/whiteboard/slide`：内容生成
- `verify`：质量验证
- `side_agent`：并行验证
- `deliver`：总结交付

## 快速开始

### 环境准备

```bash
# 安装依赖
uv sync

# 配置 LLM（火山引擎）
cp .env.example .env
# 编辑 .env 填入 VOLC_API_KEY、VOLC_BASE_URL、VOLC_MODEL
```

### Web UI

```bash
uv run python -m im_copilot.main --web
# 打开 http://localhost:8000
```

### CLI

```bash
# 直接运行
uv run python -m im_copilot.main "帮我写一份产品方案"

# 指定会话 ID
uv run python -m im_copilot.main --thread-id my-session "帮我生成PPT"

# 从中断恢复
uv run python -m im_copilot.main --thread-id my-session --resume '{"approved":true}'
```

### 测试

```bash
uv run python -m pytest tests/ -v
```

## 项目结构

```
src/im_copilot/
├── main.py                 # CLI 入口
├── state.py                # PipelineState 定义
├── llm.py                  # LLM 客户端配置
├── checkpointer.py         # SQLite/Memory 持久化
├── graph/
│   ├── pipeline.py         # LangGraph 组装
│   └── nodes/              # 各节点实现
│       ├── intent_node.py
│       ├── planner_node.py
│       ├── clarification_node.py
│       ├── plan_approval_node.py
│       ├── doc_node.py
│       ├── whiteboard_node.py
│       ├── slide_node.py
│       ├── verify_node.py
│       ├── side_agent_node.py
│       └── deliver_node.py
└── web/                    # Web UI
    ├── app.py              # FastAPI 应用
    ├── static/             # CSS/JS
    └── templates/          # Jinja2 模板
```

## 技术栈

- **LangGraph**：状态图编排、HITL 中断、并行执行
- **LangChain OpenAI**：兼容 OpenAI API 的 LLM 调用
- **FastAPI + Jinja2**：Web 服务端渲染
- **SQLite**：会话持久化
- **Volcengine**：LLM 推理

## 许可证

MIT
