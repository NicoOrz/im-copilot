# Phase 1 LangGraph Workflow Design

## Goal

Build the minimum Agent-Pilot workflow foundation:

```text
IM text input -> intent -> planner -> mock business nodes -> deliver
```

Phase 1 proves orchestration only. It does not call Feishu CLI, Feishu APIs, LLMs, voice services, or multi-device sync.

## Scope

Included:

- Minimal Python package under `src/im_copilot`.
- LangGraph `StateGraph` workflow.
- Rule-based intent detection.
- Planner mapping intent to ordered task steps.
- Mock doc, whiteboard, and slide nodes.
- CLI entry point for local verification.
- TDD for nodes and graph behavior.
- `checks` state field reserved for a future verifier side agent.

Excluded:

- `feishu_cli.py` adapter.
- Real document, whiteboard, or PPT generation.
- LLM structured output.
- Feishu webhook integration.
- Human approval.
- Persistence.
- Multi-device sync.

## Architecture

Files:

```text
pyproject.toml
src/im_copilot/__init__.py
src/im_copilot/main.py
src/im_copilot/state.py
src/im_copilot/graph/__init__.py
src/im_copilot/graph/pipeline.py
src/im_copilot/graph/nodes/__init__.py
src/im_copilot/graph/nodes/intent_node.py
src/im_copilot/graph/nodes/planner_node.py
src/im_copilot/graph/nodes/doc_node.py
src/im_copilot/graph/nodes/whiteboard_node.py
src/im_copilot/graph/nodes/slide_node.py
src/im_copilot/graph/nodes/deliver_node.py
tests/test_nodes.py
tests/test_pipeline.py
```

Dependencies:

```toml
langgraph = ">=1.0,<2.0"
langchain-core = ">=1.0,<2.0"
```

Use `unittest` from the Python standard library. Do not add pytest.

## State

`source` only supports `cli` and `feishu`. Client technology names are not part of workflow state.

```python
from operator import add
from typing import Annotated, Literal, TypedDict


IntentType = Literal[
    "create_doc",
    "create_whiteboard",
    "create_slide",
    "create_multi",
    "chat",
]

PlanStep = Literal["doc", "whiteboard", "slide", "deliver"]


class MockResult(TypedDict):
    kind: str
    title: str
    status: str
    preview: str


class CheckResult(TypedDict):
    task: str
    status: Literal["pass", "revise", "clarify"]
    reason: str


class PipelineState(TypedDict, total=False):
    raw_message: str
    chat_id: str
    message_id: str
    source: Literal["feishu", "cli"]

    intent_type: IntentType
    intent_params: dict[str, str]

    plan: list[PlanStep]
    mock_results: dict[str, MockResult]
    checks: Annotated[list[CheckResult], add]
    summary: str

    errors: Annotated[list[str], add]
```

`checks` is not used in Phase 1 routing. It exists so a future `verifier_node` can record whether a task passed, needs revision, or needs user clarification.

## Nodes

### intent_node

Rules:

- doc keywords: `文档`, `报告`, `纪要`, `方案`
- whiteboard keywords: `白板`, `流程图`, `思维导图`
- slide keywords: `PPT`, `ppt`, `幻灯片`, `演示稿`
- if at least two categories match: `create_multi`
- if none match: `chat`

Return:

```python
{
    "intent_type": "...",
    "intent_params": {"topic": raw_message},
}
```

### planner_node

Mapping:

| intent_type | plan |
| --- | --- |
| `create_doc` | `["doc", "deliver"]` |
| `create_whiteboard` | `["whiteboard", "deliver"]` |
| `create_slide` | `["slide", "deliver"]` |
| `create_multi` | `["doc", "whiteboard", "slide", "deliver"]` |
| `chat` | `["deliver"]` |

### mock business nodes

`doc_node`, `whiteboard_node`, and `slide_node` each read existing `mock_results` and return a new dict containing their own result.

Each result has:

```python
{
    "kind": "...",
    "title": "...",
    "status": "created",
    "preview": "...",
}
```

### deliver_node

Rules:

- If `errors` exists, include error information in `summary`.
- If `intent_type == "chat"`, return a short general response.
- Otherwise, summarize mock results in plan order.

## Graph

Use `StateGraph(PipelineState)`.

Static edges:

```text
START -> intent
intent -> planner
slide -> deliver
deliver -> END
```

Conditional edges:

```text
planner -> doc | whiteboard | slide | deliver
doc -> whiteboard | slide | deliver
whiteboard -> slide | deliver
```

Use `add_conditional_edges()`. Do not use `Command` in Phase 1.

## CLI

`python -m im_copilot.main "<message>"` should:

- create the initial state with `source="cli"`;
- invoke the graph;
- print `summary`;
- print usage text when no message is provided.

## Tests

Use TDD for:

- intent classification;
- planner mapping;
- mock node output structure;
- deliver summaries;
- `build_pipeline()`;
- graph invocation for multi-task, whiteboard-only, and chat inputs.

## Verification

```bash
uv sync
uv run python -m unittest
uv run python -c "from im_copilot.graph.pipeline import build_pipeline; build_pipeline(); print('Graph built OK')"
uv run python -m im_copilot.main "帮我写一份 Q2 季度报告并生成 PPT"
```

## Future Extension

Final product may add:

```text
IM input
-> intent
-> planner
-> doc / whiteboard / slide
-> verifier
-> revise / clarify / next task
-> deliver
```

The future verifier should only inspect task output and decide `pass`, `revise`, or `clarify`. It should not generate document, whiteboard, or PPT content.
