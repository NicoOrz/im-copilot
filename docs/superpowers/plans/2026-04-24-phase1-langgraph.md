# Phase 1 LangGraph Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum Agent-Pilot LangGraph workflow with rule intent detection, planning, mock business nodes, delivery summary, CLI entry, and node plus graph TDD.

**Architecture:** Use a small `src/im_copilot` Python package. Each LangGraph node lives in one focused module and returns partial state updates. `pipeline.py` owns graph wiring and routing; `main.py` owns the local CLI entry.

**Tech Stack:** Python 3.12, `langgraph`, `langchain-core`, standard-library `unittest`, `uv`.

---

## File Structure

- Create: `pyproject.toml`
  - Minimal project metadata and dependencies.
- Create: `src/im_copilot/__init__.py`
  - Package marker.
- Create: `src/im_copilot/state.py`
  - `PipelineState`, `IntentType`, `PlanStep`, `MockResult`, `CheckResult`.
- Create: `src/im_copilot/graph/__init__.py`
  - Graph package marker.
- Create: `src/im_copilot/graph/nodes/__init__.py`
  - Node package marker.
- Create: `src/im_copilot/graph/nodes/intent_node.py`
  - Rule-based intent classification.
- Create: `src/im_copilot/graph/nodes/planner_node.py`
  - Intent-to-plan mapping.
- Create: `src/im_copilot/graph/nodes/doc_node.py`
  - Mock document result.
- Create: `src/im_copilot/graph/nodes/whiteboard_node.py`
  - Mock whiteboard result.
- Create: `src/im_copilot/graph/nodes/slide_node.py`
  - Mock slide result.
- Create: `src/im_copilot/graph/nodes/deliver_node.py`
  - Summary generation.
- Create: `src/im_copilot/graph/pipeline.py`
  - `build_pipeline()`, routing functions, `run_pipeline()`.
- Create: `src/im_copilot/main.py`
  - CLI entry.
- Create: `tests/test_nodes.py`
  - Node TDD.
- Create: `tests/test_pipeline.py`
  - Graph TDD.

Do not create `src/im_copilot/integrations/feishu_cli.py` in Phase 1.

## Task 1: Project Skeleton and State

**Files:**
- Create: `pyproject.toml`
- Create: `src/im_copilot/__init__.py`
- Create: `src/im_copilot/state.py`

- [ ] **Step 1: Write initial state import test**

Create `tests/test_nodes.py` with:

```python
import unittest

from im_copilot.state import PipelineState


class StateTests(unittest.TestCase):
    def test_pipeline_state_type_imports(self):
        state: PipelineState = {
            "raw_message": "hello",
            "chat_id": "cli",
            "message_id": "cli",
            "source": "cli",
            "errors": [],
            "checks": [],
        }

        self.assertEqual(state["source"], "cli")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
uv run python -m unittest tests.test_nodes
```

Expected: fail because `im_copilot` does not exist.

- [ ] **Step 3: Add project metadata**

Create `pyproject.toml`:

```toml
[project]
name = "im-copilot"
version = "0.1.0"
description = "Agent-Pilot Phase 1 LangGraph workflow"
requires-python = ">=3.10"
dependencies = [
    "langgraph>=1.0,<2.0",
    "langchain-core>=1.0,<2.0",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 4: Add package and state module**

Create `src/im_copilot/__init__.py`:

```python
"""Agent-Pilot Phase 1 package."""
```

Create `src/im_copilot/state.py`:

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

- [ ] **Step 5: Verify the test passes**

Run:

```bash
uv sync
uv run python -m unittest tests.test_nodes
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/im_copilot/__init__.py src/im_copilot/state.py tests/test_nodes.py
git commit -m "Add Phase 1 project state"
```

## Task 2: Node Tests and Implementations

**Files:**
- Modify: `tests/test_nodes.py`
- Create: `src/im_copilot/graph/__init__.py`
- Create: `src/im_copilot/graph/nodes/__init__.py`
- Create: `src/im_copilot/graph/nodes/intent_node.py`
- Create: `src/im_copilot/graph/nodes/planner_node.py`
- Create: `src/im_copilot/graph/nodes/doc_node.py`
- Create: `src/im_copilot/graph/nodes/whiteboard_node.py`
- Create: `src/im_copilot/graph/nodes/slide_node.py`
- Create: `src/im_copilot/graph/nodes/deliver_node.py`

- [ ] **Step 1: Add failing node tests**

Replace `tests/test_nodes.py` with:

```python
import unittest

from im_copilot.graph.nodes.deliver_node import deliver_node
from im_copilot.graph.nodes.doc_node import doc_node
from im_copilot.graph.nodes.intent_node import intent_node
from im_copilot.graph.nodes.planner_node import planner_node
from im_copilot.graph.nodes.slide_node import slide_node
from im_copilot.graph.nodes.whiteboard_node import whiteboard_node


class IntentNodeTests(unittest.TestCase):
    def test_classifies_doc_intent(self):
        result = intent_node({"raw_message": "帮我写一份产品方案"})
        self.assertEqual(result["intent_type"], "create_doc")
        self.assertEqual(result["intent_params"]["topic"], "帮我写一份产品方案")

    def test_classifies_whiteboard_intent(self):
        result = intent_node({"raw_message": "帮我画一个项目流程图"})
        self.assertEqual(result["intent_type"], "create_whiteboard")

    def test_classifies_slide_intent(self):
        result = intent_node({"raw_message": "帮我生成 PPT"})
        self.assertEqual(result["intent_type"], "create_slide")

    def test_classifies_multi_intent(self):
        result = intent_node({"raw_message": "帮我写报告并生成 PPT"})
        self.assertEqual(result["intent_type"], "create_multi")

    def test_classifies_chat_intent(self):
        result = intent_node({"raw_message": "你好"})
        self.assertEqual(result["intent_type"], "chat")


class PlannerNodeTests(unittest.TestCase):
    def test_maps_multi_intent_to_all_business_steps(self):
        result = planner_node({"intent_type": "create_multi"})
        self.assertEqual(result["plan"], ["doc", "whiteboard", "slide", "deliver"])

    def test_maps_chat_to_deliver_only(self):
        result = planner_node({"intent_type": "chat"})
        self.assertEqual(result["plan"], ["deliver"])


class MockNodeTests(unittest.TestCase):
    def test_doc_node_adds_doc_result(self):
        result = doc_node({"mock_results": {}})
        self.assertEqual(result["mock_results"]["doc"]["kind"], "doc")
        self.assertEqual(result["mock_results"]["doc"]["status"], "created")

    def test_whiteboard_node_preserves_existing_results(self):
        result = whiteboard_node({"mock_results": {"doc": {"kind": "doc", "title": "x", "status": "created", "preview": "x"}}})
        self.assertIn("doc", result["mock_results"])
        self.assertEqual(result["mock_results"]["whiteboard"]["kind"], "whiteboard")

    def test_slide_node_adds_slide_result(self):
        result = slide_node({"mock_results": {}})
        self.assertEqual(result["mock_results"]["slide"]["kind"], "slide")


class DeliverNodeTests(unittest.TestCase):
    def test_deliver_chat_summary(self):
        result = deliver_node({"intent_type": "chat", "plan": ["deliver"], "errors": []})
        self.assertIn("收到", result["summary"])

    def test_deliver_mock_results_in_plan_order(self):
        result = deliver_node(
            {
                "intent_type": "create_multi",
                "plan": ["doc", "slide", "deliver"],
                "mock_results": {
                    "doc": {"kind": "doc", "title": "Mock doc", "status": "created", "preview": "doc done"},
                    "slide": {"kind": "slide", "title": "Mock slide", "status": "created", "preview": "slide done"},
                },
                "errors": [],
            }
        )

        self.assertLess(result["summary"].index("doc done"), result["summary"].index("slide done"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify node tests fail**

Run:

```bash
uv run python -m unittest tests.test_nodes
```

Expected: fail because node modules do not exist.

- [ ] **Step 3: Add graph package markers**

Create `src/im_copilot/graph/__init__.py`:

```python
"""LangGraph workflow package."""
```

Create `src/im_copilot/graph/nodes/__init__.py`:

```python
"""LangGraph node package."""
```

- [ ] **Step 4: Implement intent node**

Create `src/im_copilot/graph/nodes/intent_node.py`:

```python
from im_copilot.state import PipelineState


DOC_KEYWORDS = ("文档", "报告", "纪要", "方案")
WHITEBOARD_KEYWORDS = ("白板", "流程图", "思维导图")
SLIDE_KEYWORDS = ("PPT", "ppt", "幻灯片", "演示稿")


def _contains_any(message: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in message for keyword in keywords)


def intent_node(state: PipelineState) -> dict:
    raw_message = state.get("raw_message", "")
    matches = {
        "create_doc": _contains_any(raw_message, DOC_KEYWORDS),
        "create_whiteboard": _contains_any(raw_message, WHITEBOARD_KEYWORDS),
        "create_slide": _contains_any(raw_message, SLIDE_KEYWORDS),
    }

    matched_intents = [intent for intent, matched in matches.items() if matched]
    if len(matched_intents) >= 2:
        intent_type = "create_multi"
    elif matched_intents:
        intent_type = matched_intents[0]
    else:
        intent_type = "chat"

    return {
        "intent_type": intent_type,
        "intent_params": {"topic": raw_message},
    }
```

- [ ] **Step 5: Implement planner node**

Create `src/im_copilot/graph/nodes/planner_node.py`:

```python
from im_copilot.state import PipelineState


PLAN_BY_INTENT = {
    "create_doc": ["doc", "deliver"],
    "create_whiteboard": ["whiteboard", "deliver"],
    "create_slide": ["slide", "deliver"],
    "create_multi": ["doc", "whiteboard", "slide", "deliver"],
    "chat": ["deliver"],
}


def planner_node(state: PipelineState) -> dict:
    intent_type = state.get("intent_type", "chat")
    return {"plan": PLAN_BY_INTENT.get(intent_type, ["deliver"])}
```

- [ ] **Step 6: Implement mock business nodes**

Create `src/im_copilot/graph/nodes/doc_node.py`:

```python
from im_copilot.state import PipelineState


def doc_node(state: PipelineState) -> dict:
    return {
        "mock_results": {
            **state.get("mock_results", {}),
            "doc": {
                "kind": "doc",
                "title": "Mock 文档",
                "status": "created",
                "preview": "已生成文档草稿占位结果",
            },
        }
    }
```

Create `src/im_copilot/graph/nodes/whiteboard_node.py`:

```python
from im_copilot.state import PipelineState


def whiteboard_node(state: PipelineState) -> dict:
    return {
        "mock_results": {
            **state.get("mock_results", {}),
            "whiteboard": {
                "kind": "whiteboard",
                "title": "Mock 白板",
                "status": "created",
                "preview": "已生成白板占位结果",
            },
        }
    }
```

Create `src/im_copilot/graph/nodes/slide_node.py`:

```python
from im_copilot.state import PipelineState


def slide_node(state: PipelineState) -> dict:
    return {
        "mock_results": {
            **state.get("mock_results", {}),
            "slide": {
                "kind": "slide",
                "title": "Mock PPT",
                "status": "created",
                "preview": "已生成演示稿占位结果",
            },
        }
    }
```

- [ ] **Step 7: Implement deliver node**

Create `src/im_copilot/graph/nodes/deliver_node.py`:

```python
from im_copilot.state import PipelineState


def deliver_node(state: PipelineState) -> dict:
    errors = state.get("errors", [])
    if errors:
        return {"summary": "任务执行出现错误：" + "；".join(errors)}

    if state.get("intent_type") == "chat":
        return {"summary": "收到。Phase 1 当前支持文档、白板、PPT mock 工作流。"}

    plan = state.get("plan", [])
    mock_results = state.get("mock_results", {})
    lines = ["Phase 1 mock 结果："]
    for step in plan:
        if step == "deliver":
            continue
        result = mock_results.get(step)
        if result:
            lines.append(f"- {result['title']}：{result['preview']}")

    return {"summary": "\n".join(lines)}
```

- [ ] **Step 8: Verify node tests pass**

Run:

```bash
uv run python -m unittest tests.test_nodes
```

Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add src/im_copilot/graph tests/test_nodes.py
git commit -m "Add Phase 1 workflow nodes"
```

## Task 3: Graph Pipeline TDD

**Files:**
- Create: `tests/test_pipeline.py`
- Create: `src/im_copilot/graph/pipeline.py`

- [ ] **Step 1: Add failing graph tests**

Create `tests/test_pipeline.py`:

```python
import unittest

from im_copilot.graph.pipeline import build_pipeline, run_pipeline


class PipelineTests(unittest.TestCase):
    def test_build_pipeline(self):
        graph = build_pipeline()
        self.assertIsNotNone(graph)

    def test_multi_input_invokes_doc_whiteboard_slide(self):
        result = run_pipeline("帮我写一份报告，画流程图，并生成 PPT")

        self.assertEqual(result["intent_type"], "create_multi")
        self.assertEqual(result["plan"], ["doc", "whiteboard", "slide", "deliver"])
        self.assertIn("doc", result["mock_results"])
        self.assertIn("whiteboard", result["mock_results"])
        self.assertIn("slide", result["mock_results"])
        self.assertIn("Phase 1 mock 结果", result["summary"])

    def test_whiteboard_only_input(self):
        result = run_pipeline("帮我画一个项目流程图")

        self.assertEqual(result["intent_type"], "create_whiteboard")
        self.assertEqual(result["plan"], ["whiteboard", "deliver"])
        self.assertEqual(set(result["mock_results"].keys()), {"whiteboard"})

    def test_chat_input(self):
        result = run_pipeline("你好")

        self.assertEqual(result["intent_type"], "chat")
        self.assertEqual(result["plan"], ["deliver"])
        self.assertNotIn("mock_results", result)
        self.assertIn("Phase 1", result["summary"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify graph tests fail**

Run:

```bash
uv run python -m unittest tests.test_pipeline
```

Expected: fail because `im_copilot.graph.pipeline` does not exist.

- [ ] **Step 3: Implement pipeline**

Create `src/im_copilot/graph/pipeline.py`:

```python
from langgraph.graph import END, START, StateGraph

from im_copilot.graph.nodes.deliver_node import deliver_node
from im_copilot.graph.nodes.doc_node import doc_node
from im_copilot.graph.nodes.intent_node import intent_node
from im_copilot.graph.nodes.planner_node import planner_node
from im_copilot.graph.nodes.slide_node import slide_node
from im_copilot.graph.nodes.whiteboard_node import whiteboard_node
from im_copilot.state import PipelineState


def route_after_planner(state: PipelineState) -> str:
    plan = state.get("plan", [])
    if "doc" in plan:
        return "doc"
    if "whiteboard" in plan:
        return "whiteboard"
    if "slide" in plan:
        return "slide"
    return "deliver"


def route_after_doc(state: PipelineState) -> str:
    plan = state.get("plan", [])
    if "whiteboard" in plan:
        return "whiteboard"
    if "slide" in plan:
        return "slide"
    return "deliver"


def route_after_whiteboard(state: PipelineState) -> str:
    if "slide" in state.get("plan", []):
        return "slide"
    return "deliver"


def build_pipeline():
    builder = StateGraph(PipelineState)
    builder.add_node("intent", intent_node)
    builder.add_node("planner", planner_node)
    builder.add_node("doc", doc_node)
    builder.add_node("whiteboard", whiteboard_node)
    builder.add_node("slide", slide_node)
    builder.add_node("deliver", deliver_node)

    builder.add_edge(START, "intent")
    builder.add_edge("intent", "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        ["doc", "whiteboard", "slide", "deliver"],
    )
    builder.add_conditional_edges(
        "doc",
        route_after_doc,
        ["whiteboard", "slide", "deliver"],
    )
    builder.add_conditional_edges(
        "whiteboard",
        route_after_whiteboard,
        ["slide", "deliver"],
    )
    builder.add_edge("slide", "deliver")
    builder.add_edge("deliver", END)
    return builder.compile()


def run_pipeline(
    message: str,
    *,
    chat_id: str = "cli",
    message_id: str = "cli",
    source: str = "cli",
) -> PipelineState:
    graph = build_pipeline()
    initial_state: PipelineState = {
        "raw_message": message,
        "chat_id": chat_id,
        "message_id": message_id,
        "source": source,
        "errors": [],
        "checks": [],
    }
    return graph.invoke(initial_state)
```

- [ ] **Step 4: Verify graph tests pass**

Run:

```bash
uv run python -m unittest tests.test_pipeline
```

Expected: pass.

- [ ] **Step 5: Verify full tests pass**

Run:

```bash
uv run python -m unittest
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/im_copilot/graph/pipeline.py tests/test_pipeline.py
git commit -m "Add Phase 1 LangGraph pipeline"
```

## Task 4: CLI Entry

**Files:**
- Create: `src/im_copilot/main.py`

- [ ] **Step 1: Add CLI module**

Create `src/im_copilot/main.py`:

```python
import sys

from im_copilot.graph.pipeline import run_pipeline


USAGE = 'Usage: python -m im_copilot.main "<message>"'


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(USAGE)
        return 1

    message = " ".join(args)
    result = run_pipeline(message)
    print(result.get("summary", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify CLI success path**

Run:

```bash
uv run python -m im_copilot.main "帮我写一份 Q2 季度报告并生成 PPT"
```

Expected: output contains `Phase 1 mock 结果`, `Mock 文档`, and `Mock PPT`.

- [ ] **Step 3: Verify CLI usage path**

Run:

```bash
uv run python -m im_copilot.main
```

Expected: output contains `Usage:` and process exits with code `1`.

- [ ] **Step 4: Verify import command**

Run:

```bash
uv run python -c "from im_copilot.graph.pipeline import build_pipeline; build_pipeline(); print('Graph built OK')"
```

Expected: output is `Graph built OK`.

- [ ] **Step 5: Verify full tests pass**

Run:

```bash
uv run python -m unittest
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/im_copilot/main.py
git commit -m "Add Phase 1 CLI entry"
```

## Task 5: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Verify no Feishu adapter exists**

Run:

```bash
test ! -e src/im_copilot/integrations/feishu_cli.py
```

Expected: exit code `0`.

- [ ] **Step 2: Verify test suite**

Run:

```bash
uv run python -m unittest
```

Expected: all tests pass.

- [ ] **Step 3: Verify graph import**

Run:

```bash
uv run python -c "from im_copilot.graph.pipeline import build_pipeline; build_pipeline(); print('Graph built OK')"
```

Expected: output is `Graph built OK`.

- [ ] **Step 4: Verify sample commands**

Run:

```bash
uv run python -m im_copilot.main "帮我写一份 Q2 季度报告并生成 PPT"
uv run python -m im_copilot.main "帮我画一个项目流程图"
uv run python -m im_copilot.main "你好"
```

Expected:

- first output contains document and PPT mock content;
- second output contains whiteboard mock content;
- third output contains the Phase 1 chat response.

- [ ] **Step 5: Inspect changed files**

Run:

```bash
git status --short
```

Expected: only intentional Phase 1 files are changed or untracked.

- [ ] **Step 6: Commit any verification-only plan/doc updates if needed**

Only commit if an implementation step had to update this plan or the design spec.
