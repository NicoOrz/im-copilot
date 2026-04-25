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


class ClarificationTurn(TypedDict):
    question: str
    answer: str


class ApprovalGate(TypedDict):
    gate_name: str
    status: Literal["pending", "approved", "rejected", "skipped"]
    feedback: str
    timestamp: str


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

    # Phase 2 fields
    clarification_history: Annotated[list[ClarificationTurn], add]
    approvals: Annotated[list[ApprovalGate], add]
    reflection_iteration: int
    side_agent_results: Annotated[list[dict], add]
    thread_id: str
    user_id: str
    pending_questions: list[str]
