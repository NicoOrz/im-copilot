from datetime import datetime, timezone

from langgraph.types import interrupt

from im_copilot.state import ApprovalGate, PipelineState


def plan_approval_node(state: PipelineState) -> dict:
    """HITL interrupt for plan approval.

    Surfaces the proposed plan and waits for human decision.
    In non-interactive mode (no checkpointer), auto-approves.
    """
    plan = state.get("plan", [])
    intent_type = state.get("intent_type", "")
    intent_params = state.get("intent_params", {})

    # Auto-approve in CLI mode to avoid blocking the terminal
    if state.get("source") == "cli":
        decision = {"approved": True, "feedback": "auto-approved (cli)"}
    else:
        try:
            decision = interrupt({
                "gate": "plan_approval",
                "plan": plan,
                "intent_type": intent_type,
                "intent_params": intent_params,
                "message": "请审阅以下执行计划，确认或提出修改意见：",
            })
        except RuntimeError:
            # No checkpointer configured (e.g., tests) — auto-approve
            decision = {"approved": True, "feedback": "auto-approved"}

    # decision: {"approved": bool, "feedback": str}
    approved = decision.get("approved", False) if isinstance(decision, dict) else False
    feedback = decision.get("feedback", "") if isinstance(decision, dict) else str(decision)

    now = datetime.now(timezone.utc).isoformat()

    if approved:
        return {
            "approvals": [ApprovalGate(
                gate_name="plan_approval",
                status="approved",
                feedback=feedback,
                timestamp=now,
            )],
        }

    return {
        "approvals": [ApprovalGate(
            gate_name="plan_approval",
            status="rejected",
            feedback=feedback,
            timestamp=now,
        )],
        "artifacts": {},
        "checks": [],
        "summary": "",
        "reflection_iteration": 0,
        "side_agent_results": [],
        "pending_questions": [],
    }
