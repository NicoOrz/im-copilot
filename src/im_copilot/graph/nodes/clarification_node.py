from langgraph.types import interrupt

from im_copilot.state import ClarificationTurn, PipelineState


def clarification_node(state: PipelineState) -> dict:
    """HITL interrupt to ask clarifying questions.

    Surfaces pending questions to the user and waits for answers.
    Appends Q&A to clarification_history.
    """
    questions = state.get("pending_questions", [])
    if not questions:
        return {"pending_questions": []}

    try:
        answers = interrupt({
            "gate": "clarification",
            "questions": questions,
            "message": "为了更准确地制定计划，请回答以下问题：",
        })
    except RuntimeError:
        # No checkpointer configured (e.g., tests) — provide empty answers
        answers = [""] * len(questions)

    # answers should be a list of strings matching questions length
    if isinstance(answers, str):
        answers = [answers]
    elif not isinstance(answers, list):
        answers = list(answers) if answers else [""] * len(questions)

    # Pad answers if needed
    while len(answers) < len(questions):
        answers.append("")

    history = [
        ClarificationTurn(question=q, answer=a)
        for q, a in zip(questions, answers)
    ]

    return {
        "clarification_history": history,
        "pending_questions": [],
    }
