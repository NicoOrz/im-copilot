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
