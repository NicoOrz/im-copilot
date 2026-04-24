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
