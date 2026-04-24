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
