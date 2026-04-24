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
