import unittest
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import InMemorySaver

from im_copilot.graph.pipeline import build_pipeline


class MockLLM:
    """Mock LLM whose invoke method is a MagicMock, allowing dynamic return values."""

    def __init__(self):
        self.invoke = MagicMock()

    def with_structured_output(self, *args, **kwargs):
        return self


class PipelineTests(unittest.TestCase):
    def test_build_pipeline(self):
        graph = build_pipeline()
        self.assertIsNotNone(graph)

    @patch("im_copilot.graph.nodes.slide_node.run_lark_cli")
    @patch("im_copilot.graph.nodes.whiteboard_node.run_lark_cli")
    @patch("im_copilot.graph.nodes.doc_node.run_lark_cli")
    @patch("im_copilot.graph.nodes.intent_node.get_llm_for_node")
    @patch("im_copilot.graph.nodes.doc_node.get_llm_for_node")
    @patch("im_copilot.graph.nodes.whiteboard_node.get_llm_for_node")
    @patch("im_copilot.graph.nodes.slide_node.get_llm_for_node")
    @patch("im_copilot.graph.nodes.deliver_node.get_llm_for_node")
    def test_multi_input_invokes_doc_whiteboard_slide(
        self,
        mock_deliver,
        mock_slide,
        mock_wb,
        mock_doc,
        mock_intent,
        mock_doc_cli,
        mock_wb_cli,
        mock_slide_cli,
    ):
        mock_intent.return_value = MockLLM()
        mock_intent.return_value.invoke.return_value = MagicMock(
            intent_type="create_multi",
            topic="报告",
            confidence=0.9,
            needs_clarification=False,
            questions=[],
            plan=["doc", "whiteboard", "slide", "deliver"],
        )
        mock_doc.return_value = MockLLM()
        mock_doc.return_value.invoke.return_value = MagicMock(content="doc内容")
        mock_wb.return_value = MockLLM()
        mock_wb.return_value.invoke.return_value = MagicMock(content="wb内容")
        mock_slide.return_value = MockLLM()
        mock_slide.return_value.invoke.return_value = MagicMock(content="slide内容")
        mock_deliver.return_value = MockLLM()
        mock_deliver.return_value.invoke.return_value = MagicMock(content="汇总结果")
        mock_doc_cli.return_value = {"data": {"document": {"document_id": "doc-token"}}}
        mock_wb_cli.side_effect = [
            {"data": {"obj_token": "docx-token"}},
            {},
            {"data": {"document": {"content": '<whiteboard token="wb-token"></whiteboard>'}}},
            {},
        ]
        mock_slide_cli.return_value = {"data": {"presentation": {"presentation_token": "slide-token"}}}

        checkpointer = InMemorySaver()
        graph = build_pipeline(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-multi"}}
        initial_state = {
            "raw_message": "帮我写一份报告，画流程图，并生成 PPT",
            "chat_id": "cli",
            "message_id": "cli",
            "source": "cli",
            "errors": [],
            "checks": [],
            "reflection_iteration": 0,
        }

        result = graph.invoke(initial_state, config=config)
        self.assertEqual(result["intent_type"], "create_multi")
        self.assertEqual(result["plan"], ["doc", "whiteboard", "slide", "deliver"])
        self.assertIn("doc", result["artifacts"])
        self.assertIn("whiteboard", result["artifacts"])
        self.assertIn("slide", result["artifacts"])
        mock_deliver.assert_not_called()
        self.assertIn("文档：报告：https://www.feishu.cn/docx/doc-token", result["summary"])
        self.assertIn("白板：报告：https://www.feishu.cn/docx/docx-token", result["summary"])
        self.assertIn("PPT：报告：https://www.feishu.cn/slides/slide-token", result["summary"])

    @patch("im_copilot.graph.nodes.whiteboard_node.run_lark_cli")
    @patch("im_copilot.graph.nodes.intent_node.get_llm_for_node")
    @patch("im_copilot.graph.nodes.whiteboard_node.get_llm_for_node")
    @patch("im_copilot.graph.nodes.deliver_node.get_llm_for_node")
    def test_whiteboard_only_input(self, mock_deliver, mock_wb, mock_intent, mock_wb_cli):
        mock_intent.return_value = MockLLM()
        mock_intent.return_value.invoke.return_value = MagicMock(
            intent_type="create_whiteboard",
            topic="流程图",
            confidence=0.9,
            needs_clarification=False,
            questions=[],
            plan=["whiteboard", "deliver"],
        )
        mock_wb.return_value = MockLLM()
        mock_wb.return_value.invoke.return_value = MagicMock(content="wb内容")
        mock_deliver.return_value = MockLLM()
        mock_deliver.return_value.invoke.return_value = MagicMock(content="汇总结果")
        mock_wb_cli.side_effect = [
            {"data": {"obj_token": "docx-token"}},
            {},
            {"data": {"document": {"content": '<whiteboard token="wb-token"></whiteboard>'}}},
            {},
        ]

        checkpointer = InMemorySaver()
        graph = build_pipeline(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-wb"}}
        initial_state = {
            "raw_message": "帮我画一个项目流程图",
            "chat_id": "cli",
            "message_id": "cli",
            "source": "cli",
            "errors": [],
            "checks": [],
            "reflection_iteration": 0,
        }

        result = graph.invoke(initial_state, config=config)

        self.assertEqual(result["intent_type"], "create_whiteboard")
        self.assertEqual(result["plan"], ["whiteboard", "deliver"])
        self.assertEqual(set(result["artifacts"].keys()), {"whiteboard"})
        mock_deliver.assert_not_called()

    @patch("im_copilot.graph.nodes.intent_node.get_llm_for_node")
    @patch("im_copilot.graph.nodes.deliver_node.get_llm_for_node")
    def test_chat_input(self, mock_deliver, mock_intent):
        mock_intent.return_value = MockLLM()
        mock_intent.return_value.invoke.return_value = MagicMock(
            intent_type="chat",
            topic="你好",
            confidence=0.9,
            needs_clarification=False,
            questions=[],
            plan=["deliver"],
        )
        mock_deliver.return_value = MockLLM()
        mock_deliver.return_value.invoke.return_value = MagicMock(content="你好！有什么可以帮你的？")

        checkpointer = InMemorySaver()
        graph = build_pipeline(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-chat"}}
        initial_state = {
            "raw_message": "你好",
            "chat_id": "cli",
            "message_id": "cli",
            "source": "cli",
            "errors": [],
            "checks": [],
            "reflection_iteration": 0,
        }

        result = graph.invoke(initial_state, config=config)

        self.assertEqual(result["intent_type"], "chat")
        self.assertEqual(result["plan"], ["deliver"])
        self.assertNotIn("artifacts", result)
        self.assertEqual(result["summary"], "你好！有什么可以帮你的？")


if __name__ == "__main__":
    unittest.main()
