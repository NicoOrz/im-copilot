import unittest
from unittest.mock import MagicMock, patch

from im_copilot.graph.pipeline import build_pipeline, run_pipeline


class PipelineTests(unittest.TestCase):
    def test_build_pipeline(self):
        graph = build_pipeline()
        self.assertIsNotNone(graph)

    @patch("im_copilot.graph.nodes.intent_node._llm")
    @patch("im_copilot.graph.nodes.planner_node._llm")
    @patch("im_copilot.graph.nodes.doc_node._llm")
    @patch("im_copilot.graph.nodes.whiteboard_node._llm")
    @patch("im_copilot.graph.nodes.slide_node._llm")
    @patch("im_copilot.graph.nodes.deliver_node._llm")
    def test_multi_input_invokes_doc_whiteboard_slide(self, mock_deliver, mock_slide, mock_wb, mock_doc, mock_planner, mock_intent):
        mock_intent.invoke.return_value = MagicMock(intent_type="create_multi", topic="报告")
        mock_planner.invoke.return_value = MagicMock(plan=["doc", "whiteboard", "slide", "deliver"])
        mock_doc.invoke.return_value = MagicMock(content="doc内容")
        mock_wb.invoke.return_value = MagicMock(content="wb内容")
        mock_slide.invoke.return_value = MagicMock(content="slide内容")
        mock_deliver.invoke.return_value = MagicMock(content="汇总结果")

        result = run_pipeline("帮我写一份报告，画流程图，并生成 PPT")

        self.assertEqual(result["intent_type"], "create_multi")
        self.assertEqual(result["plan"], ["doc", "whiteboard", "slide", "deliver"])
        self.assertIn("doc", result["mock_results"])
        self.assertIn("whiteboard", result["mock_results"])
        self.assertIn("slide", result["mock_results"])
        self.assertEqual(result["summary"], "汇总结果")

    @patch("im_copilot.graph.nodes.intent_node._llm")
    @patch("im_copilot.graph.nodes.planner_node._llm")
    @patch("im_copilot.graph.nodes.whiteboard_node._llm")
    @patch("im_copilot.graph.nodes.deliver_node._llm")
    def test_whiteboard_only_input(self, mock_deliver, mock_wb, mock_planner, mock_intent):
        mock_intent.invoke.return_value = MagicMock(intent_type="create_whiteboard", topic="流程图")
        mock_planner.invoke.return_value = MagicMock(plan=["whiteboard", "deliver"])
        mock_wb.invoke.return_value = MagicMock(content="wb内容")
        mock_deliver.invoke.return_value = MagicMock(content="汇总结果")

        result = run_pipeline("帮我画一个项目流程图")

        self.assertEqual(result["intent_type"], "create_whiteboard")
        self.assertEqual(result["plan"], ["whiteboard", "deliver"])
        self.assertEqual(set(result["mock_results"].keys()), {"whiteboard"})

    @patch("im_copilot.graph.nodes.intent_node._llm")
    @patch("im_copilot.graph.nodes.planner_node._llm")
    @patch("im_copilot.graph.nodes.deliver_node._llm")
    def test_chat_input(self, mock_deliver, mock_planner, mock_intent):
        mock_intent.invoke.return_value = MagicMock(intent_type="chat", topic="你好")
        mock_planner.invoke.return_value = MagicMock(plan=["deliver"])
        mock_deliver.invoke.return_value = MagicMock(content="你好！有什么可以帮你的？")

        result = run_pipeline("你好")

        self.assertEqual(result["intent_type"], "chat")
        self.assertEqual(result["plan"], ["deliver"])
        self.assertNotIn("mock_results", result)
        self.assertEqual(result["summary"], "你好！有什么可以帮你的？")


if __name__ == "__main__":
    unittest.main()
