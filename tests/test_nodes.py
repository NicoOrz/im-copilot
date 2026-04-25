import unittest
from unittest.mock import MagicMock, patch

from im_copilot.graph.nodes.deliver_node import deliver_node
from im_copilot.graph.nodes.doc_node import doc_node
from im_copilot.graph.nodes.intent_node import intent_node
from im_copilot.graph.nodes.planner_node import planner_node
from im_copilot.graph.nodes.slide_node import slide_node
from im_copilot.graph.nodes.whiteboard_node import whiteboard_node


class MockLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, *args, **kwargs):
        m = MagicMock()
        m.content = self._content
        return m

    def with_structured_output(self, *args, **kwargs):
        return self


class IntentNodeTests(unittest.TestCase):
    @patch("im_copilot.graph.nodes.intent_node._llm")
    def test_classifies_doc_intent(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(intent_type="create_doc", topic="帮我写一份产品方案")
        result = intent_node({"raw_message": "帮我写一份产品方案"})
        self.assertEqual(result["intent_type"], "create_doc")
        self.assertEqual(result["intent_params"]["topic"], "帮我写一份产品方案")

    @patch("im_copilot.graph.nodes.intent_node._llm")
    def test_classifies_multi_intent(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(intent_type="create_multi", topic="Q2报告")
        result = intent_node({"raw_message": "帮我写报告并生成 PPT"})
        self.assertEqual(result["intent_type"], "create_multi")


class PlannerNodeTests(unittest.TestCase):
    @patch("im_copilot.graph.nodes.planner_node._llm")
    def test_maps_multi_intent_to_all_business_steps(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(plan=["doc", "whiteboard", "slide", "deliver"])
        result = planner_node({"intent_type": "create_multi"})
        self.assertEqual(result["plan"], ["doc", "whiteboard", "slide", "deliver"])

    @patch("im_copilot.graph.nodes.planner_node._llm")
    def test_maps_chat_to_deliver_only(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(plan=["deliver"])
        result = planner_node({"intent_type": "chat"})
        self.assertEqual(result["plan"], ["deliver"])


class MockNodeTests(unittest.TestCase):
    @patch("im_copilot.graph.nodes.doc_node._llm", MockLLM("文档内容"))
    def test_doc_node_adds_doc_result(self):
        result = doc_node({"mock_results": {}, "intent_params": {"topic": "测试"}})
        self.assertEqual(result["mock_results"]["doc"]["kind"], "doc")
        self.assertEqual(result["mock_results"]["doc"]["status"], "created")
        self.assertEqual(result["mock_results"]["doc"]["preview"], "文档内容")

    @patch("im_copilot.graph.nodes.whiteboard_node._llm", MockLLM("白板内容"))
    def test_whiteboard_node_preserves_existing_results(self):
        result = whiteboard_node({"mock_results": {"doc": {"kind": "doc", "title": "x", "status": "created", "preview": "x"}}, "intent_params": {"topic": "测试"}})
        self.assertIn("doc", result["mock_results"])
        self.assertEqual(result["mock_results"]["whiteboard"]["kind"], "whiteboard")

    @patch("im_copilot.graph.nodes.slide_node._llm", MockLLM("PPT内容"))
    def test_slide_node_adds_slide_result(self):
        result = slide_node({"mock_results": {}, "intent_params": {"topic": "测试"}})
        self.assertEqual(result["mock_results"]["slide"]["kind"], "slide")


class DeliverNodeTests(unittest.TestCase):
    @patch("im_copilot.graph.nodes.deliver_node._llm", MockLLM("收到，你好！"))
    def test_deliver_chat_summary(self):
        result = deliver_node({"intent_type": "chat", "plan": ["deliver"], "errors": [], "raw_message": "你好"})
        self.assertEqual(result["summary"], "收到，你好！")

    @patch("im_copilot.graph.nodes.deliver_node._llm", MockLLM("汇总结果"))
    def test_deliver_mock_results(self):
        result = deliver_node(
            {
                "intent_type": "create_multi",
                "plan": ["doc", "slide", "deliver"],
                "mock_results": {
                    "doc": {"kind": "doc", "title": "Mock doc", "status": "created", "preview": "doc done"},
                    "slide": {"kind": "slide", "title": "Mock slide", "status": "created", "preview": "slide done"},
                },
                "errors": [],
                "raw_message": "测试",
            }
        )
        self.assertEqual(result["summary"], "汇总结果")


if __name__ == "__main__":
    unittest.main()
