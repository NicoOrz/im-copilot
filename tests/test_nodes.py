import unittest

from im_copilot.graph.nodes.deliver_node import deliver_node
from im_copilot.graph.nodes.doc_node import doc_node
from im_copilot.graph.nodes.intent_node import intent_node
from im_copilot.graph.nodes.planner_node import planner_node
from im_copilot.graph.nodes.slide_node import slide_node
from im_copilot.graph.nodes.whiteboard_node import whiteboard_node


class IntentNodeTests(unittest.TestCase):
    def test_classifies_doc_intent(self):
        result = intent_node({"raw_message": "帮我写一份产品方案"})
        self.assertEqual(result["intent_type"], "create_doc")
        self.assertEqual(result["intent_params"]["topic"], "帮我写一份产品方案")

    def test_classifies_whiteboard_intent(self):
        result = intent_node({"raw_message": "帮我画一个项目流程图"})
        self.assertEqual(result["intent_type"], "create_whiteboard")

    def test_classifies_slide_intent(self):
        result = intent_node({"raw_message": "帮我生成 PPT"})
        self.assertEqual(result["intent_type"], "create_slide")

    def test_classifies_multi_intent(self):
        result = intent_node({"raw_message": "帮我写报告并生成 PPT"})
        self.assertEqual(result["intent_type"], "create_multi")

    def test_classifies_chat_intent(self):
        result = intent_node({"raw_message": "你好"})
        self.assertEqual(result["intent_type"], "chat")


class PlannerNodeTests(unittest.TestCase):
    def test_maps_multi_intent_to_all_business_steps(self):
        result = planner_node({"intent_type": "create_multi"})
        self.assertEqual(result["plan"], ["doc", "whiteboard", "slide", "deliver"])

    def test_maps_chat_to_deliver_only(self):
        result = planner_node({"intent_type": "chat"})
        self.assertEqual(result["plan"], ["deliver"])

    def test_returns_fresh_plan_list(self):
        first = planner_node({"intent_type": "create_doc"})
        first["plan"].append("slide")

        second = planner_node({"intent_type": "create_doc"})
        self.assertEqual(second["plan"], ["doc", "deliver"])


class MockNodeTests(unittest.TestCase):
    def test_doc_node_adds_doc_result(self):
        result = doc_node({"mock_results": {}})
        self.assertEqual(result["mock_results"]["doc"]["kind"], "doc")
        self.assertEqual(result["mock_results"]["doc"]["status"], "created")

    def test_whiteboard_node_preserves_existing_results(self):
        result = whiteboard_node({"mock_results": {"doc": {"kind": "doc", "title": "x", "status": "created", "preview": "x"}}})
        self.assertIn("doc", result["mock_results"])
        self.assertEqual(result["mock_results"]["whiteboard"]["kind"], "whiteboard")

    def test_slide_node_adds_slide_result(self):
        result = slide_node({"mock_results": {}})
        self.assertEqual(result["mock_results"]["slide"]["kind"], "slide")


class DeliverNodeTests(unittest.TestCase):
    def test_deliver_chat_summary(self):
        result = deliver_node({"intent_type": "chat", "plan": ["deliver"], "errors": []})
        self.assertIn("收到", result["summary"])

    def test_deliver_mock_results_in_plan_order(self):
        result = deliver_node(
            {
                "intent_type": "create_multi",
                "plan": ["doc", "slide", "deliver"],
                "mock_results": {
                    "doc": {"kind": "doc", "title": "Mock doc", "status": "created", "preview": "doc done"},
                    "slide": {"kind": "slide", "title": "Mock slide", "status": "created", "preview": "slide done"},
                },
                "errors": [],
            }
        )

        self.assertLess(result["summary"].index("doc done"), result["summary"].index("slide done"))


if __name__ == "__main__":
    unittest.main()
