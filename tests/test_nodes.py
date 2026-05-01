import unittest
from unittest.mock import MagicMock, patch

from im_copilot.graph.nodes.deliver_node import deliver_node
from im_copilot.graph.nodes.doc_node import doc_node
from im_copilot.graph.nodes.intent_node import intent_node
from im_copilot.graph.nodes.planner_node import planner_node
from im_copilot.graph.nodes.slide_node import slide_node
from im_copilot.graph.nodes.side_agent_node import side_agent_node
from im_copilot.graph.nodes.verify_node import verify_node
from im_copilot.graph.nodes.whiteboard_node import whiteboard_node


class MockLLM:
    """Mock LLM whose invoke method is a MagicMock, allowing dynamic return values."""

    def __init__(self):
        self.invoke = MagicMock()

    def with_structured_output(self, *args, **kwargs):
        return self


class IntentNodeTests(unittest.TestCase):
    @patch("im_copilot.graph.nodes.intent_node.get_llm_for_node")
    def test_classifies_doc_intent(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(intent_type="create_doc", topic="帮我写一份产品方案", confidence=0.95)
        mock_get_llm.return_value = mock_llm
        result = intent_node({"raw_message": "帮我写一份产品方案"})
        self.assertEqual(result["intent_type"], "create_doc")
        self.assertEqual(result["intent_params"]["topic"], "帮我写一份产品方案")

    @patch("im_copilot.graph.nodes.intent_node.get_llm_for_node")
    def test_classifies_multi_intent(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(intent_type="create_multi", topic="Q2报告", confidence=0.9)
        mock_get_llm.return_value = mock_llm
        result = intent_node({"raw_message": "帮我写报告并生成 PPT"})
        self.assertEqual(result["intent_type"], "create_multi")

    @patch("im_copilot.graph.nodes.intent_node.get_llm_for_node")
    def test_includes_artifact_context_for_followup_intent(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(
            intent_type="create_whiteboard",
            topic="修改上一版白板",
            confidence=0.9,
            plan=["whiteboard", "deliver"],
        )
        mock_get_llm.return_value = mock_llm
        result = intent_node({
            "raw_message": "这个有问题，重新处理一下",
            "artifacts": {
                "whiteboard": {
                    "kind": "whiteboard",
                    "title": "白板：流程图",
                    "status": "created",
                    "url": "https://www.feishu.cn/docx/x",
                }
            },
        })
        prompt = mock_llm.invoke.call_args.args[0]
        self.assertIn("近期产物", prompt)
        self.assertIn("白板：流程图", prompt)
        self.assertEqual(result["intent_type"], "create_whiteboard")
        self.assertEqual(result["plan"], ["whiteboard", "deliver"])
        self.assertEqual(result["artifacts"], {})

    @patch("im_copilot.graph.nodes.intent_node.get_llm_for_node")
    def test_chat_intent_keeps_existing_artifacts_out_of_update(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(intent_type="chat", topic="继续聊", confidence=0.9)
        mock_get_llm.return_value = mock_llm
        result = intent_node({
            "raw_message": "谢谢",
            "artifacts": {"whiteboard": {"kind": "whiteboard", "title": "白板：流程图"}},
        })
        self.assertEqual(result["intent_type"], "chat")
        self.assertEqual(result["plan"], ["deliver"])
        self.assertNotIn("artifacts", result)


class PlannerNodeTests(unittest.TestCase):
    @patch("im_copilot.graph.nodes.planner_node.get_llm_for_node")
    def test_maps_multi_intent_to_all_business_steps(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(
            plan=["doc", "whiteboard", "slide", "deliver"],
            needs_clarification=False,
            questions=[],
        )
        mock_get_llm.return_value = mock_llm
        result = planner_node({"intent_type": "create_multi"})
        self.assertEqual(result["plan"], ["doc", "whiteboard", "slide", "deliver"])

    @patch("im_copilot.graph.nodes.planner_node.get_llm_for_node")
    def test_maps_chat_to_deliver_only(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(
            plan=["deliver"],
            needs_clarification=False,
            questions=[],
        )
        mock_get_llm.return_value = mock_llm
        result = planner_node({"intent_type": "chat"})
        self.assertEqual(result["plan"], ["deliver"])

    @patch("im_copilot.graph.nodes.planner_node.get_llm_for_node")
    def test_needs_clarification(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(
            plan=[],
            needs_clarification=True,
            questions=["目标受众是谁？", "需要什么格式？"],
        )
        mock_get_llm.return_value = mock_llm
        result = planner_node({"intent_type": "create_doc", "intent_confidence": 0.3})
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["pending_questions"], ["目标受众是谁？", "需要什么格式？"])


class MockNodeTests(unittest.TestCase):
    @patch("im_copilot.graph.nodes.doc_node.get_llm_for_node")
    def test_doc_node_adds_doc_result(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(content="文档内容")
        mock_get_llm.return_value = mock_llm
        result = doc_node({"artifacts": {}, "intent_params": {"topic": "测试"}})
        self.assertEqual(result["artifacts"]["doc"]["kind"], "doc")
        self.assertEqual(result["artifacts"]["doc"]["status"], "created")
        self.assertEqual(result["artifacts"]["doc"]["preview"], "文档内容")

    @patch("im_copilot.graph.nodes.whiteboard_node.get_llm_for_node")
    def test_whiteboard_node_preserves_existing_results(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(content="白板内容")
        mock_get_llm.return_value = mock_llm
        result = whiteboard_node({"artifacts": {"doc": {"kind": "doc", "title": "x", "status": "created", "preview": "x"}}, "intent_params": {"topic": "测试"}})
        self.assertIn("doc", result["artifacts"])
        self.assertEqual(result["artifacts"]["whiteboard"]["kind"], "whiteboard")

    @patch("im_copilot.graph.nodes.slide_node.get_llm_for_node")
    def test_slide_node_adds_slide_result(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(content="PPT内容")
        mock_get_llm.return_value = mock_llm
        result = slide_node({"artifacts": {}, "intent_params": {"topic": "测试"}})
        self.assertEqual(result["artifacts"]["slide"]["kind"], "slide")


class DeliverNodeTests(unittest.TestCase):
    @patch("im_copilot.graph.nodes.deliver_node.get_llm_for_node")
    def test_deliver_chat_summary(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(content="收到，你好！")
        mock_get_llm.return_value = mock_llm
        result = deliver_node({"intent_type": "chat", "plan": ["deliver"], "errors": [], "raw_message": "你好"})
        self.assertEqual(result["summary"], "收到，你好！")

    @patch("im_copilot.graph.nodes.deliver_node.get_llm_for_node")
    def test_deliver_chat_retries_invalid_llm_response(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.side_effect = [
            IndexError("list index out of range"),
            MagicMock(content="重试成功"),
        ]
        mock_get_llm.return_value = mock_llm
        result = deliver_node({"intent_type": "chat", "plan": ["deliver"], "errors": [], "raw_message": "你好"})
        self.assertEqual(result["summary"], "重试成功")
        self.assertEqual(mock_llm.invoke.call_count, 2)

    @patch("im_copilot.graph.nodes.deliver_node.get_llm_for_node")
    def test_deliver_chat_invalid_llm_response_after_retry(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.side_effect = IndexError("list index out of range")
        mock_get_llm.return_value = mock_llm
        result = deliver_node({"intent_type": "chat", "plan": ["deliver"], "errors": [], "raw_message": "你好"})
        self.assertEqual(result["summary"], "抱歉，模型这次没有返回有效内容，请稍后再试。")
        self.assertEqual(mock_llm.invoke.call_count, 2)

    @patch("im_copilot.graph.nodes.deliver_node.get_llm_for_node")
    def test_deliver_artifacts(self, mock_get_llm):
        result = deliver_node(
            {
                "intent_type": "create_multi",
                "plan": ["doc", "slide", "deliver"],
                "artifacts": {
                    "doc": {"kind": "doc", "title": "Mock doc", "status": "created", "preview": "doc done"},
                    "slide": {"kind": "slide", "title": "Mock slide", "status": "created", "preview": "slide done"},
                },
                "errors": [],
                "raw_message": "测试",
            }
        )
        mock_get_llm.assert_not_called()
        self.assertEqual(result["summary"], "已完成。\n- Mock doc：已创建\n- Mock slide：已创建")


class VerifyNodeTests(unittest.TestCase):
    @patch("im_copilot.graph.nodes.verify_node.get_llm_for_node")
    def test_verify_pass(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(status="pass", reason="质量合格")
        mock_get_llm.return_value = mock_llm
        result = verify_node({
            "plan": ["doc", "deliver"],
            "artifacts": {"doc": {"kind": "doc", "title": "t", "status": "created", "preview": "content"}},
            "raw_message": "写文档",
            "intent_type": "create_doc",
            "reflection_iteration": 0,
        })
        self.assertEqual(result["checks"][0]["status"], "pass")
        self.assertEqual(result["reflection_iteration"], 1)

    @patch("im_copilot.graph.nodes.verify_node.get_llm_for_node")
    def test_verify_revise(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(status="revise", reason="内容不完整")
        mock_get_llm.return_value = mock_llm
        result = verify_node({
            "plan": ["doc", "deliver"],
            "artifacts": {"doc": {"kind": "doc", "title": "t", "status": "created", "preview": "content"}},
            "raw_message": "写文档",
            "intent_type": "create_doc",
            "reflection_iteration": 0,
        })
        self.assertEqual(result["checks"][0]["status"], "revise")
        self.assertEqual(result["checks"][0]["task"], "doc")

    def test_verify_no_content(self):
        result = verify_node({"plan": ["deliver"], "artifacts": {}})
        self.assertEqual(result["checks"][0]["status"], "pass")
        self.assertEqual(result["checks"][0]["task"], "none")


class SideAgentNodeTests(unittest.TestCase):
    @patch("im_copilot.graph.nodes.side_agent_node.get_llm_for_node")
    def test_side_agent_evaluates_content(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(
            validation_score=0.95,
            relevance="高度相关",
            completeness="完整",
            accuracy="准确",
            readability="清晰",
            issues=[],
        )
        mock_get_llm.return_value = mock_llm
        result = side_agent_node({
            "plan": ["doc", "deliver"],
            "artifacts": {"doc": {"kind": "doc", "title": "t", "status": "created", "preview": "content"}},
            "raw_message": "写文档",
            "intent_type": "create_doc",
        })
        self.assertEqual(len(result["side_agent_results"]), 1)
        self.assertEqual(result["side_agent_results"][0]["task"], "doc")
        self.assertEqual(result["side_agent_results"][0]["validation_score"], 0.95)

    def test_side_agent_no_content(self):
        result = side_agent_node({"plan": ["deliver"], "artifacts": {}})
        self.assertEqual(result["side_agent_results"][0]["task"], "none")


if __name__ == "__main__":
    unittest.main()
