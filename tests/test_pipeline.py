import unittest
from unittest.mock import MagicMock, patch

from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from im_copilot.graph.pipeline import build_pipeline, run_pipeline


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

    @patch("im_copilot.graph.nodes.side_agent_node._get_llm")
    @patch("im_copilot.graph.nodes.verify_node._get_llm")
    @patch("im_copilot.graph.nodes.intent_node._get_llm")
    @patch("im_copilot.graph.nodes.planner_node._get_llm")
    @patch("im_copilot.graph.nodes.doc_node._get_llm")
    @patch("im_copilot.graph.nodes.whiteboard_node._get_llm")
    @patch("im_copilot.graph.nodes.slide_node._get_llm")
    @patch("im_copilot.graph.nodes.deliver_node._get_llm")
    def test_multi_input_invokes_doc_whiteboard_slide(self, mock_deliver, mock_slide, mock_wb, mock_doc, mock_planner, mock_intent, mock_verify, mock_side_agent):
        mock_intent.return_value = MockLLM()
        mock_intent.return_value.invoke.return_value = MagicMock(intent_type="create_multi", topic="报告", confidence=0.9)
        mock_planner.return_value = MockLLM()
        mock_planner.return_value.invoke.return_value = MagicMock(plan=["doc", "whiteboard", "slide", "deliver"], needs_clarification=False, questions=[])
        mock_doc.return_value = MockLLM()
        mock_doc.return_value.invoke.return_value = MagicMock(content="doc内容")
        mock_wb.return_value = MockLLM()
        mock_wb.return_value.invoke.return_value = MagicMock(content="wb内容")
        mock_slide.return_value = MockLLM()
        mock_slide.return_value.invoke.return_value = MagicMock(content="slide内容")
        mock_deliver.return_value = MockLLM()
        mock_deliver.return_value.invoke.return_value = MagicMock(content="汇总结果")
        mock_verify.return_value = MockLLM()
        mock_verify.return_value.invoke.return_value = MagicMock(status="pass", reason="质量合格")
        mock_side_agent.return_value = MockLLM()
        mock_side_agent.return_value.invoke.return_value = MagicMock(
            validation_score=0.95,
            relevance="高度相关",
            completeness="完整",
            accuracy="准确",
            readability="清晰",
            issues=[],
        )

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
        # Handle plan_approval interrupt
        if "__interrupt__" in result:
            result = graph.invoke(Command(resume={"approved": True, "feedback": "test"}), config=config)

        self.assertEqual(result["intent_type"], "create_multi")
        self.assertEqual(result["plan"], ["doc", "whiteboard", "slide", "deliver"])
        self.assertIn("doc", result["artifacts"])
        self.assertIn("whiteboard", result["artifacts"])
        self.assertIn("slide", result["artifacts"])
        self.assertIn("汇总结果", result["summary"])

    @patch("im_copilot.graph.nodes.side_agent_node._get_llm")
    @patch("im_copilot.graph.nodes.verify_node._get_llm")
    @patch("im_copilot.graph.nodes.intent_node._get_llm")
    @patch("im_copilot.graph.nodes.planner_node._get_llm")
    @patch("im_copilot.graph.nodes.whiteboard_node._get_llm")
    @patch("im_copilot.graph.nodes.deliver_node._get_llm")
    def test_whiteboard_only_input(self, mock_deliver, mock_wb, mock_planner, mock_intent, mock_verify, mock_side_agent):
        mock_intent.return_value = MockLLM()
        mock_intent.return_value.invoke.return_value = MagicMock(intent_type="create_whiteboard", topic="流程图", confidence=0.9)
        mock_planner.return_value = MockLLM()
        mock_planner.return_value.invoke.return_value = MagicMock(plan=["whiteboard", "deliver"], needs_clarification=False, questions=[])
        mock_wb.return_value = MockLLM()
        mock_wb.return_value.invoke.return_value = MagicMock(content="wb内容")
        mock_deliver.return_value = MockLLM()
        mock_deliver.return_value.invoke.return_value = MagicMock(content="汇总结果")
        mock_verify.return_value = MockLLM()
        mock_verify.return_value.invoke.return_value = MagicMock(status="pass", reason="质量合格")
        mock_side_agent.return_value = MockLLM()
        mock_side_agent.return_value.invoke.return_value = MagicMock(
            validation_score=0.95,
            relevance="高度相关",
            completeness="完整",
            accuracy="准确",
            readability="清晰",
            issues=[],
        )

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
        if "__interrupt__" in result:
            result = graph.invoke(Command(resume={"approved": True, "feedback": "test"}), config=config)

        self.assertEqual(result["intent_type"], "create_whiteboard")
        self.assertEqual(result["plan"], ["whiteboard", "deliver"])
        self.assertEqual(set(result["artifacts"].keys()), {"whiteboard"})

    @patch("im_copilot.graph.nodes.intent_node._get_llm")
    @patch("im_copilot.graph.nodes.planner_node._get_llm")
    @patch("im_copilot.graph.nodes.deliver_node._get_llm")
    def test_chat_input(self, mock_deliver, mock_planner, mock_intent):
        mock_intent.return_value = MockLLM()
        mock_intent.return_value.invoke.return_value = MagicMock(intent_type="chat", topic="你好", confidence=0.9)
        mock_planner.return_value = MockLLM()
        mock_planner.return_value.invoke.return_value = MagicMock(plan=["deliver"], needs_clarification=False, questions=[])
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
        if "__interrupt__" in result:
            result = graph.invoke(Command(resume={"approved": True, "feedback": "test"}), config=config)

        self.assertEqual(result["intent_type"], "chat")
        self.assertEqual(result["plan"], ["deliver"])
        self.assertNotIn("artifacts", result)
        self.assertEqual(result["summary"], "你好！有什么可以帮你的？")


if __name__ == "__main__":
    unittest.main()
