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
from im_copilot.lark_handlers import _calendar_summary
from im_copilot.lark_handlers import _meeting_card_source_summary
from im_copilot.lark_handlers import _send_meeting_confirmation_cards
from im_copilot.lark_handlers import _unverified_artifact_link_lines
from im_copilot.memory.group_board_store import GroupBoardItem
from im_copilot.memory.todo_extractor import TodoExtractionOutput, WindowMessage, extract_todos_from_window
from im_copilot.skills.config import get_skill_config
from im_copilot.skills.registry import get_skill, planner_capability_text


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

    @patch("im_copilot.graph.nodes.planner_node.get_llm_for_node")
    def test_planner_prompt_uses_registry_capabilities(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(
            plan=["deliver"],
            needs_clarification=False,
            questions=[],
        )
        mock_get_llm.return_value = mock_llm
        planner_node({"intent_type": "chat"})
        prompt = mock_llm.invoke.call_args.args[0]
        self.assertIn("lark_doc.create", prompt)
        self.assertIn("lark_whiteboard.create", prompt)
        self.assertIn("lark_slide.create", prompt)


class SkillRegistryTests(unittest.TestCase):
    def test_registry_has_create_skills(self):
        self.assertEqual(get_skill("lark_doc.create").plan_step, "doc")
        self.assertEqual(get_skill("lark_whiteboard.create").plan_step, "whiteboard")
        self.assertEqual(get_skill("lark_slide.create").plan_step, "slide")

    def test_planner_capability_text_from_registry(self):
        text = planner_capability_text()
        self.assertIn("lark_doc.create", text)
        self.assertIn("lark_whiteboard.create", text)
        self.assertIn("lark_slide.create", text)

    def test_skill_config_uses_defaults(self):
        self.assertIn("system_prompt", get_skill_config("lark_doc"))


class LarkReplyGuardTests(unittest.TestCase):
    def test_flags_feishu_artifact_link_without_result_artifact(self):
        result = MagicMock(
            artifacts={},
            summary="已生成：[打开文档](https://jcneyh7qlo8i.feishu.cn/docx/UfCIdmhmeoGpIBxSuXMcnMLunIb)",
        )

        lines = _unverified_artifact_link_lines(result)

        self.assertEqual(lines, ["- https://jcneyh7qlo8i.feishu.cn/docx/UfCIdmhmeoGpIBxSuXMcnMLunIb"])

    def test_allows_source_feishu_links_without_result_artifact(self):
        url = "https://jcneyh7qlo8i.feishu.cn/docx/UfCIdmhmeoGpIBxSuXMcnMLunIb"
        result = MagicMock(artifacts={}, summary=f"这个链接是 {url}")

        lines = _unverified_artifact_link_lines(result, source_text=f"帮我看这个 {url}")

        self.assertEqual(lines, [])


class TodoExtractorTests(unittest.TestCase):
    @patch("im_copilot.memory.todo_extractor.invoke_structured")
    def test_bot_artifact_request_does_not_create_personal_todo(self, mock_invoke):
        mock_invoke.return_value = TodoExtractionOutput(items=[])
        extract_todos_from_window(
            [
                WindowMessage(
                    message_id="m1",
                    open_id="ou_sender",
                    name="",
                    text="帮我们汇总讨论，生成一份技术方案文档，并在下周一前生成汇报 PPT",
                    ts=0,
                    is_trigger=True,
                )
            ],
            existing_open_todos=[],
            is_bot_request=True,
        )

        prompt = mock_invoke.call_args.args[2]

        self.assertIn("is_bot_request：True", prompt)
        self.assertIn("这类请求由 Agent 执行，不创建个人待办", prompt)
        self.assertIn("action_phrase 是该待办对应的动作短语", prompt)
        self.assertIn("绝对不要输出 create / update / new", prompt)
        self.assertIn("触发消息内含 @ mention", prompt)
        self.assertIn("回到窗口里找出真正的承诺者", prompt)
        self.assertIn("仅当触发消息和某条 existing todo 是严格语义上同一件事", prompt)
        self.assertIn("新的子任务", prompt)


class MeetingCardTests(unittest.TestCase):
    @patch("im_copilot.lark_handlers.get_llm_for_node")
    def test_meeting_card_source_summary_uses_llm(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="群聊明确提出今晚 19:00 在会议室对齐。")
        mock_get_llm.return_value = mock_llm

        result = _meeting_card_source_summary(
            title="团队对齐会",
            start="2026-05-03T19:00+08:00",
            end="2026-05-03T19:30+08:00",
            source_text="今晚 7 点在会议室拉会对齐",
        )

        self.assertEqual(result, "群聊明确提出今晚 19:00 在会议室对齐。")

    @patch("im_copilot.lark_handlers._meeting_card_source_summary", return_value="群聊明确提出会议候选。")
    def test_meeting_confirmation_card_is_sent_to_organizer(self, mock_summary):
        item = GroupBoardItem(
            id=42,
            chat_id="oc_group",
            message_id="om_1",
            item_type="meeting",
            title="评审会",
            owner_open_id="ou_liran",
            owner_name="李然",
            due_at="2026-05-04T19:00+08:00",
            status="pending_confirmation",
            source_open_id="ou_organizer",
            source_text="今晚 19:00 开评审会",
            metadata_json=(
                '{"recipients":["ou_liran"],"start":"2026-05-04T19:00+08:00",'
                '"end":"2026-05-04T19:30+08:00","topic":"评审会"}'
            ),
            created_at=0,
            updated_at=0,
        )
        bot = MagicMock()
        bot.send_ephemeral_card.return_value = {"code": 0}

        _send_meeting_confirmation_cards(bot, [item])

        bot.send_ephemeral_card.assert_called_once()
        self.assertEqual(bot.send_ephemeral_card.call_args.args[0], "oc_group")
        self.assertEqual(bot.send_ephemeral_card.call_args.args[1], "ou_organizer")
        mock_summary.assert_called_once()

    def test_calendar_summary_prefers_metadata_topic(self):
        item = GroupBoardItem(
            id=1,
            chat_id="oc_group",
            message_id="om_1",
            item_type="meeting",
            title="今晚 19 点评审会",
            owner_open_id="ou_liran",
            owner_name="李然",
            due_at="",
            status="pending_confirmation",
            source_open_id="ou_organizer",
            source_text="今晚 19 点开评审会",
            metadata_json='{"topic":"评审会"}',
            created_at=0,
            updated_at=0,
        )

        self.assertEqual(_calendar_summary(item), "评审会")

    def test_calendar_summary_uses_title_when_topic_missing(self):
        item = GroupBoardItem(
            id=1,
            chat_id="oc_group",
            message_id="om_1",
            item_type="meeting",
            title="今晚 19 点评审会",
            owner_open_id="ou_liran",
            owner_name="李然",
            due_at="",
            status="pending_confirmation",
            source_open_id="ou_organizer",
            source_text="今晚 19 点开评审会",
            metadata_json="{}",
            created_at=0,
            updated_at=0,
        )

        self.assertEqual(_calendar_summary(item), "今晚 19 点评审会")

    def test_calendar_summary_uses_default_for_empty_item(self):
        item = GroupBoardItem(
            id=1,
            chat_id="oc_group",
            message_id="om_1",
            item_type="meeting",
            title="",
            owner_open_id="ou_liran",
            owner_name="李然",
            due_at="",
            status="pending_confirmation",
            source_open_id="ou_organizer",
            source_text="今晚 19 点开评审会",
            metadata_json="not json",
            created_at=0,
            updated_at=0,
        )

        self.assertEqual(_calendar_summary(item), "群聊会议")


class MockNodeTests(unittest.TestCase):
    @patch("im_copilot.skills.lark_doc.run_lark_cli")
    @patch("im_copilot.skills.lark_doc.get_llm_for_node")
    def test_doc_node_adds_doc_result(self, mock_get_llm, mock_run_cli):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(content="文档内容")
        mock_get_llm.return_value = mock_llm
        mock_run_cli.return_value = {
            "data": {"document": {"document_id": "doc123", "url": "https://x/doc123"}}
        }
        result = doc_node({"artifacts": {}, "intent_params": {"topic": "测试"}, "user_access_token": "uat"})
        self.assertEqual(result["artifacts"]["doc"]["kind"], "doc")
        self.assertEqual(result["artifacts"]["doc"]["status"], "created")
        self.assertEqual(result["artifacts"]["doc"]["preview"], "文档内容")

    @patch("im_copilot.skills.lark_whiteboard.run_lark_cli")
    @patch("im_copilot.skills.lark_whiteboard.get_llm_for_node")
    def test_whiteboard_node_preserves_existing_results(self, mock_get_llm, mock_run_cli):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(content="白板内容")
        mock_get_llm.return_value = mock_llm
        mock_run_cli.side_effect = [
            {
                "data": {
                    "document": {
                        "document_id": "doc123",
                        "url": "https://x/doc123",
                        "new_blocks": [
                            {"block_type": "whiteboard", "block_token": "wb123"}
                        ],
                    }
                }
            },
            {"ok": True},
        ]
        result = whiteboard_node({
            "artifacts": {"doc": {"kind": "doc", "title": "x", "status": "created", "preview": "x"}},
            "intent_params": {"topic": "测试"},
            "user_access_token": "uat",
        })
        self.assertIn("doc", result["artifacts"])
        self.assertEqual(result["artifacts"]["whiteboard"]["kind"], "whiteboard")
        self.assertEqual(result["artifacts"]["whiteboard"]["status"], "created")

    @patch("im_copilot.skills.lark_slide.run_lark_cli")
    @patch("im_copilot.skills.lark_slide.get_llm_for_node")
    def test_slide_node_adds_slide_result(self, mock_get_llm, mock_run_cli):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(content='["<slide xmlns=\\"http://www.larkoffice.com/sml/2.0\\"><data><span fontSize=\\"36\\">测试标题</span></data></slide>"]')
        mock_get_llm.return_value = mock_llm
        mock_run_cli.return_value = {"data": {"xml_presentation_id": "ppt123", "url": "https://x/ppt123"}}
        result = slide_node({"artifacts": {}, "intent_params": {"topic": "测试"}, "user_access_token": "uat"})
        self.assertEqual(result["artifacts"]["slide"]["kind"], "slide")
        self.assertEqual(result["artifacts"]["slide"]["status"], "created")

    @patch("im_copilot.skills.lark_doc.run_lark_cli")
    @patch("im_copilot.skills.lark_doc.get_llm_for_node")
    def test_doc_node_returns_draft_without_token(self, mock_get_llm, mock_run_cli):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(content="文档内容")
        mock_get_llm.return_value = mock_llm
        mock_run_cli.return_value = {"data": {"document": {}}}
        result = doc_node({"artifacts": {}, "intent_params": {"topic": "测试"}, "user_access_token": "uat"})
        self.assertEqual(result["artifacts"]["doc"]["status"], "draft")

    @patch("im_copilot.skills.lark_whiteboard.get_llm_for_node")
    def test_whiteboard_node_returns_draft_without_user_token(self, mock_get_llm):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(content="graph TD\nA-->B")
        mock_get_llm.return_value = mock_llm
        result = whiteboard_node({"artifacts": {}, "intent_params": {"topic": "测试"}})
        self.assertEqual(result["artifacts"]["whiteboard"]["status"], "draft")

    @patch("im_copilot.skills.lark_slide.run_lark_cli")
    @patch("im_copilot.skills.lark_slide.get_llm_for_node")
    def test_slide_node_returns_draft_on_cli_exception(self, mock_get_llm, mock_run_cli):
        mock_llm = MockLLM()
        mock_llm.invoke.return_value = MagicMock(content='["<slide xmlns=\\"http://www.larkoffice.com/sml/2.0\\"><data><span fontSize=\\"36\\">测试标题</span></data></slide>"]')
        mock_get_llm.return_value = mock_llm
        mock_run_cli.side_effect = RuntimeError("failed")
        result = slide_node({"artifacts": {}, "intent_params": {"topic": "测试"}, "user_access_token": "uat"})
        self.assertEqual(result["artifacts"]["slide"]["status"], "draft")


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


class SlideXmlDirectTests(unittest.TestCase):
    def test_cover_title_from_xml_picks_largest_font(self):
        from im_copilot.skills.lark_slide import _cover_title_from_xml
        xml = (
            '<slide xmlns="http://www.larkoffice.com/sml/2.0">'
            '<data>'
            '<shape type="text"><content><p>'
            '<strong><span color="rgb(255,255,255)" fontSize="44">季度汇报</span></strong>'
            '</p></content></shape>'
            '<shape type="text"><content><p>'
            '<span color="rgb(148,163,184)" fontSize="20">副标题</span>'
            '</p></content></shape>'
            '</data></slide>'
        )
        self.assertEqual(_cover_title_from_xml(xml), "季度汇报")

    def test_cover_title_from_xml_returns_empty_on_no_span(self):
        from im_copilot.skills.lark_slide import _cover_title_from_xml
        xml = '<slide><data><shape type="text"><content><p>纯文字</p></content></shape></data></slide>'
        self.assertEqual(_cover_title_from_xml(xml), "")

    def test_cover_title_from_xml_skips_whitespace_only(self):
        from im_copilot.skills.lark_slide import _cover_title_from_xml
        xml = (
            '<slide><data>'
            '<shape type="text"><content><p>'
            '<span color="rgb(255,255,255)" fontSize="44">  </span>'
            '</p></content></shape>'
            '<shape type="text"><content><p>'
            '<span color="rgb(148,163,184)" fontSize="20">真实标题</span>'
            '</p></content></shape>'
            '</data></slide>'
        )
        self.assertEqual(_cover_title_from_xml(xml), "真实标题")

    @patch("im_copilot.skills.lark_slide.get_llm_for_node")
    def test_generate_slide_xml_uses_xml_directly(self, mock_get_llm):
        from im_copilot.skills.lark_slide import generate_slide_xml
        import json
        slide_xml = (
            '<slide xmlns="http://www.larkoffice.com/sml/2.0">'
            '<style><fill><fillColor color="rgb(15,23,42)"/></fill></style>'
            '<data><shape type="text" topLeftX="80" topLeftY="160" width="800" height="70">'
            '<content><p textAlign="center"><strong>'
            '<span color="rgb(255,255,255)" fontSize="44">季度汇报</span>'
            '</strong></p></content></shape></data></slide>'
        )
        mock_llm = MockLLM()
        # LLM 直接返回 JSON XML 数组
        mock_llm.invoke.return_value = MagicMock(content=json.dumps([slide_xml]))
        mock_get_llm.return_value = mock_llm

        result_json, cover_title = generate_slide_xml("季度汇报 PPT")
        result = json.loads(result_json)
        # 结果是 XML 数组，直接传透，不经过 Python 渲染器
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn("季度汇报", result[0])
        # 封面标题从 XML 提取，不是 fallback
        self.assertEqual(cover_title, "季度汇报")

    @patch("im_copilot.skills.lark_slide.get_llm_for_node")
    def test_generate_slide_xml_falls_back_to_json_outline(self, mock_get_llm):
        from im_copilot.skills.lark_slide import generate_slide_xml
        import json
        mock_llm = MockLLM()
        # LLM 返回 JSON 轮廓（旧格式）
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "style": "business",
            "slides": [
                {"layout": "cover", "title": "旧格式标题", "subtitle": "", "bullets": ["要点"], "metrics": []},
                {"layout": "content", "title": "内容页", "subtitle": "", "bullets": ["要点 A", "要点 B"], "metrics": []},
                {"layout": "content", "title": "内容页2", "subtitle": "", "bullets": ["C", "D"], "metrics": []},
                {"layout": "action", "title": "行动", "subtitle": "", "bullets": ["任务 1"], "metrics": []},
                {"layout": "closing", "title": "感谢", "subtitle": "", "bullets": [], "metrics": []},
            ]
        }))
        mock_get_llm.return_value = mock_llm

        result_json, cover_title = generate_slide_xml("旧格式 PPT")
        result = json.loads(result_json)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # fallback 路径通过 deck 提取 cover title
        self.assertEqual(cover_title, "旧格式标题")


if __name__ == "__main__":
    unittest.main()
