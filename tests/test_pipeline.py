import unittest

from im_copilot.graph.pipeline import build_pipeline, run_pipeline


class PipelineTests(unittest.TestCase):
    def test_build_pipeline(self):
        graph = build_pipeline()
        self.assertIsNotNone(graph)

    def test_multi_input_invokes_doc_whiteboard_slide(self):
        result = run_pipeline("帮我写一份报告，画流程图，并生成 PPT")

        self.assertEqual(result["intent_type"], "create_multi")
        self.assertEqual(result["plan"], ["doc", "whiteboard", "slide", "deliver"])
        self.assertIn("doc", result["mock_results"])
        self.assertIn("whiteboard", result["mock_results"])
        self.assertIn("slide", result["mock_results"])
        self.assertIn("Phase 1 mock 结果", result["summary"])

    def test_whiteboard_only_input(self):
        result = run_pipeline("帮我画一个项目流程图")

        self.assertEqual(result["intent_type"], "create_whiteboard")
        self.assertEqual(result["plan"], ["whiteboard", "deliver"])
        self.assertEqual(set(result["mock_results"].keys()), {"whiteboard"})

    def test_chat_input(self):
        result = run_pipeline("你好")

        self.assertEqual(result["intent_type"], "chat")
        self.assertEqual(result["plan"], ["deliver"])
        self.assertNotIn("mock_results", result)
        self.assertIn("Phase 1", result["summary"])


if __name__ == "__main__":
    unittest.main()
