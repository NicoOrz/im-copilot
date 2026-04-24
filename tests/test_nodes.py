import unittest

from im_copilot.state import PipelineState


class StateTests(unittest.TestCase):
    def test_pipeline_state_type_imports(self):
        state: PipelineState = {
            "raw_message": "hello",
            "chat_id": "cli",
            "message_id": "cli",
            "source": "cli",
            "errors": [],
            "checks": [],
        }

        self.assertEqual(state["source"], "cli")


if __name__ == "__main__":
    unittest.main()
