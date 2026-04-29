import unittest

from im_copilot.commands import CommandResult, execute_command, parse_command
from im_copilot.graph.nodes.history_utils import format_history


class ParseCommandTests(unittest.TestCase):
    def test_new(self):
        result = parse_command("/new")
        self.assertEqual(result, ("new", ""))

    def test_help(self):
        result = parse_command("/help")
        self.assertEqual(result, ("help", ""))

    def test_with_args(self):
        result = parse_command("/new some extra args")
        self.assertEqual(result, ("new", "some extra args"))

    def test_regular_text(self):
        self.assertIsNone(parse_command("帮我写一份报告"))

    def test_empty_string(self):
        self.assertIsNone(parse_command(""))

    def test_whitespace(self):
        result = parse_command("  /new  ")
        self.assertEqual(result, ("new", ""))

    def test_slash_in_middle(self):
        self.assertIsNone(parse_command("请帮我 /new 一下"))

    def test_case_insensitive(self):
        result = parse_command("/NEW")
        self.assertEqual(result, ("new", ""))


class ExecuteCommandTests(unittest.TestCase):
    def test_new_command(self):
        result = execute_command("new", "", "chat1", "thread1", "feishu")
        self.assertTrue(result.handled)
        self.assertEqual(result.command, "new")
        self.assertEqual(result.metadata["action"], "reset_thread")

    def test_help_command(self):
        result = execute_command("help", "", "chat1", "thread1", "feishu")
        self.assertTrue(result.handled)
        self.assertEqual(result.command, "help")
        self.assertIn("/new", result.response_text)

    def test_unknown_command(self):
        result = execute_command("foobar", "", "chat1", "thread1", "feishu")
        self.assertFalse(result.handled)
        self.assertIn("未知命令", result.response_text)


class FormatHistoryTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(format_history([]), "（无历史对话）")

    def test_recent_only(self):
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        ]
        result = format_history(history, recent_n=6)
        self.assertIn("user: 你好", result)
        self.assertIn("assistant: 你好！有什么可以帮你的？", result)

    def test_progressive_truncation(self):
        history = [{"role": "user", "content": f"消息{i}" * 20} for i in range(10)]
        result = format_history(history, recent_n=3, summary_n=4)
        lines = result.strip().split("\n")
        older_lines = [l for l in lines if l.startswith("[")]
        recent_lines = [l for l in lines if l.startswith("user:")]
        self.assertEqual(len(recent_lines), 3)
        self.assertTrue(len(older_lines) <= 4)
        for line in older_lines:
            self.assertIn("...", line)


if __name__ == "__main__":
    unittest.main()
