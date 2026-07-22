import unittest

from agent_core.context import Session
from agent_core.tools import ToolError, build_default_registry


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_default_registry()
        self.session = Session(session_id="s1", user_id="u1")

    def test_registry_has_at_least_three_tools_with_schema(self) -> None:
        self.assertGreaterEqual(len(self.registry.names()), 3)
        for schema in self.registry.schemas():
            self.assertIn("name", schema)
            self.assertIn("description", schema)
            self.assertIn("properties", schema["parameters"])

    def test_calculator_basic(self) -> None:
        result = self.registry.call("calculator", {"expression": "(3 + 4) * 2"}, self.session)
        self.assertIn("14", result)

    def test_calculator_rejects_unsafe_expression(self) -> None:
        with self.assertRaises(ToolError):
            self.registry.call(
                "calculator", {"expression": "__import__('os').system('echo hi')"}, self.session
            )

    def test_search_mock_hit_and_miss(self) -> None:
        hit = self.registry.call("search", {"query": "什么是 python"}, self.session)
        self.assertIn("Python", hit)
        miss = self.registry.call("search", {"query": "一个不存在的关键词xyz"}, self.session)
        self.assertIn("没有找到", miss)

    def test_weather_mock(self) -> None:
        result = self.registry.call("weather", {"city": "Beijing"}, self.session)
        self.assertIn("北京", result)

    def test_todo_is_session_scoped(self) -> None:
        self.registry.call("todo", {"action": "add", "item": "买菜"}, self.session)
        listing = self.registry.call("todo", {"action": "list"}, self.session)
        self.assertIn("买菜", listing)

        other_session = Session(session_id="s2", user_id="u1")
        other_listing = self.registry.call("todo", {"action": "list"}, other_session)
        self.assertNotIn("买菜", other_listing)

        done = self.registry.call("todo", {"action": "done", "index": 1}, self.session)
        self.assertIn("买菜", done)
        self.assertEqual(self.registry.call("todo", {"action": "list"}, self.session), "当前没有待办事项。")

    def test_unknown_tool_raises_tool_error(self) -> None:
        with self.assertRaises(ToolError):
            self.registry.call("no_such_tool", {}, self.session)


if __name__ == "__main__":
    unittest.main()
