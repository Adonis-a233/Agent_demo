import unittest

from agent_core import context as context_module
from agent_core.agent import Agent, AgentParseError, parse_llm_output
from agent_core.context import SessionManager
from agent_core.tools import build_default_registry
from tests.helpers import ScriptedLLM, json_action, json_final


def make_agent(responses, max_tool_steps=5):
    llm = ScriptedLLM(responses)
    agent = Agent(
        tool_registry=build_default_registry(),
        session_manager=SessionManager(),
        llm_fn=llm,
        max_tool_steps=max_tool_steps,
    )
    return agent, llm


class ParseLLMOutputTests(unittest.TestCase):
    def test_parses_final_answer(self) -> None:
        parsed = parse_llm_output(json_final("t", "42"))
        self.assertEqual(parsed.final_answer, "42")
        self.assertIsNone(parsed.action)

    def test_parses_action(self) -> None:
        parsed = parse_llm_output(json_action("t", "calculator", {"expression": "1+1"}))
        self.assertIsNone(parsed.final_answer)
        self.assertEqual(parsed.action["tool"], "calculator")

    def test_tolerates_code_fence(self) -> None:
        raw = "```json\n" + json_final("t", "ok") + "\n```"
        self.assertEqual(parse_llm_output(raw).final_answer, "ok")

    def test_raises_on_garbage(self) -> None:
        with self.assertRaises(AgentParseError):
            parse_llm_output("这不是 JSON")


class AgentBasicLoopTests(unittest.TestCase):
    def test_direct_reply_without_tool(self) -> None:
        agent, llm = make_agent([json_final("直接回答", "你好，有什么可以帮你？")])
        answer = agent.handle_message("s1", "u1", "你好")
        self.assertEqual(answer, "你好，有什么可以帮你？")
        self.assertEqual(len(llm.calls), 1)

    def test_tool_call_then_final_answer(self) -> None:
        responses = [
            json_action("需要计算", "calculator", {"expression": "12*(3+5)"}),
            json_final("已得到结果", "结果是 96"),
        ]
        agent, llm = make_agent(responses)
        answer = agent.handle_message("s1", "u1", "12*(3+5) 等于多少")
        self.assertEqual(answer, "结果是 96")
        self.assertEqual(len(llm.calls), 2)

        session = agent.session_manager.get("s1")
        tool_messages = [m for m in session.messages if m.role == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("96", tool_messages[0].content)

        # 第二次调用 LLM 时，工具执行结果应该已经出现在传给它的 messages 里
        second_call_contents = [m["content"] for m in llm.calls[1]]
        self.assertTrue(any("96" in c for c in second_call_contents))


class AgentSessionIsolationTests(unittest.TestCase):
    def test_two_windows_do_not_interfere(self) -> None:
        # 模拟：窗口1记"带伞"，窗口2记"写周报"，再回窗口1追问，只应看到"带伞"
        responses = [
            json_action("记待办", "todo", {"action": "add", "item": "带伞"}),
            json_final("done", "已记录：带伞"),
            json_action("记待办", "todo", {"action": "add", "item": "写周报"}),
            json_final("done", "已记录：写周报"),
            json_action("查待办", "todo", {"action": "list"}),
            json_final("done", "你的待办：带伞"),
        ]
        llm = ScriptedLLM(responses)
        manager = SessionManager()
        agent = Agent(tool_registry=build_default_registry(), session_manager=manager, llm_fn=llm)

        agent.handle_message("window1", "userA", "记一条待办：带伞")
        agent.handle_message("window2", "userA", "记一条待办：写周报")
        answer = agent.handle_message("window1", "userA", "我刚才记了什么待办？")

        self.assertIn("带伞", answer)
        self.assertNotIn("写周报", answer)
        self.assertEqual(manager.get("window1").todos, ["带伞"])
        self.assertEqual(manager.get("window2").todos, ["写周报"])


class AgentFollowUpTests(unittest.TestCase):
    def test_plain_conversation_follow_up_keeps_context(self) -> None:
        responses = [json_final("t1", "我是助手"), json_final("t2", "刚才你问了 '你是谁'")]
        agent, llm = make_agent(responses)
        agent.handle_message("s1", "u1", "你是谁")
        agent.handle_message("s1", "u1", "我刚才问了你什么？")

        second_call_contents = [m["content"] for m in llm.calls[1]]
        self.assertTrue(any("你是谁" in c for c in second_call_contents))
        self.assertTrue(any("我是助手" in c for c in second_call_contents))

    def test_follow_up_with_tool_reuses_earlier_state(self) -> None:
        responses = [
            json_action("记待办", "todo", {"action": "add", "item": "买菜"}),
            json_final("done", "已记录：买菜"),
            json_action("再查一次", "todo", {"action": "list"}),
            json_final("done", "你的待办：买菜"),
        ]
        agent, llm = make_agent(responses)
        agent.handle_message("s1", "u1", "记一条待办：买菜")
        answer = agent.handle_message("s1", "u1", "我现在有哪些待办？")
        self.assertIn("买菜", answer)


class AgentErrorHandlingTests(unittest.TestCase):
    def test_max_steps_protection_stops_infinite_tool_loop(self) -> None:
        max_steps = 3
        responses = [json_action("继续调用", "calculator", {"expression": "1+1"})] * max_steps
        agent, llm = make_agent(responses, max_tool_steps=max_steps)
        answer = agent.handle_message("s1", "u1", "一直算")
        self.assertIn("无法在限定步数内", answer)
        self.assertEqual(len(llm.calls), max_steps)

    def test_recovers_from_malformed_llm_output(self) -> None:
        responses = ["这不是 JSON，模型抽风了", json_final("重新回答", "抱歉刚才输出错了，这是正确答案")]
        agent, llm = make_agent(responses)
        answer = agent.handle_message("s1", "u1", "你好")
        self.assertEqual(answer, "抱歉刚才输出错了，这是正确答案")
        session = agent.session_manager.get("s1")
        self.assertTrue(any(e.observation and "解析失败" in e.observation for e in session.trace))

    def test_unknown_tool_error_is_fed_back_and_agent_recovers(self) -> None:
        responses = [
            json_action("调用一个不存在的工具", "no_such_tool", {}),
            json_final("改用正确方式回答", "好的，已改正"),
        ]
        agent, llm = make_agent(responses)
        answer = agent.handle_message("s1", "u1", "帮我处理一下")
        self.assertEqual(answer, "好的，已改正")
        session = agent.session_manager.get("s1")
        tool_message = [m for m in session.messages if m.role == "tool"][0]
        self.assertIn("未知工具", tool_message.content)

    def test_llm_call_failure_returns_fallback_without_raising(self) -> None:
        def broken_llm(messages):
            raise ConnectionError("网络不通")

        agent = Agent(tool_registry=build_default_registry(), session_manager=SessionManager(), llm_fn=broken_llm)
        answer = agent.handle_message("s1", "u1", "你好")
        self.assertIn("失败", answer)


class AgentCompressionIntegrationTests(unittest.TestCase):
    def test_long_conversation_gets_compressed(self) -> None:
        n_turns = 8
        responses = [json_final(f"t{i}", f"回复{i}") for i in range(n_turns)]
        agent, llm = make_agent(responses)
        agent._summarize = lambda prev, dropped: f"summary({len(dropped)})"

        original_threshold = context_module.COMPRESS_THRESHOLD
        original_keep = context_module.KEEP_LAST_MESSAGES
        context_module.COMPRESS_THRESHOLD, context_module.KEEP_LAST_MESSAGES = 6, 4
        try:
            for i in range(n_turns):
                agent.handle_message("s1", "u1", f"消息{i}")
        finally:
            context_module.COMPRESS_THRESHOLD = original_threshold
            context_module.KEEP_LAST_MESSAGES = original_keep

        session = agent.session_manager.get("s1")
        self.assertLessEqual(len(session.messages), 6)
        self.assertNotEqual(session.summary, "")


if __name__ == "__main__":
    unittest.main()
