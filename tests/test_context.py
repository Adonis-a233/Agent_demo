import unittest

from agent_core.context import COMPRESS_THRESHOLD, KEEP_LAST_MESSAGES, Session, SessionManager, maybe_compress


class SessionManagerTests(unittest.TestCase):
    def test_sessions_are_isolated(self) -> None:
        manager = SessionManager()
        s1 = manager.get_or_create("s1", "userA")
        s2 = manager.get_or_create("s2", "userA")
        s1.add_message("user", "hello from s1")
        s2.add_message("user", "hello from s2")

        self.assertIsNot(s1, s2)
        self.assertEqual([m.content for m in s1.messages], ["hello from s1"])
        self.assertEqual([m.content for m in s2.messages], ["hello from s2"])

    def test_get_or_create_returns_same_instance(self) -> None:
        manager = SessionManager()
        s1 = manager.get_or_create("s1", "userA")
        s1_again = manager.get_or_create("s1", "userA")
        self.assertIs(s1, s1_again)


class CompressionTests(unittest.TestCase):
    def test_no_compression_below_threshold(self) -> None:
        session = Session(session_id="s1", user_id="u1")
        for i in range(COMPRESS_THRESHOLD):
            session.add_message("user", f"msg-{i}")

        compressed = maybe_compress(session, summarize_fn=lambda prev, dropped: "SHOULD NOT RUN")
        self.assertFalse(compressed)
        self.assertEqual(len(session.messages), COMPRESS_THRESHOLD)
        self.assertEqual(session.summary, "")

    def test_compression_triggers_and_keeps_recent_messages(self) -> None:
        session = Session(session_id="s1", user_id="u1")
        for i in range(COMPRESS_THRESHOLD + 5):
            session.add_message("user", f"msg-{i}")

        calls = []

        def fake_summarize(prev_summary, dropped):
            calls.append(dropped)
            return f"summary-of-{len(dropped)}-messages"

        compressed = maybe_compress(session, summarize_fn=fake_summarize)
        self.assertTrue(compressed)
        self.assertEqual(len(session.messages), KEEP_LAST_MESSAGES)
        self.assertEqual(session.messages[-1].content, f"msg-{COMPRESS_THRESHOLD + 4}")
        self.assertTrue(session.summary.startswith("summary-of-"))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
