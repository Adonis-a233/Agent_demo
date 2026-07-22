"""演示入口：模拟"用户 A 同时开了两个窗口"的场景。

窗口 1 (session-weather): 查天气 + 记待办
窗口 2 (session-report):  写周报 + 记待办
两个 session 使用同一个 SessionManager，但历史/待办完全隔离，
并且各自都支持"接着上一轮继续聊"的追问。
"""
import logging
import sys

from agent_core import Agent, SessionManager, build_default_registry

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # 避免 Windows 终端默认编码把中文打印成乱码

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    tool_registry = build_default_registry()
    session_manager = SessionManager()
    agent = Agent(tool_registry=tool_registry, session_manager=session_manager)

    user_id = "user-A"
    window1 = "session-weather"
    window2 = "session-report"

    turns = [
        (window1, "北京今天天气怎么样？"),
        (window1, "帮我记一条待办：带伞"),
        (window2, "帮我算一下 12 * (3 + 5) 等于多少"),
        (window2, "再帮我记一条待办：写周报"),
        (window1, "刚才那条待办是什么来着？"),  # 追问：验证窗口 1 不受窗口 2 影响
    ]

    for session_id, user_input in turns:
        print(f"\n[{session_id}] 用户: {user_input}")
        answer = agent.handle_message(session_id=session_id, user_id=user_id, user_input=user_input)
        print(f"[{session_id}] Agent: {answer}")

    print("\n--- 两个 session 的 todo 状态（验证互不影响）---")
    for session_id in (window1, window2):
        session = session_manager.get(session_id)
        print(f"{session_id}: {session.todos}")


if __name__ == "__main__":
    main()
