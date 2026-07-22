"""测试专用的假 LLM，避免真正发网络请求，让测试可控、确定、可离线运行。"""
import json
from typing import Any, Dict, List


class ScriptedLLM:
    """按预设脚本依次返回固定回复，同时记录每次收到的 messages，方便断言 context 内容。"""

    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls: List[List[Dict[str, str]]] = []

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("ScriptedLLM 的预设回复已经用完，测试脚本和实际调用次数不匹配")
        return self.responses.pop(0)


def json_final(thought: str, answer: str) -> str:
    return json.dumps({"thought": thought, "final_answer": answer}, ensure_ascii=False)


def json_action(thought: str, tool: str, args: Dict[str, Any]) -> str:
    return json.dumps({"thought": thought, "action": {"tool": tool, "args": args}}, ensure_ascii=False)
