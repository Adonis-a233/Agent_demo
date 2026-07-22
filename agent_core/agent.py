"""核心 Agent Runtime：不依赖 langgraph / openhands 等任何 Agent 框架，
自己实现"接收输入 -> 判断直接回复/调用工具 -> 调用工具 -> 判断继续循环/返回结果"这一基本循环。

LLM 输出协议（在 system prompt 里约定，Agent 自己解析，不用任何厂商私有的 function-calling 字段）：
    直接回答:   {"thought": "...", "final_answer": "..."}
    调用工具:   {"thought": "...", "action": {"tool": "工具名", "args": {...}}}
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .context import Session, SessionManager, TraceEntry, maybe_compress
from .llm_client import LLMFunc, call_llm
from .tools import ToolError, ToolRegistry

logger = logging.getLogger("agent_core")

DEFAULT_MAX_TOOL_STEPS = 5  # 单轮用户消息内，Thought/Action 循环的最大步数（防止死循环）


class AgentParseError(Exception):
    """LLM 输出无法解析为约定协议时抛出。"""


@dataclass
class ParsedOutput:
    thought: str
    final_answer: Optional[str]
    action: Optional[Dict[str, Any]]


def parse_llm_output(raw: str) -> ParsedOutput:
    """从 LLM 原始回复中提取 思考过程 / 工具调用 / 最终答案。

    做了两层容错：
    1. 允许模型把 JSON 包在 ```json ... ``` 代码块里；
    2. 允许 JSON 前后有多余文字，只截取第一个完整的 JSON 对象。
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    if start == -1:
        raise AgentParseError(f"LLM 输出中没有找到 JSON 对象: {raw!r}")
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise AgentParseError(f"LLM 输出不是合法 JSON: {exc}; 原始输出: {raw!r}") from exc

    if not isinstance(data, dict):
        raise AgentParseError(f"LLM 输出的 JSON 顶层必须是对象: {raw!r}")

    thought = str(data.get("thought", ""))

    if data.get("final_answer") is not None:
        return ParsedOutput(thought=thought, final_answer=str(data["final_answer"]), action=None)

    action = data.get("action")
    if not isinstance(action, dict) or "tool" not in action:
        raise AgentParseError(f"LLM 输出既没有 final_answer 也没有合法 action: {raw!r}")
    return ParsedOutput(thought=thought, final_answer=None, action=action)


class Agent:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        llm_fn: LLMFunc = call_llm,
        max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS,
    ) -> None:
        self.tool_registry = tool_registry
        self.session_manager = session_manager
        self.llm_fn = llm_fn
        self.max_tool_steps = max_tool_steps

    # -- system prompt / context 拼装 -------------------------------------------------

    def _build_system_prompt(self) -> str:
        schema_text = json.dumps(self.tool_registry.schemas(), ensure_ascii=False, indent=2)
        return (
            "你是一个可以调用工具的智能助手。\n"
            "可用工具的 JSON Schema 如下：\n"
            f"{schema_text}\n\n"
            "每一步你必须只输出一个 JSON 对象，不要输出任何解释性文字或代码块标记，二选一：\n"
            '1. 需要调用工具: {"thought": "你的思考", "action": {"tool": "工具名", "args": {"参数名": "参数值"}}}\n'
            '2. 可以直接回答: {"thought": "你的思考", "final_answer": "最终回复用户的内容"}\n'
            "只有当已经拿到足够信息、可以直接回答用户时，才使用 final_answer；"
            "需要计算、查资料、查天气、记待办等操作时，必须先用 action 调用对应工具。"
        )

    def _build_messages(self, session: Session) -> List[Dict[str, str]]:
        """把 session 的历史状态组装成喂给 LLM 的 messages 列表。

        这里塞入 context 的信息包括：
        - 系统提示词（工具 Schema + 输出协议）；
        - 早期历史的摘要（如果发生过压缩）；
        - 最近的原始对话历史：用户输入、Agent 的思考+工具调用、工具执行结果。
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": self._build_system_prompt()}]
        if session.summary:
            messages.append({"role": "system", "content": f"以下是更早对话的摘要，供你参考:\n{session.summary}"})
        for m in session.messages:
            if m.role == "tool":
                # chat 接口通常只认 system/user/assistant，工具结果作为 user 消息喂回，并标注来源
                messages.append({"role": "user", "content": f"[工具 {m.meta.get('tool')} 的执行结果]\n{m.content}"})
            else:
                messages.append({"role": m.role, "content": m.content})
        return messages

    # -- 基础压缩用的摘要函数 -------------------------------------------------

    def _summarize(self, previous_summary: str, dropped_messages) -> str:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in dropped_messages)
        prompt = [
            {"role": "system", "content": "请用不超过 5 句话概括以下对话内容，保留关键事实、结论和用户偏好，用作后续对话的上下文摘要。"},
            {"role": "user", "content": (previous_summary + "\n" if previous_summary else "") + transcript},
        ]
        try:
            return self.llm_fn(prompt)
        except Exception as exc:  # 压缩失败不应该影响主流程，退化为简单截断
            logger.warning("上下文摘要失败，退化为截断: %s", exc)
            fallback = (previous_summary + "\n" if previous_summary else "") + transcript
            return fallback[:800]

    # -- 主循环 -------------------------------------------------

    def handle_message(self, session_id: str, user_id: str, user_input: str) -> str:
        """Step one: 接收用户输入。"""
        session = self.session_manager.get_or_create(session_id, user_id)
        session.turn_count += 1
        session.add_message("user", user_input)

        for step in range(1, self.max_tool_steps + 1):
            messages = self._build_messages(session)

            try:
                raw = self.llm_fn(messages)
            except Exception as exc:  # 网络/接口异常：不崩溃，给用户一个兜底回复
                logger.error("LLM 调用失败 session=%s step=%s: %s", session_id, step, exc)
                answer = "抱歉，调用大模型服务失败，请稍后再试。"
                session.add_message("assistant", answer)
                return answer

            try:
                parsed = parse_llm_output(raw)
            except AgentParseError as exc:
                # Step two 判断失败：格式不对。记录 trace，把原始输出留痕后结束本轮，避免死循环。
                logger.warning("LLM 输出解析失败 session=%s step=%s: %s", session_id, step, exc)
                session.trace.append(TraceEntry(step=step, thought="", action=None, observation=f"解析失败: {exc}"))
                if step >= self.max_tool_steps:
                    answer = "抱歉，我暂时没能生成有效回复，请换个说法再试试。"
                    session.add_message("assistant", answer)
                    return answer
                session.add_message("assistant", f"[格式错误，已丢弃] {raw}")
                continue

            # Step two: 判断是直接回复，还是调用工具
            if parsed.final_answer is not None:
                session.trace.append(TraceEntry(step=step, thought=parsed.thought, action=None, observation=None))
                session.add_message("assistant", parsed.final_answer)
                maybe_compress(session, self._summarize)
                return parsed.final_answer

            # Step three: 调用工具
            action = parsed.action or {}
            tool_name = str(action.get("tool", ""))
            tool_args = action.get("args") or {}
            try:
                observation = self.tool_registry.call(tool_name, tool_args, session)
            except ToolError as exc:
                observation = f"错误: {exc}"

            session.trace.append(TraceEntry(step=step, thought=parsed.thought, action=action, observation=observation))
            logger.info(
                "session=%s step=%s thought=%s action=%s observation=%s",
                session_id, step, parsed.thought, action, observation,
            )
            session.add_message(
                "assistant",
                json.dumps({"thought": parsed.thought, "action": action}, ensure_ascii=False),
                tool=tool_name,
            )
            session.add_message("tool", observation, tool=tool_name)
            # Step four: 根据工具结果判断是继续 loop（下一次 for 迭代），还是返回结果给用户
            # —— 判断逻辑完全交给下一轮 LLM 调用决定，Agent 本身不做业务判断

        # 超过最大步数仍未给出最终答案，返回兜底回复而不是无限循环
        answer = "抱歉，这个问题需要的步骤较多，我暂时无法在限定步数内给出答案。"
        session.add_message("assistant", answer)
        maybe_compress(session, self._summarize)
        return answer
