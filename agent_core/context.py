"""Session / Context 管理。

解决三件事：
1. 多窗口互不干扰 —— 每个 session_id 拥有独立的消息历史和工具状态（如 todo 列表）；
2. 多轮追问 —— 完整历史会作为 messages 传给 LLM，天然支持"纯聊天追问"和"带工具的追问"；
3. context 过长时的基础压缩 —— 超过阈值后，把较早的历史总结成一段摘要文本，
   替换掉原始消息，避免 messages 无限增长。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class TraceEntry:
    """记录一次 Agent 内部循环 step 的执行轨迹，用于调试/审计（要求里的"执行日志"）。"""

    step: int
    thought: str
    action: Optional[Dict[str, Any]]
    observation: Optional[str]
    ts: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    summary: str = ""  # 被压缩掉的历史消息的摘要，会作为额外的 system 消息喂给 LLM
    todos: List[str] = field(default_factory=list)  # todo 工具的状态，随 session 隔离，互不影响
    trace: List[TraceEntry] = field(default_factory=list)
    turn_count: int = 0  # 用户发了多少轮消息
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, **meta: Any) -> Message:
        message = Message(role=role, content=content, meta=meta)
        self.messages.append(message)
        self.last_active_at = time.time()
        return message


class SessionManager:
    """维护 session_id -> Session 的映射，是"多个窗口各自独立会话"的核心。"""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def get_or_create(self, session_id: str, user_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id, user_id=user_id)
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def all_sessions(self) -> List[Session]:
        return list(self._sessions.values())


# ---------------------------------------------------------------------------
# Context 压缩（基础版：不做语义分块/向量检索，只做"超阈值就摘要"）
# ---------------------------------------------------------------------------

KEEP_LAST_MESSAGES = 12  # 触发压缩后，仍以原文保留的最近消息条数
COMPRESS_THRESHOLD = 20  # 消息数超过这个阈值就触发一次压缩

SummarizeFunc = Callable[[str, List[Message]], str]


def maybe_compress(session: Session, summarize_fn: SummarizeFunc) -> bool:
    """基础压缩策略：

    消息数超过 COMPRESS_THRESHOLD 时，把"最近 KEEP_LAST_MESSAGES 条之前"的历史
    通过 summarize_fn 总结成一段文本，累加进 session.summary，
    并把这部分原始消息从 messages 里丢弃，只保留最近的原文。

    返回是否发生了压缩，方便测试断言。
    """
    if len(session.messages) <= COMPRESS_THRESHOLD:
        return False
    to_drop = session.messages[:-KEEP_LAST_MESSAGES]
    session.messages = session.messages[-KEEP_LAST_MESSAGES:]
    session.summary = summarize_fn(session.summary, to_drop)
    return True
