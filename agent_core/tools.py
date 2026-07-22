"""工具注册机制。

每个工具包含 名称 / 描述 / 参数 Schema / 具体实现函数四部分。
Agent 会把所有已注册工具的 Schema 塞进系统提示词，LLM 据此自主决定
"要不要调用工具、调用哪个、传什么参数"，Agent 本身不做任何硬编码的意图判断。
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Session


class ToolError(Exception):
    """工具执行失败时抛出。会被 Agent 捕获并转成 Observation 反馈给 LLM，而不是让程序崩溃。"""


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]  # 简化版 JSON Schema: {参数名: {type, description}}
    required: List[str]
    # 工具实现统一签名：(args, session) -> 结果文本；不需要 session 状态的工具可以忽略该参数
    func: Callable[[Dict[str, Any], "Session"], str]

    def to_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


class ToolRegistry:
    """工具注册表：LLM 只能"看到"并调用这里注册过的工具。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def call(self, name: str, args: Dict[str, Any], session: "Session") -> str:
        tool = self.get(name)
        if tool is None:
            raise ToolError(f"未知工具: {name!r}，可用工具: {', '.join(self.names())}")
        try:
            return tool.func(args, session)
        except ToolError:
            raise
        except Exception as exc:  # 工具内部任何异常都归一化为 ToolError，不让单个工具打断主循环
            raise ToolError(f"工具 {name} 执行失败: {exc}") from exc


# ---------------------------------------------------------------------------
# 内置工具 1: calculator —— 只允许 + - * / ** % 和括号，用 ast 求值，避免 eval() 的代码注入风险
# ---------------------------------------------------------------------------

_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_SAFE_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ToolError(f"不支持的表达式片段: {ast.dump(node)}")


def calculator_func(args: Dict[str, Any], session: "Session") -> str:
    expression = str(args.get("expression", ""))
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"表达式不合法: {expression!r} ({exc})") from exc
    return f"{expression} = {result}"


calculator_tool = Tool(
    name="calculator",
    description="计算一个只包含 + - * / ** % 和括号的数学表达式，返回计算结果。",
    parameters={"expression": {"type": "string", "description": "要计算的数学表达式，例如 '(3 + 4) * 2'"}},
    required=["expression"],
    func=calculator_func,
)


# ---------------------------------------------------------------------------
# 内置工具 2: search（mock）—— 用关键词词典模拟搜索引擎，不接真实网络
# ---------------------------------------------------------------------------

_MOCK_SEARCH_DB = {
    "python": "Python 是一种解释型、面向对象的高级编程语言，以简洁易读著称。",
    "react": "ReAct 是一种让大模型交替进行 Thought / Action / Observation 的推理框架，"
    "出自论文《ReAct: Synergizing Reasoning and Acting in Language Models》。",
    "beijing": "北京是中华人民共和国的首都，人口约 2100 万。",
    "北京": "北京是中华人民共和国的首都，人口约 2100 万。",
}


def search_func(args: Dict[str, Any], session: "Session") -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        raise ToolError("search 工具需要非空的 query 参数")
    for keyword, result in _MOCK_SEARCH_DB.items():
        if keyword.lower() in query.lower():
            return result
    return f"没有找到关于「{query}」的相关资料（mock 搜索引擎，只收录了少量关键词）。"


search_tool = Tool(
    name="search",
    description="在一个 mock 搜索引擎里按关键词查资料，适合查百科类/事实类问题。",
    parameters={"query": {"type": "string", "description": "搜索关键词或问题"}},
    required=["query"],
    func=search_func,
)


# ---------------------------------------------------------------------------
# 内置工具 3: weather（mock）
# ---------------------------------------------------------------------------

_MOCK_WEATHER_DB = {
    "beijing": "北京: 25°C, 晴, 湿度 60%",
    "shanghai": "上海: 28°C, 多云, 湿度 70%",
}


def weather_func(args: Dict[str, Any], session: "Session") -> str:
    city = str(args.get("city", "")).strip()
    if not city:
        raise ToolError("weather 工具需要非空的 city 参数")
    return _MOCK_WEATHER_DB.get(city.lower(), f"{city}: 22°C, 多云, 湿度 55%（mock 数据，非真实天气）")


weather_tool = Tool(
    name="weather",
    description="查询指定城市的天气（mock 数据，非真实接口）。",
    parameters={"city": {"type": "string", "description": "城市名，例如 'Beijing'"}},
    required=["city"],
    func=weather_func,
)


# ---------------------------------------------------------------------------
# 内置工具 4: todo —— 有状态工具，状态挂在 session 上，天然随会话隔离
# ---------------------------------------------------------------------------


def todo_func(args: Dict[str, Any], session: "Session") -> str:
    action = str(args.get("action", "list")).lower()
    if action == "add":
        item = str(args.get("item", "")).strip()
        if not item:
            raise ToolError("todo add 操作需要非空的 item 参数")
        session.todos.append(item)
        return f"已添加待办: {item}（当前共 {len(session.todos)} 条）"
    if action == "list":
        if not session.todos:
            return "当前没有待办事项。"
        return "; ".join(f"{i + 1}. {t}" for i, t in enumerate(session.todos))
    if action == "done":
        try:
            index = int(args.get("index", 0)) - 1
        except (TypeError, ValueError) as exc:
            raise ToolError(f"index 参数必须是数字: {args.get('index')!r}") from exc
        if not (0 <= index < len(session.todos)):
            raise ToolError(f"没有第 {args.get('index')} 条待办")
        done_item = session.todos.pop(index)
        return f"已完成并移除待办: {done_item}"
    raise ToolError(f"todo 工具不支持的 action: {action!r}（支持 add/list/done）")


todo_tool = Tool(
    name="todo",
    description="管理当前会话的待办事项列表：action=add 新增一条，list 查看全部，done 完成并移除某一条。",
    parameters={
        "action": {"type": "string", "description": "add / list / done 之一"},
        "item": {"type": "string", "description": "action=add 时要新增的待办内容"},
        "index": {"type": "integer", "description": "action=done 时要完成的待办序号，从 1 开始"},
    },
    required=["action"],
    func=todo_func,
)


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (calculator_tool, search_tool, weather_tool, todo_tool):
        registry.register(tool)
    return registry
