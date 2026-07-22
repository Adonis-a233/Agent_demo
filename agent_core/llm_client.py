"""最基础的 LLM 调用封装。

只负责"发一个 chat 请求、拿到文本回复"这一件事，不包含任何 Agent 编排逻辑，
也不依赖 openai/langchain 等 SDK —— 直接用 requests 打 HTTP 请求，
这样"调用大模型"这一步对上层完全透明、可替换（测试时可以换成假实现）。
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ["AGENT_MODEL_NAME"]
API_KEY = os.environ["AGENT_API_KEY"]
BASE_URL = os.environ.get("AGENT_BASE_URL", "https://api.openai.com/v1").rstrip("/")

# Agent 内部对"LLM 调用"的统一签名：传入 OpenAI 风格的 messages 列表，返回回复文本。
# Agent 只依赖这个签名，不依赖具体实现，因此测试时可以传入任意假函数替换掉真实网络调用。
LLMFunc = Callable[[List[Dict[str, str]]], str]


def call_llm(messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 800) -> str:
    """向 OpenAI 兼容的 /chat/completions 接口发起一次同步请求，返回回复文本。"""
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
