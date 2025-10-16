"""LLM客户端。"""

from __future__ import annotations

import os
from typing import Any, Dict

from openai import OpenAI


def call_llm(
    messages: list[Dict[str, str]],
    model: str = "o3",
    temperature: float = 0.0,
    response_format: Dict[str, str] | None = None,
) -> str:
    """调用OpenAI LLM并返回响应内容。

    Args:
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
        model: 模型名称
        temperature: 温度参数
        response_format: 响应格式，如 {"type": "json_object"}

    Environment Variables:
        OPENAI_API_KEY: OpenAI API密钥（必需）
        OPENAI_BASE_URL: 自定义API端点URL（可选）
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if response_format:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


__all__ = ["call_llm"]
