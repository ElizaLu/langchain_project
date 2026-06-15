from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI


DEFAULT_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://192.168.10.59:8000/v1")
DEFAULT_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen3-30b-a3b-instruct")
# DEFAULT_MODEL = os.getenv("LOCAL_LLM_MODEL", "gemma-4-26B-A4B-it")
DEFAULT_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "Empty")


def build_llm(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout: float | None = 120,
    max_retries: int = 2,
    stream_usage: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
) -> ChatOpenAI:
    params: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "base_url": base_url or DEFAULT_BASE_URL,
        "api_key": api_key or DEFAULT_API_KEY,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
        "stream_usage": stream_usage,
    }

    if extra_kwargs:
        params.update(extra_kwargs)

    return ChatOpenAI(**params)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """
    单例版模型。
    整个项目只用一套本地模型
    """
    return build_llm()


def build_router_llm() -> ChatOpenAI:
    """
    路由器温度固定为 0，保证分类稳定。
    """
    return build_llm(temperature=0.0)


def build_agent_llm() -> ChatOpenAI:
    """
    专家 Agent 的主模型。
    """
    return build_llm(temperature=0.0)


def build_review_llm() -> ChatOpenAI:
    """
    负责总结结果、生成下一步建议的模型。
    """
    return build_llm(temperature=0.0)