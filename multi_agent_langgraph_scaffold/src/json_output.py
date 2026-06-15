from __future__ import annotations

import json
import re
from typing import TypeVar, Type

from pydantic import BaseModel, ValidationError
from langchain_core.messages import BaseMessage


T = TypeVar("T", bound=BaseModel)


def extract_text(content: object) -> str:
    """
    兼容 AIMessage.content 可能是 str 或 content blocks 的情况。
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()

    return str(content).strip()


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in: {text!r}")
    return text[start:end + 1]


def parse_model_json(message: BaseMessage, schema):
    text = extract_text(getattr(message, "content", ""))
    json_text = extract_json_object(text)
    data = json.loads(json_text)
    return schema.model_validate(data)


def parse_model_json_with_retry(
    llm,
    schema: Type[T],
    messages: list[dict | BaseMessage],
    *,
    retry_prompt: str,
    max_retries: int = 1,
) -> T:
    last_err: Exception | None = None
    cur_messages = list(messages)

    for _ in range(max_retries + 1):
        resp = llm.invoke(cur_messages)

        print("\n===================")
        print("RAW MODEL OUTPUT")
        print("===================")
        print(resp.content)
        print("===================\n")

        try:
            return parse_model_json(resp, schema)
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            cur_messages = [
                *cur_messages,
                {
                    "role": "user",
                    "content": (
                        f"{retry_prompt}\n\n"
                        f"上一次输出如下：\n{extract_text(getattr(resp, 'content', ''))}\n\n"
                        "请重新输出，且只能输出合法 JSON。"
                    ),
                },
            ]

    raise last_err or RuntimeError("Failed to parse structured output.")