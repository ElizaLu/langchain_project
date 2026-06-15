from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


RouteName = Literal[
    "risk_agent",
    "data_report_agent",
    "knowledge_qa_agent",
    "workflow_agent",
    "page_agent",
    "chat_agent",
]


class AppState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    route: RouteName
    route_reason: str
    route_confidence: float

    selected_agent: str
    selected_tools: list[str]
    tool_plan: list[dict[str, Any]]
    tool_trace: list[str]

    final_answer: str
    suggestions: list[str]
    recommended_actions: list[str]
    awaiting_user: bool


@dataclass(frozen=True)
class ToolCallSpec:
    name: str
    args: dict[str, Any]
    reason: str


def last_user_text(state: AppState) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""