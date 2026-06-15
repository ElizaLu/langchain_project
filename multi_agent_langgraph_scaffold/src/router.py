from __future__ import annotations

from pydantic import BaseModel, Field

from .state import AppState, RouteName, last_user_text
from .json_output import parse_model_json_with_retry


class RouteDecision(BaseModel):
    route: RouteName = Field(description="要路由到的专家 Agent")
    reason: str = Field(description="路由原因")
    confidence: float = Field(description="0~1 置信度", ge=0.0, le=1.0)


def build_router_node(llm):
    def router_node(state: AppState) -> dict:
        user_text = last_user_text(state)

        raw = parse_model_json_with_retry(
            llm,
            RouteDecision,
            [
                {
                    "role": "system",
                    "content": (
                        "你是一个多专家路由器。"
                        "请在以下路由中选择一个："
                        "risk_agent, data_report_agent, knowledge_qa_agent, "
                        "workflow_agent, page_agent, chat_agent。"
                        "优先根据用户真实意图分类，不要只看关键词。"
                        "只输出严格合法的 JSON，不要 markdown，不要解释，不要多余文本。"
                        "JSON 必须且只包含 route, reason, confidence 三个字段。"
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            retry_prompt=(
                "你的输出不是合法的 JSON，或者不符合 schema。"
                "请严格按要求输出。"
            ),
            max_retries=1,
        )

        return {
            "route": raw.route,
            "route_reason": raw.reason,
            "route_confidence": raw.confidence,
        }

    return router_node