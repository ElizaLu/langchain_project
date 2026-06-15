from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from .state import AppState
from .json_output import parse_model_json_with_retry


def build_tool_catalog(tool_names: list[str], tool_registry: dict[str, Any]) -> str:
    lines = []
    for name in tool_names:
        tool_obj = tool_registry[name]
        desc = getattr(tool_obj, "description", "") or "无描述"
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


class AgentReview(BaseModel):
    summary: str = Field(description="对本次执行结果的简洁总结")
    suggestions: list[str] = Field(default_factory=list, description="下一步建议")
    recommended_actions: list[str] = Field(
        default_factory=list, description="可直接执行的推荐动作"
    )


class HandoffDecision(BaseModel):
    action: Literal["stay", "handoff_router"] = Field(
        description="stay=继续当前 agent；handoff_router=切换任务，返回 router"
    )


def build_specialist_agent(
    *,
    name: str,
    description: str,
    tool_names: list[str],
    tool_registry: dict[str, Any],
    llm,
):
    tools = [tool_registry[t] for t in tool_names]
    tool_catalog = build_tool_catalog(tool_names, tool_registry)

    # 1) 主模型：负责 tool calling
    llm_with_tools = llm.bind_tools(tools)

    # 2) ToolNode：LangGraph 官方提供的工具执行节点
    tool_node = ToolNode(tools)


    def call_model(state: AppState) -> dict:
        system = SystemMessage(
            content=(
                f"你是 {name}，职责是：{description}。"
                "先根据当前任务决定是否调用工具。"
                "如果需要工具，请发起 tool calls。"
                "如果工具结果已经足够，就直接给出最终回复。"
                "当前是 copilot 模式：你需要在最终回复后给出下一步建议。"
            )
        )

        response = llm_with_tools.invoke([system, *state.get("messages", [])])
        return {
            "selected_agent": name,
            "messages": [response],
        }

    def should_continue(state: AppState) -> Literal["tools", "finalize"]:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        return "tools" if tool_calls else "finalize"

    def finalize(state: AppState) -> dict:

        review = parse_model_json_with_retry(
            llm,
            AgentReview,
            [
                SystemMessage(
                    content=(
                        f"你是 {name} 的结果总结器。\n"
                        f"当前 agent 职责：{description}\n\n"
                        f"当前可用工具只有以下这些：\n{tool_catalog}\n\n"
                        "生成建议时必须遵守：\n"
                        "1. suggestions 和 recommended_actions 只能基于以上工具的能力。\n"
                        "2. 不允许编造工具不存在的能力。\n"
                        "3. 如果当前问题超出这些工具范围，suggestions 和 recommended_actions 可以返回空数组。\n"
                        "4. summary 只能总结已经发生的对话和工具结果，不能扩展到工具范围外。\n"
                        "5. 只输出严格合法 JSON。"
                    )
                ),
                *state["messages"],
            ],
            retry_prompt=(
                "你返回的 JSON 不符合要求，或者建议没有严格基于当前工具范围。"
                "请重新输出合法 JSON，且 suggestions / recommended_actions 只能来自当前工具能力。"
            ),
            max_retries=1,
        )

        tool_names_used = []
        tool_trace = []
        for msg in state["messages"]:
            if getattr(msg, "type", None) == "tool":
                tool_names_used.append(getattr(msg, "name", "unknown_tool"))
                tool_trace.append(f"[TOOL] {getattr(msg, 'name', 'unknown_tool')} | {getattr(msg, 'content', '')}")

        return {
            "selected_tools": tool_names_used,
            "tool_trace": tool_trace,
            "final_answer": review.summary,
            "suggestions": review.suggestions,
            "recommended_actions": review.recommended_actions,
            "messages": [AIMessage(content=review.summary)],
        }

    def ask_user(state: AppState) -> Command:
        # interrupt 的 payload 必须是 JSON-serializable
        payload = {
            "agent": name,
            "selected_tools": state.get("selected_tools", []),
            "tool_trace": state.get("tool_trace", []),
            "summary": state.get("final_answer", ""),
            "suggestions": state.get("suggestions", []),
            "recommended_actions": state.get("recommended_actions", []),
        }

        # 暂停，等外部输入
        user_text = interrupt(payload) # resume接收到的结果

        # resume 后，这一节点会从头重跑，interrupt 这里会拿到用户输入
        user_text = str(user_text).strip()

        # 用 LLM 判断：继续当前 agent，还是切回 router
        decision = parse_model_json_with_retry(
            llm,
            HandoffDecision,
            [
                SystemMessage(
                    content=(
                        "你是任务边界判断器。\n"
                        "判断用户回复是否仍然属于当前专家 agent 的工具范围。\n\n"
                        f"当前专家：{name}\n"
                        f"当前专家职责：{description}\n"
                        f"当前可用工具：{tool_catalog}\n\n"
                        "规则：\n"
                        "1. 如果用户仍在要求这些工具能完成的任务，action=stay。\n"
                        "2. 如果用户明显提出了其它领域的问题，action=handoff_router。\n"
                        "3. 不要猜测，不要扩展工具能力。\n"
                        "4. 只输出 JSON。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"当前专家：{name}\n"
                        f"当前建议：{payload}\n"
                        f"用户回复：{user_text}"
                    )
                ),
            ],
            retry_prompt=(
                "请重新输出合法 JSON，且 action 只能是 stay 或 handoff_router。"
            ),
            max_retries=1,
        )

        update = {
            "messages": [HumanMessage(content=user_text)],
            "awaiting_user": False,
        }

        if decision.action == "handoff_router":
            # 跳回母图的 router，重新做全局路由
            return Command(
                update=update,
                goto="router",
                graph=Command.PARENT,
            )

        # 留在当前 agent 内，继续 call_model -> tools -> call_model
        return Command(
            update=update,
            goto="call_model",
        )

    builder = StateGraph(AppState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_node("finalize", finalize)
    builder.add_node("ask_user", ask_user)

    builder.add_edge(START, "call_model")

    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )

    builder.add_edge("tools", "call_model")
    builder.add_edge("finalize", "ask_user")

    # ask_user通过 Command 动态决定回到 call_model，还是回到母图 router
    return builder.compile()