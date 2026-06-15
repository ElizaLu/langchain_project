from __future__ import annotations

from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from .agents import build_specialist_agent
from .router import build_router_node
from .state import AppState
from . import tools


def build_tool_registry() -> dict[str, object]:
    return {
        "risk_policy_lookup": tools.risk_policy_lookup,
        "risk_case_assess": tools.risk_case_assess,
        "risk_report_generate": tools.risk_report_generate,
        "sensor_timeseries_query": tools.sensor_timeseries_query,
        "anomaly_check": tools.anomaly_check,
        "report_compose": tools.report_compose,
        "knowledge_retrieve": tools.knowledge_retrieve,
        "knowledge_answer": tools.knowledge_answer,
        "workflow_status": tools.workflow_status,
        "workflow_history": tools.workflow_history,
        "workflow_action_suggest": tools.workflow_action_suggest,
        "page_open": tools.page_open,
        "page_click": tools.page_click,
        "page_fill": tools.page_fill,
        "chat_reply": tools.chat_reply,
        "notify_user": tools.notify_user,
    }


def build_graph(llm):
    tool_registry = build_tool_registry()

    router_node = build_router_node(llm)

    agents = {
        "risk_agent": build_specialist_agent(
            name="risk_agent",
            description="风险研判 Agent",
            tool_names=["risk_policy_lookup", "risk_case_assess", "risk_report_generate"],
            tool_registry=tool_registry,
            llm=llm,
        ),
        "data_report_agent": build_specialist_agent(
            name="data_report_agent",
            description="数据分析与报表 Agent",
            tool_names=["sensor_timeseries_query", "anomaly_check", "report_compose"],
            tool_registry=tool_registry,
            llm=llm,
        ),
        "knowledge_qa_agent": build_specialist_agent(
            name="knowledge_qa_agent",
            description="知识问答 Agent",
            tool_names=["knowledge_retrieve", "knowledge_answer"],
            tool_registry=tool_registry,
            llm=llm,
        ),
        "workflow_agent": build_specialist_agent(
            name="workflow_agent",
            description="工作流查询 Agent",
            tool_names=["workflow_status", "workflow_history", "workflow_action_suggest"],
            tool_registry=tool_registry,
            llm=llm,
        ),
        "page_agent": build_specialist_agent(
            name="page_agent",
            description="页面操作 Agent",
            tool_names=["page_open", "page_click", "page_fill"],
            tool_registry=tool_registry,
            llm=llm,
        ),
        "chat_agent": build_specialist_agent(
            name="chat_agent",
            description="普通聊天 Agent",
            tool_names=["chat_reply", "notify_user"],
            tool_registry=tool_registry,
            llm=llm,
        ),
    }

    builder = StateGraph(AppState)
    builder.add_node("router", router_node)
    builder.add_node("risk_agent", agents["risk_agent"])
    builder.add_node("data_report_agent", agents["data_report_agent"])
    builder.add_node("knowledge_qa_agent", agents["knowledge_qa_agent"])
    builder.add_node("workflow_agent", agents["workflow_agent"])
    builder.add_node("page_agent", agents["page_agent"])
    builder.add_node("chat_agent", agents["chat_agent"])

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "risk_agent": "risk_agent",
            "data_report_agent": "data_report_agent",
            "knowledge_qa_agent": "knowledge_qa_agent",
            "workflow_agent": "workflow_agent",
            "page_agent": "page_agent",
            "chat_agent": "chat_agent",
        },
    )

    # 开发阶段用内存检查点；生产建议换持久化 checkpointer
    return builder.compile(checkpointer=InMemorySaver())