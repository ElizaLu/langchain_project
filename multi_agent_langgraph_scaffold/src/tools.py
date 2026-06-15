from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


def _mock_sentence(tool_name: str, payload: dict[str, Any], result: str) -> str:
    compact = ", ".join(f"{k}={v!r}" for k, v in payload.items()) if payload else "no-args"
    return f"[MOCK TOOL] {tool_name}({compact}) -> {result}"


@tool
def risk_policy_lookup(case_id: str, query: str) -> str:
    """Lookup risk policies or historical risk rules for a case."""
    return _mock_sentence(
        "risk_policy_lookup",
        {"case_id": case_id, "query": query},
        "已命中风险规则库，返回 3 条相关规则摘要。",
    )


@tool
def risk_case_assess(case_id: str, evidence: str) -> str:
    """Assess risk level for a case using evidence."""
    return _mock_sentence(
        "risk_case_assess",
        {"case_id": case_id, "evidence": evidence},
        "风险评分为 0.72，结论为中风险，需要人工复核。",
    )


@tool
def risk_report_generate(case_id: str, evidence: str) -> str:
    """Generate a risk report."""
    return _mock_sentence(
        "risk_report_generate",
        {"case_id": case_id, "evidence": evidence},
        "已生成风险研判简报，包含结论、依据、建议动作。",
    )


@tool
def sensor_timeseries_query(sensor_id: str, time_range: str) -> str:
    """Query sensor timeseries data."""
    return _mock_sentence(
        "sensor_timeseries_query",
        {"sensor_id": sensor_id, "time_range": time_range},
        "已返回传感器时间序列样本，共 240 条记录。",
    )


@tool
def anomaly_check(sensor_id: str, time_range: str) -> str:
    """Run anomaly detection on sensor data."""
    return _mock_sentence(
        "anomaly_check",
        {"sensor_id": sensor_id, "time_range": time_range},
        "检测到 2 处疑似异常峰值，建议进一步确认。",
    )


@tool
def report_compose(topic: str, source_summary: str) -> str:
    """Compose a report from summarized evidence."""
    return _mock_sentence(
        "report_compose",
        {"topic": topic, "source_summary": source_summary},
        "已生成一页式报表摘要，可直接转成 PDF/HTML。",
    )


@tool
def knowledge_retrieve(query: str, kb_name: str = "default_kb") -> str:
    """Retrieve knowledge base passages."""
    return _mock_sentence(
        "knowledge_retrieve",
        {"query": query, "kb_name": kb_name},
        "已检索到 5 段知识片段，相关度较高。",
    )


@tool
def knowledge_answer(query: str, context: str) -> str:
    """Answer a question grounded in retrieved context."""
    return _mock_sentence(
        "knowledge_answer",
        {"query": query, "context": context},
        "已基于知识库上下文给出答案。",
    )


@tool
def workflow_status(workflow_id: str) -> str:
    """Query workflow runtime status."""
    return _mock_sentence(
        "workflow_status",
        {"workflow_id": workflow_id},
        "工作流当前处于运行中，已完成 3/5 个节点。",
    )


@tool
def workflow_history(workflow_id: str) -> str:
    """Query workflow execution history."""
    return _mock_sentence(
        "workflow_history",
        {"workflow_id": workflow_id},
        "已返回最近 10 条工作流事件日志。",
    )


@tool
def workflow_action_suggest(workflow_id: str, issue: str) -> str:
    """Suggest a workflow action such as retry, cancel, or escalate."""
    return _mock_sentence(
        "workflow_action_suggest",
        {"workflow_id": workflow_id, "issue": issue},
        "建议执行重试，并同步通知负责人。",
    )


@tool
def page_open(url: str) -> str:
    """Open a page."""
    return _mock_sentence(
        "page_open",
        {"url": url},
        "页面已打开，DOM 结构已可供后续操作。",
    )


@tool
def page_click(selector: str) -> str:
    """Click an element on a page."""
    return _mock_sentence(
        "page_click",
        {"selector": selector},
        "已完成点击动作，页面状态已更新。",
    )


@tool
def page_fill(selector: str, value: str) -> str:
    """Fill a form field on a page."""
    return _mock_sentence(
        "page_fill",
        {"selector": selector, "value": value},
        "已写入表单值，等待提交。",
    )


@tool
def chat_reply(style: str, message: str) -> str:
    """Generate a friendly chat response."""
    return _mock_sentence(
        "chat_reply",
        {"style": style, "message": message},
        "已生成自然语言回复。",
    )


@tool
def notify_user(channel: str, message: str) -> str:
    """Send a notification to a user-facing channel."""
    return _mock_sentence(
        "notify_user",
        {"channel": channel, "message": message},
        "已通知用户，并记录为消息事件。",
    )
