from __future__ import annotations

import argparse
from pprint import pprint

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from .graph import build_graph
from .llm import build_llm


def run_copilot(user_text: str, thread_id: str = "demo-thread"):
    llm = build_llm()
    graph = build_graph(llm)

    config = {"configurable": {"thread_id": thread_id}}

    stream_input: dict | Command = {
        "messages": [HumanMessage(content=user_text)]
    }

    while True:
        stream = graph.stream_events(stream_input, config=config, version="v3")

        if not stream.interrupted:
            return stream.output

        interrupt_payload = stream.interrupts[0].value

        print("\n=== Agent 建议 ===")
        pprint(interrupt_payload)

        user_reply = input("\n请输入你的自然语言动作：").strip()
        stream_input = Command(resume=user_reply)


def main():
    parser = argparse.ArgumentParser(description="Copilot LangGraph demo")
    parser.add_argument("query", nargs="?", default="帮我查一下今天的传感器异常并生成报表")
    args = parser.parse_args()

    result = run_copilot(args.query)

    print("\n=== FINAL ANSWER ===\n")
    print(result.get("final_answer", ""))

    print("\n=== ROUTE INFO ===\n")
    pprint(
        {
            "route": result.get("route"),
            "route_reason": result.get("route_reason"),
            "route_confidence": result.get("route_confidence"),
            "selected_agent": result.get("selected_agent"),
            "selected_tools": result.get("selected_tools"),
            "suggestions": result.get("suggestions"),
            "recommended_actions": result.get("recommended_actions"),
        }
    )


if __name__ == "__main__":
    main()