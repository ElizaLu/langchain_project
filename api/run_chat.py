from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

import json
from pathlib import Path
import time

from app.tools.gas_status_tool import get_latest_gas_status

LATEST_FILE = Path("/home/sente/langchain_project/TranAD-lumh/data/predictions/_latest_prediction.json")


def build_agent():
    llm = ChatOpenAI(
        base_url="http://192.168.10.59:8000/v1",
        api_key="Empty",
        # model="gemma-4-26B-A4B-it",
        model="qwen3-30b-a3b-instruct",
        temperature=0,
    )

    tools = [get_latest_gas_status]

    system_prompt = (
        "你是一个燃气在线监测助手。"
        "当用户询问“现在是否异常”“最新状态”“当前燃气有没有异常”这类问题时，"
        "必须调用工具 get_latest_gas_status 来获取最新推理结果。"
        "如果用户问的是别的普通问题，你可以直接回答。"
        "回答要简洁、明确。"
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


def main():
    agent = build_agent()
    print("燃气在线异常查询助手已启动，输入 exit 退出。")

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("已退出。")
            break

        start = time.time()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )
        end = time.time()

        # create_agent 的结果是带 messages 的 state，最后一条通常是 AI 回复
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            print("\n助手：")
            print(last_msg.content if hasattr(last_msg, "content") else str(last_msg))
            print(f"\n耗时: {end - start:.2f} 秒")
        else:
            print("\n助手：没有返回内容")
            print(f"\n耗时: {end - start:.2f} 秒")


if __name__ == "__main__":
    main()
