# Multi-agent LangGraph Scaffold

这是一个**可直接跑通的多智能体编排骨架**，对应以下角色：

- 用户 query 语义解析 / Supervisor
- 风险研判 Agent
- 数据分析与报表 Agent
- 知识问答 Agent
- 工作流查询 Agent
- 页面操作 Agent
- 普通聊天 Agent

目前所有外部 tool 都是 **stub**，返回一句话字符串，目的是先把：

1. 路由逻辑
2. 代理选择逻辑
3. tool 调用顺序
4. 最终汇总输出

先完整跑通。

## 运行

```bash
pip install -e .
multi-agent-demo "帮我查一下这个设备今天的传感器异常并生成报表"
```

## 后续替换的位置

- `tools.py`：把 mock tool 换成真实 API / DB / 页面操作
- `router.py`：把关键词路由换成 LLM 结构化分类
- `agents.py`：把规则型 planner 换成 LLM planner / ReAct agent

## 说明

这里使用了 LangGraph 的 `StateGraph` 作为主编排器；`StateGraph` 是主要图抽象，构建完后需要 `compile()` 才能执行。工具定义使用 LangChain 的 `@tool` 装饰器。
