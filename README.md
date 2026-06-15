# TranAD-lumh + LangGraph Scaffold

A research-oriented fork of **TranAD** for **multivariate gas anomaly detection** in industrial sensor streams, now extended with a **LangGraph-based agent scaffold** for future tool-oriented orchestration.

This repository preserves the core TranAD architecture for gas monitoring and anomaly detection, while adding a LangGraph workflow layer that will eventually wrap the gas detection pipeline as a callable tool inside a multi-agent system.

---

## Overview

This project combines two parts:

1. **Gas anomaly detection core**
   - TranAD-based multivariate time-series anomaly detection
   - Custom preprocessing for gas sensor data
   - Sliding-window reconstruction modeling
   - Score calibration and POT-based thresholding
   - Visualization and diagnosis utilities

2. **LangGraph scaffold**
   - Router-based agent orchestration
   - Specialist agent execution
   - Tool calling loop
   - Interrupt/resume control flow
   - Designed to integrate the gas anomaly detector as a tool node in later development

In short:

> **TranAD-lumh** provides the anomaly detection engine.  
> **LangGraph scaffold** provides the agentic control layer that will eventually call that engine as a tool.

---

## Why this fork

Compared with the original TranAD repository, this version is tailored to a gas-monitoring workflow and future agent integration:

- Custom preprocessing for raw gas sensor text data
- Window-based multivariate time-series modeling
- Transformer-based reconstruction and anomaly scoring
- Score transformation for more stable thresholding
- POT/SPOT-based threshold estimation
- Per-dimension inspection and visualization
- Training / testing / plotting integrated in one pipeline
- LangGraph scaffold for routing, tool use, and interrupt/resume control
- Future plan: expose gas anomaly detection as a LangGraph tool

---

## Main idea

The current gas anomaly detection workflow is:

1. Load raw gas sensor data
2. Preprocess and split it into `train / test / labels`
3. Convert sequences into sliding windows
4. Train TranAD on normal patterns
5. Compute reconstruction error as anomaly score
6. Calibrate scores and estimate thresholds with POT
7. Visualize predictions, scores, and labels

The future agentic workflow will be:

1. Receive user input
2. Route the request to a specialist agent
3. Decide whether a tool call is needed
4. Call the gas anomaly detection tool when appropriate
5. Return the result to the user
6. Ask whether to continue, switch agent, or stop

---

## LangGraph workflow

The LangGraph scaffold in this repository follows the flow below:

```mermaid
flowchart TD

    A[用户输入] --> B[Router Node<br/>选择专家模型]

    B --> C[Specialist Agent]

    C --> D{选择工具<br/>Call Model}

    D -- 是 --> E[执行工具<br/>ToolNode]
    E --> C

    D -- 否 --> F[结果总结<br/>Finalize]

    F --> G[Ask User<br/>是否继续?]

    G --> H["interrupt()"]

    H --> I{用户选择}

    I --> J["Command(resume)"]
    J -- 继续当前agent --> C

    J-- 重选agent --> B

    subgraph LangGraph
        B
        C
        E
        F
        G
        H
    end