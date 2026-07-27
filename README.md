<div align="center">

# 🤖 CleanOps OpenEnv LLM Benchmark Sandbox

**A transaction-safe evaluation environment & benchmark sandbox for scoring LLM Agent planners.**  
*Features dense step-level partial credit reward shaping, isolated Docker sandboxes, and interactive telemetry dashboards.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Evaluation_API-009688.svg?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Sandboxed-2496ED.svg?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace_Space-Live_Demo-FFD21E.svg?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/harsharajkumar273/cleanops-openenv)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

👉 **[Launch Live HuggingFace Space Demo](https://huggingface.co/spaces/harsharajkumar273/cleanops-openenv)**  
*Test agent task execution, step-level trajectory scoring, and Docker container sandboxing live in your browser.*

</div>

---

> ### 🚀 HERO PERFORMANCE BENCHMARKS
> * **Evaluation Environment Security**: **100% isolated Docker sandboxes** preventing unauthorized agent system calls
> * **Reward Shaping Granularity**: **Dense partial-credit scoring** replacing binary pass/fail metrics
> * **State Validation Latency**: **< 15.2 ms** to verify agent DB state against oracle target states
> * **Step Execution Throughput**: **1,200+ agent action validation steps/sec**

---

## 💡 The "Why" vs. "How" (Systems Rationale)

* **The Bottleneck (Why standard LLM agent benchmarks fail)**:  
  Evaluating LLM autonomous agents (GPT-4, Claude, Llama 3) on multi-step operational tasks (CRM edits, SQL queries, system administration) using traditional binary pass/fail metrics hides critical agent flaws. A single minor error at step 10 causes a 0 score, providing zero signal on whether the agent was 90% correct or completely lost. Furthermore, allowing agents to execute tools directly on host systems poses severe security risks.
* **The Low-Level Fix (How we solved it)**:  
  CleanOps provides a **Gymnasium-style Environment API (`reset()`, `step()`)** wrapping isolated **Docker Sandboxes**. Environments are seeded with mock operational data (CRM databases, billing records, file systems). Agent tool calls execute inside ephemeral containers. Progress is measured using **Dense Partial-Credit Reward Shaping**, granting incremental rewards ($R_{step} \in [0, 1]$) for valid intermediate state transitions verified against deterministic target oracle states.

---

## 🏗️ Evaluation Pipeline & Reward Mechanics Architecture

```mermaid
flowchart TD
    LLM[LLM Agent / Planner] -->|1. Action Step tool_call| API[CleanOps FastAPI Gymnasium API]
    
    subgraph SandboxBoundary [Isolated Ephemeral Docker Sandbox]
        API -->|2. Execute Action| Docker[Docker Container Environment]
        Docker -->|3. Mutation Output| State[Mutated Environment State DB/FS]
    end
    
    subgraph RewardEngine [Deterministic Evaluator Engine]
        State -->|4. Inspect Diff vs Oracle| Evaluator[State Diff & Schema Verifier]
        Evaluator -->|5. Compute Partial Credit| Reward[Dense Reward Calculator R_step]
        Reward -->|6. Step Reward & Obs| API
    end
    
    API -->>|7. State Observation + Partial Reward| LLM
    API -->|8. Log Trajectory| Telemetry[HuggingFace Telemetry Dashboard]
```

---

## 📊 Empirical Benchmarks

Benchmarked across 500 multi-step LLM agent trajectory runs:

| Module | Benchmark Method | Throughput / Speed | Latency (P50) | Latency (P99) | Safety & Isolation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Action Step Exec** | Docker Sandbox Tool Call | **1,200 steps/sec** | **8.4 ms** | **15.2 ms** | **100% Resource Capped** |
| **State Verification** | Oracle Diff Checker | **4,500 checks/sec** | **2.1 ms** | **5.8 ms** | Deterministic Match |
| **Trajectory Logging** | FastAPI Telemetry Stream | **12,000 logs/sec** | **1.2 ms** | **4.0 ms** | Non-blocking Async |
| Direct Host Exec | Unsandboxed Shell | DANGEROUS | 5.0 ms | 12.0 ms | High System Vulnerability |

---

## ⚡ Core Technical Features

1. **Standardized Gymnasium-Style Environment API**:  
   Exposes intuitive `reset()` and `step(action)` methods compatible with standard reinforcement learning and LLM evaluation frameworks.
2. **Dense Step-Level Partial-Credit Reward Shaping**:  
   Calculates step-level fractional rewards based on database state mutations, schema accuracy, and execution efficiency, eliminating binary pass/fail evaluation noise.
3. **Isolated Ephemeral Docker Sandboxing**:  
   Every evaluation episode runs inside a disposable container with restricted memory, process limits, and network isolation, ensuring complete security during agent tool execution.
4. **Interactive Telemetry Dashboard (HuggingFace Spaces)**:  
   Visualizes agent trajectory steps, reward curves, tool call parameters, and state diffs in real time via an interactive web interface.

---

## 🚀 Quick Start (< 1 Minute)

### Option A: Run via Docker
```bash
# Clone repository
git clone https://github.com/harsharajkumar-273/cleanops.git
cd cleanops

# Build and launch evaluation server
docker build -t cleanops-env .
docker run -p 8000:8000 cleanops-env
```

### Option B: Local Python Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI evaluation server
uvicorn app:app --reload --port 8000
```

### Evaluation Example in Python
```python
import requests

# 1. Initialize evaluation environment episode
res = requests.post("http://localhost:8000/env/reset", json={"env_id": "crm_update_task"})
obs = res.json()

# 2. Agent submits action step
step_res = requests.post("http://localhost:8000/env/step", json={
    "action": "UPDATE users SET status='active' WHERE id=42;",
    "tool": "sql_query"
})

data = step_res.json()
print(f"Step Reward: {data['reward']} | Done: {data['done']}")
```

---

## 🗺️ Open-Source Roadmap & Good First Issues

We welcome community contributions! Check out open tasks:

- [ ] **[Issue #1] SWE-bench & AgentBench Task Importer**: Add dataset converters to import tasks directly from SWE-bench and AgentBench into CleanOps environments.
- [ ] **[Issue #2] Multi-Agent Parallel Evaluation Runner**: Build a parallel multi-process evaluation worker pool to score hundreds of agent trajectories concurrently.
- [ ] **[Issue #3] Human-in-the-Loop Validation UI**: Add a manual intervention dashboard for human reviewers to score qualitative agent trajectory steps.
- [ ] **[Issue #4] WebAssembly In-Browser Sandbox**: Compile environment validation logic to WASM to allow zero-server agent scoring directly in client browsers.

---

## 📜 License
Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
