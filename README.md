<div align="center">

# 🤖 CleanOps OpenEnv LLM Benchmark Sandbox

**A transaction-safe evaluation environment & benchmark sandbox for scoring LLM Agent planners.**  
*Features dense step-level partial credit reward shaping, an in-process simulated environment, and interactive telemetry dashboards.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Evaluation_API-009688.svg?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace_Space-Live_Demo-FFD21E.svg?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/harsharajkumar273/cleanops-openenv)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

👉 **[Launch Live HuggingFace Space Demo](https://huggingface.co/spaces/harsharajkumar273/cleanops-openenv)**  
*Test agent task execution and step-level trajectory scoring live in your browser.*

</div>

> **Accuracy note (Nov 2026):** This README previously overstated what the code does. CleanOps runs its simulated environment **in-process** (an in-memory state store in `src/env.py`) — agent actions are **not** executed in Docker containers, and the performance figures originally shown were design targets, not measurements. The text below has been corrected to match the code.

---

> ### 📌 WHAT'S ACTUALLY IMPLEMENTED
> * **Environment**: Gymnasium-style `reset()` / `step()` API over an in-process simulated state store — no container isolation
> * **Reward Shaping Granularity**: **Dense partial-credit scoring** replacing binary pass/fail metrics
> * **Scoring**: deterministic state-diff checks against oracle target states
> * **Performance**: not formally benchmarked (earlier figures in this README were design targets, not measurements, and have been removed)

---

## 💡 The "Why" vs. "How" (Systems Rationale)

* **The Bottleneck (Why standard LLM agent benchmarks fail)**:  
  Evaluating LLM autonomous agents (GPT-4, Claude, Llama 3) on multi-step operational tasks (CRM edits, SQL queries, system administration) using traditional binary pass/fail metrics hides critical agent flaws. A single minor error at step 10 causes a 0 score, providing zero signal on whether the agent was 90% correct or completely lost. Furthermore, allowing agents to execute tools directly on host systems poses severe security risks.
* **The Low-Level Fix (How we solved it)**:  
  CleanOps provides a **Gymnasium-style Environment API (`reset()`, `step()`)** wrapping an in-process simulated environment. Environments are seeded with mock operational data (CRM databases, billing records, file systems). Agent tool calls execute against the simulated state in-process. Progress is measured using **Dense Partial-Credit Reward Shaping**, granting incremental rewards ($R_{step} \in [0, 1]$) for valid intermediate state transitions verified against deterministic target oracle states.

---

## 🛠️ How It Was Achieved (Engineering Deep-Dive)

Three core systems were built:

### 1. Standardized Gymnasium Environment Abstraction (`reset()` / `step()`)
- **RL-First API Contract**: Implements the standard OpenAI Gymnasium interface.
- **State Initialization (`reset()`)**: Restores mock database, filesystem, and billing environment states to baseline seeds.
- **Action Step Execution (`step(action)`)**: Parses agent JSON tool calls, executes them against the simulated environment, and evaluates state diffs.

```python
# CleanOps Gymnasium Environment Step Pipeline
def step(self, action: dict):
    # 1. Execute agent action against the in-process simulated environment
    exec_result = self.env.execute(action["tool"], action["args"])
    
    # 2. Extract current database / filesystem state diff
    current_state = self.state_evaluator.capture_state()
    
    # 3. Compute dense partial-credit reward against oracle target state
    reward, done = self.reward_shaper.evaluate(current_state, self.oracle_target_state)
    
    return current_state, reward, done, {"logs": exec_result.stdout}
```

### 2. Dense Step-Level Partial-Credit Reward Shaping
- **State Diff Subgraph Matching**: Instead of a binary $R \in \{0, 1\}$, CleanOps scores state mutations incrementally ($R = \sum w_i \cdot S_i$).
- **Oracle Verification**: Compares SQL table rows, file structures, and API payloads against deterministic target oracle states, granting partial credit for correct intermediate reasoning steps.

### 3. Environment Isolation — Current State
- **Current state**: Actions execute in-process against a simulated state store (`src/env.py`). There is **no container or process isolation**, so this environment must not be used to run untrusted code. Per-episode Docker containers with resource caps (`--memory=512m`, `--pids-limit=64`) and network isolation are a roadmap item, not an implemented feature.

---

## 🏗️ Evaluation Pipeline & Reward Mechanics Architecture

```mermaid
flowchart TD
    LLM[LLM Agent / Planner] -->|1. Action Step tool_call| API[CleanOps FastAPI Gymnasium API]
    
    subgraph EnvBoundary [In-Process Simulated Environment]
        API -->|2. Execute Action| Env[Simulated Environment]
        Env -->|3. Mutation Output| State[Mutated Environment State DB/FS]
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

## 📊 Benchmarks

No formal benchmark results are published for this project. An earlier version of this README listed throughput and latency figures (steps/sec, P50/P99) that were design targets rather than measured results; they have been removed rather than presented as data.

---

## ⚡ Core Technical Features

1. **Standardized Gymnasium-Style Environment API**:  
   Exposes intuitive `reset()` and `step(action)` methods compatible with standard reinforcement learning and LLM evaluation frameworks.
2. **Dense Step-Level Partial-Credit Reward Shaping**:  
   Calculates step-level fractional rewards based on database state mutations, schema accuracy, and execution efficiency, eliminating binary pass/fail evaluation noise.
3. **Simulated Environment Boundary**:  
   Episodes run against an in-process simulated state store, keeping evaluation deterministic and reproducible. Container isolation is on the roadmap, not implemented.
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
