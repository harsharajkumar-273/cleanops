# CleanOps OpenEnv 🤖🧪

CleanOps OpenEnv is an evaluation benchmark environment designed to score LLM agent planners on multi-step operational datasets (CRM, Billing, Orders). 

It runs tasks inside transaction-safe database states and evaluates agent actions against target oracle states, generating dense partial-credit feedback signals.

---

## 📂 Project Structure

```
cleanops/
├── README.md               # Documentation
├── requirements.txt        # Package dependencies
├── .env                    # Environment credentials
├── data/                   # JSON task datasets
│   ├── crm.json
│   ├── billing.json
│   └── orders.json
└── src/
    ├── env.py              # OpenEnv execution environment & state tracker
    ├── reward_shaping.py   # Dense step-level reward calculations
    ├── agent.py            # LLM planning agent wrapper
    ├── benchmark.py        # Benchmark evaluator loop
    └── app.py              # FastAPI server backend
```

---

## 🚀 Quick Start

### 1. Install Dependencies
Make sure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Add your credentials to the `.env` file (which is ignored by Git to prevent leaks):
```env
OPENAI_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here
```

### 3. Run Benchmark (CLI Mode)
You can run the evaluator locally to benchmark on a specific dataset:
```bash
python3 -m src.benchmark
```
*Note: If no `OPENAI_API_KEY` is configured in your environment, the agent automatically falls back to a robust Heuristic Rule-Based Parser to prevent script failures.*

### 4. Start FastAPI Server
To launch the backend API server:
```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```
This starts the backend API at `http://127.0.0.1:8000`. You can query `/run-benchmark` or `/datasets` to retrieve live logs and scoring telemetry.

---

## 📊 Dense Reward Shaping Metric
The reward calculation provides intermediate partial-credit rewards during task sweeps:
*   `0.2`: Baseline reward for error-free syntactic transaction.
*   `+0.25`: Targets correct database schema table.
*   `+0.20`: Aligns with correct record IDs/primary keys.
*   `1.00`: Reaches exact target oracle state (Full Task Credit).
