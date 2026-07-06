import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Append projects/cleanops to sys.path so we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark import run_evaluation

app = FastAPI(
    title="CleanOps OpenEnv API Server",
    description="Backend API serving LLM Agent task benchmarks with reward shaping checks.",
    version="1.0.0"
)

# Enable CORS for frontend visualizers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    dataset: str
    model: str = "GPT-4.1-mini"

@app.get("/")
def get_root():
    return {
        "status": "online",
        "description": "CleanOps OpenEnv Benchmarking Server active.",
        "supported_datasets": ["CRM", "Billing", "Orders"],
        "supported_models": ["GPT-4.1-mini", "Qwen-72B"]
    }

@app.post("/run-benchmark")
def execute_benchmark(request: EvaluationRequest):
    dataset = request.dataset
    model = request.model
    
    if dataset not in ["CRM", "Billing", "Orders"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported dataset: {dataset}. Choose from ['CRM', 'Billing', 'Orders']."
        )
        
    try:
        results = run_evaluation(dataset, model_name=model)
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")

@app.get("/datasets")
def get_datasets():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    datasets = {}
    
    for name in ["crm", "billing", "orders"]:
        path = os.path.join(data_dir, f"{name}.json")
        if os.path.exists(path):
            import json
            with open(path, "r") as f:
                datasets[name.upper()] = json.load(f)
                
    return datasets

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
