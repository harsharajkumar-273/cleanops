import json
import os
from src.env import OpenEnv
from src.reward_shaping import calculate_reward
from src.agent import LLMAgent

def run_evaluation(dataset_name, model_name="GPT-4.1-mini"):
    """
    Runs the benchmark evaluation loop over a selected dataset.
    Loads task descriptors, runs the agent planner inside OpenEnv,
    and returns metrics and step-by-step logs.
    """
    # Load dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    dataset_path = os.path.join(data_dir, f"{dataset_name.lower()}.json")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

    with open(dataset_path, "r") as f:
        tasks = json.load(f)

    env = OpenEnv()
    agent = LLMAgent(model_name=model_name)
    
    results = []
    total_score = 0.0
    
    print(f"\n==================================================")
    print(f"Running CleanOps OpenEnv Evaluation")
    print(f"Dataset: {dataset_name} | Model: {model_name}")
    print(f"==================================================")

    for task in tasks:
        task_id = task["task_id"]
        print(f"\nEvaluating Task: {task_id}")
        print(f"Instruction: {task['instruction']}")
        
        obs = env.reset(task)
        done = False
        step = 0
        rewards = []
        steps_log = []

        while not done:
            step += 1
            action = agent.predict_action(obs)
            obs, done, info = env.step(action)
            
            # Compute dense reward
            step_reward = calculate_reward(env.current_state, env.target_state, action, info)
            rewards.append(step_reward)
            
            log_entry = (
                f"Step {step} | Action: {action.get('action_type')} on {action.get('table')} "
                f"| Reward: {step_reward:.2f} | Status: {info.get('status')}"
            )
            steps_log.append(log_entry)
            print(f"  {log_entry}")
            
            if info.get("error"):
                print(f"  -> Error: {info['error']}")

        # Completion validation
        success = (env.current_state == env.target_state)
        final_score = rewards[-1] if rewards else 0.0
        total_score += final_score

        results.append({
            "task_id": task_id,
            "instruction": task["instruction"],
            "steps": step,
            "rewards": rewards,
            "final_score": final_score,
            "success": success,
            "logs": steps_log
        })

    avg_score = total_score / len(tasks) if tasks else 0.0
    print(f"\n--------------------------------------------------")
    print(f"Evaluation Summary:")
    print(f"Avg Score / Accuracy: {avg_score * 100:.1f}%")
    print(f"==================================================\n")
    
    return {
        "dataset": dataset_name,
        "model": model_name,
        "avg_score": avg_score,
        "tasks": results
    }

if __name__ == "__main__":
    # Test evaluation using default dataset
    run_evaluation("CRM")
