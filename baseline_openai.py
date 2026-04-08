import os
import json
import argparse
import random
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Local imports
from environment.inbox_env import InboxEnv
from models import EmailAction

load_dotenv()

SYSTEM_PROMPT = """
You are an expert email triage assistant. Given an email, you must respond with ONLY a valid JSON object with these exact keys:
{
  "intents": ["list", "of", "detected", "intents"],
  "priority": "low|medium|high|critical",
  "action": "reply|ignore|escalate|forward",
  "response": "your drafted reply here"
}
Detect intents from: [meeting_request, complaint, question, urgent, follow_up, spam, feedback, approval_needed]
No explanation. No markdown. Only raw JSON.
"""

def get_random_action() -> Dict[str, Any]:
    """Fallback random action generator."""
    return {
        "intents": [random.choice(["meeting_request", "question", "follow_up", "urgent"])],
        "priority": random.choice(["low", "medium", "high", "critical"]),
        "action": random.choice(["reply", "ignore", "escalate", "forward"]),
        "response": "Thank you for your message. We have received it and will process it accordingly."
    }

def run_evaluation(model: str, episodes_per_diff: int, difficulty: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("API_BASE_URL")  # e.g. https://api.groq.com/openai/v1

    if not api_key:
        print("WARNING: OPENAI_API_KEY not found in environment. Baseline will fail unless mocked.")

    # Supports OpenAI, Groq, or any OpenAI-compatible endpoint via API_BASE_URL
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
        print(f"Using custom API base: {base_url}")

    client = OpenAI(**client_kwargs)

    all_results = []
    diffs_to_run = ["easy", "medium", "hard"] if difficulty == "all" else [difficulty]

    for diff in diffs_to_run:
        print(f"\n>>> Starting Evaluation: Difficulty={diff} | Episodes={episodes_per_diff}")
        env = InboxEnv(difficulty=diff)
        
        for ep in range(episodes_per_diff):
            obs, info = env.reset(seed=42 + ep) # Semi-deterministic seeds
            done = False
            ep_reward = 0.0
            steps_data = []

            while not done:
                # Construct prompt
                history_str = json.dumps(obs.history, indent=2)
                prompt = f"EMAIL CONTENT:\n{obs.email}\n\nCONVERSATION HISTORY:\n{history_str}\n\nProvide triage JSON:"

                action_payload = None
                max_retries = 3
                retry_delay = 6  # Seconds to wait on 429

                for attempt in range(max_retries):
                    try:
                        completion = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0,
                            response_format={"type": "json_object"}
                        )
                        raw_content = completion.choices[0].message.content
                        action_dict = json.loads(raw_content)
                        
                        # Validate with Pydantic for schema compliance
                        validated_action = EmailAction(**action_dict)
                        action_payload = validated_action.model_dump()
                        break # Success!
                    except Exception as e:
                        if "429" in str(e) and attempt < max_retries - 1:
                            print(f"  [Step {obs.step}] Rate limit hit. Retrying in {retry_delay}s... (Attempt {attempt+1})")
                            import time
                            time.sleep(retry_delay)
                            continue
                        else:
                            print(f"  [Step {obs.step}] LLM error/parse failed: {e}. Using random fallback.")
                            action_payload = get_random_action()
                            break

                obs, reward, terminated, truncated, step_info = env.step(action_payload)
                done = terminated or truncated
                ep_reward += reward
                
                steps_data.append({
                    "step": step_info["step"],
                    "reward": reward,
                    "breakdown": step_info["reward_breakdown"]
                })

            all_results.append({
                "difficulty": diff,
                "episode": ep,
                "total_reward": round(ep_reward, 4),
                "steps": steps_data
            })
            print(f"  Episode {ep} completed. Reward: {ep_reward:.4f}")

    # Save to file
    output_file = "baseline_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    if all_results:
        df = pd.DataFrame(all_results)
        summary = df.groupby("difficulty")["total_reward"].agg(["mean", "std", "count"]).reset_index()
        print("\n" + "="*40)
        print("          BASELINE PERFORMANCE")
        print("="*40)
        print(summary.to_string(index=False))
        print("="*40)

def main():
    parser = argparse.ArgumentParser(description="Run OpenAI Baseline for SmartInboxRL")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per difficulty (default: 3)")
    parser.add_argument("--difficulty", type=str, default="all", choices=["easy", "medium", "hard", "all"], help="Difficulty to test")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    
    args = parser.parse_args()

    # Allow MODEL_NAME env var to override default model
    model = os.environ.get("MODEL_NAME", args.model)
    
    try:
        run_evaluation(model, args.episodes, args.difficulty)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")

if __name__ == "__main__":
    main()
