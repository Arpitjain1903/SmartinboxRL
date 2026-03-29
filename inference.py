"""SmartInboxRL — Inference CLI.

Run evaluation episodes with different agents and difficulty tiers.

Usage
-----
    python inference.py --agent llm --episodes 20 --difficulty all
    python inference.py --agent rule --episodes 50 --difficulty medium
    python inference.py --agent random --episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()

from environment.inbox_env import InboxEnv
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from agents.llm_agent import LLMAgent


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(name: str, seed: int | None = None):
    agents = {
        "random": lambda: RandomAgent(seed=seed),
        "rule": lambda: RuleAgent(),
        "llm": lambda: LLMAgent(),
    }
    if name not in agents:
        raise ValueError(f"Unknown agent: {name}. Choose from: {list(agents)}")
    return agents[name]()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: InboxEnv,
    agent,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a single episode and return the summary."""
    agent.reset()
    obs, info = env.reset()
    total_reward = 0.0
    steps = []

    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        step_info = {
            "email_id": info.get("email_id"),
            "action": action["action"],
            "priority": action["priority"],
            "intents": action["intents"],
            "reward": round(reward, 4),
            "breakdown": info.get("reward_breakdown", {}),
        }
        steps.append(step_info)

        if verbose:
            print(
                f"  Step {info.get('step', '?'):>2} | "
                f"Action: {action['action']:<8} | "
                f"Priority: {action['priority']:<8} | "
                f"Reward: {reward:+.4f}"
            )

        if terminated or truncated:
            break

    return {
        "total_reward": round(total_reward, 4),
        "steps": len(steps),
        "step_details": steps,
        "summary": env.get_episode_summary(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SmartInboxRL — Run evaluation episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="rule",
        choices=["random", "rule", "llm"],
        help="Agent type to evaluate (default: rule)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty tier (default: all)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max emails per episode (default: all in pool)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print step-by-step details",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  SmartInboxRL Evaluation")
    print(f"  Agent: {args.agent} | Episodes: {args.episodes}")
    print(f"  Difficulty: {args.difficulty} | Seed: {args.seed}")
    print(f"{'='*60}\n")

    agent = _make_agent(args.agent, seed=args.seed)
    env = InboxEnv(
        difficulty=args.difficulty,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    results = []
    rewards = []
    t0 = time.time()

    for ep in range(args.episodes):
        if args.verbose:
            print(f"\n--- Episode {ep + 1}/{args.episodes} ---")

        result = run_episode(env, agent, verbose=args.verbose)
        results.append(result)
        rewards.append(result["total_reward"])

        if not args.verbose:
            print(
                f"  Episode {ep + 1:>3}/{args.episodes} | "
                f"Reward: {result['total_reward']:+.4f} | "
                f"Steps: {result['steps']}"
            )

    elapsed = time.time() - t0

    # Summary statistics
    import numpy as np

    rewards_arr = np.array(rewards)
    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"  Agent:      {args.agent}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Mean reward: {rewards_arr.mean():+.4f}")
    print(f"  Std reward:  {rewards_arr.std():.4f}")
    print(f"  Min reward:  {rewards_arr.min():+.4f}")
    print(f"  Max reward:  {rewards_arr.max():+.4f}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed/args.episodes:.2f}s/episode)")
    print(f"{'='*60}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "agent": args.agent,
            "difficulty": args.difficulty,
            "episodes": args.episodes,
            "seed": args.seed,
            "mean_reward": float(rewards_arr.mean()),
            "std_reward": float(rewards_arr.std()),
            "min_reward": float(rewards_arr.min()),
            "max_reward": float(rewards_arr.max()),
            "elapsed_seconds": round(elapsed, 2),
            "episode_results": results,
        }
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
