"""SmartInboxRL — Batch evaluation with comparative metrics.

Runs all agents across all difficulty tiers and produces a comparative
results table plus JSON export.

Usage
-----
    python evaluate.py
    python evaluate.py --agents rule random --episodes 25 -o results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np

from environment.inbox_env import InboxEnv
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from agents.llm_agent import LLMAgent


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_AGENT_MAP = {
    "random": lambda seed: RandomAgent(seed=seed),
    "rule": lambda _: RuleAgent(),
    "llm": lambda _: LLMAgent(),
}


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent_name: str,
    difficulty: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run multiple episodes for one (agent, difficulty) combo."""

    agent = _AGENT_MAP[agent_name](seed)
    env = InboxEnv(difficulty=difficulty, seed=seed)

    rewards = []
    breakdowns = {"intent": [], "priority": [], "action": [], "response": [], "penalty": []}

    for _ in range(episodes):
        agent.reset()
        obs, _ = env.reset()

        while True:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        summary = env.get_episode_summary()
        rewards.append(summary["total_reward"])
        for k in breakdowns:
            breakdowns[k].append(summary["reward_breakdown"].get(k, 0.0))

    rewards_arr = np.array(rewards)
    return {
        "agent": agent_name,
        "difficulty": difficulty,
        "episodes": episodes,
        "mean_reward": round(float(rewards_arr.mean()), 4),
        "std_reward": round(float(rewards_arr.std()), 4),
        "min_reward": round(float(rewards_arr.min()), 4),
        "max_reward": round(float(rewards_arr.max()), 4),
        "component_means": {
            k: round(float(np.mean(v)), 4) for k, v in breakdowns.items()
        },
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_table(results: list[dict[str, Any]]):
    """Print a comparative results table."""
    header = (
        f"{'Agent':<10} {'Difficulty':<10} {'Episodes':<8} "
        f"{'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}"
    )
    print(f"\n{'='*70}")
    print("  SmartInboxRL — Comparative Evaluation")
    print(f"{'='*70}")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    for r in results:
        row = (
            f"  {r['agent']:<10} {r['difficulty']:<10} {r['episodes']:<8} "
            f"{r['mean_reward']:>+8.4f} {r['std_reward']:>8.4f} "
            f"{r['min_reward']:>+8.4f} {r['max_reward']:>+8.4f}"
        )
        print(row)

    print(f"{'='*70}")

    # Component breakdown
    print(f"\n  Component Breakdown (means across all episodes):")
    print(f"  {'Agent':<10} {'Diff':<8} {'Intent':>8} {'Priority':>8} {'Action':>8} {'Response':>8} {'Penalty':>8}")
    print(f"  {'-'*62}")
    for r in results:
        cm = r["component_means"]
        print(
            f"  {r['agent']:<10} {r['difficulty']:<8} "
            f"{cm.get('intent', 0):>8.4f} {cm.get('priority', 0):>8.4f} "
            f"{cm.get('action', 0):>8.4f} {cm.get('response', 0):>8.4f} "
            f"{cm.get('penalty', 0):>+8.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SmartInboxRL — Batch evaluation")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["random", "rule"],
        choices=list(_AGENT_MAP),
        help="Agents to evaluate (default: random rule)",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty tiers (default: easy medium hard)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per (agent, difficulty) combo (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default=None)

    args = parser.parse_args()

    all_results = []
    t0 = time.time()

    total_combos = len(args.agents) * len(args.difficulties)
    i = 0

    for agent_name in args.agents:
        for diff in args.difficulties:
            i += 1
            print(f"  [{i}/{total_combos}] Evaluating {agent_name} on {diff}...")
            result = evaluate_agent(agent_name, diff, args.episodes, args.seed)
            all_results.append(result)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    print_table(all_results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
