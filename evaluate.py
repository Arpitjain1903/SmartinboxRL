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

# Primary comparison metric: mean_step_reward (normalized [0, 1] per step)
# total_reward is the raw cumulative sum and is unbounded by episode length.


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def _run_single_episode(agent_name: str, difficulty: str, seed: int, ep_idx: int):
    """Run a single episode and return its summary. Used by ThreadPoolExecutor."""
    agent = _AGENT_MAP[agent_name](seed + ep_idx)
    env = InboxEnv(difficulty=difficulty, seed=seed + ep_idx)

    agent.reset()
    obs, _ = env.reset()

    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return env.get_episode_summary()


def evaluate_agent(
    agent_name: str,
    difficulty: str,
    episodes: int,
    seed: int,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Run multiple episodes for one (agent, difficulty) combo.

    Uses ThreadPoolExecutor for parallel execution when possible.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # primary metric: mean_step_reward (normalized, [0, 1])
    step_rewards: list[float] = []
    total_rewards: list[float] = []
    breakdowns: dict[str, list[float]] = {
        "intent": [], "priority": [], "action": [], "response": [], "penalty": []
    }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_single_episode, agent_name, difficulty, seed, ep): ep
            for ep in range(episodes)
        }
        for future in as_completed(futures):
            summary = future.result()
            step_rewards.append(summary["mean_step_reward"])
            total_rewards.append(summary["total_reward"])
            norm_bd = summary.get("norm_reward_breakdown", {})
            for k in breakdowns:
                breakdowns[k].append(norm_bd.get(k, 0.0))

    sr = np.array(step_rewards)
    tr = np.array(total_rewards)
    return {
        "agent": agent_name,
        "difficulty": difficulty,
        "episodes": episodes,
        # --- PRIMARY (normalized, [0, 1] per step) ---
        "mean_reward": round(float(sr.mean()), 4),
        "std_reward": round(float(sr.std()), 4),
        "min_reward": round(float(sr.min()), 4),
        "max_reward": round(float(sr.max()), 4),
        # --- reference (unbounded cumulative sum) ---
        "mean_total_episode_reward": round(float(tr.mean()), 4),
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
        f"{'Mean/Step':>10} {'Std':>8} {'Min':>8} {'Max':>8}"
    )
    print(f"\n{'='*72}")
    print("  SmartInboxRL — Comparative Evaluation")
    print("  Metric: mean_step_reward  (normalized per-step, range [0.0, 1.0])")
    print(f"{'='*72}")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    for r in results:
        row = (
            f"  {r['agent']:<10} {r['difficulty']:<10} {r['episodes']:<8} "
            f"{r['mean_reward']:>+10.4f} {r['std_reward']:>8.4f} "
            f"{r['min_reward']:>+8.4f} {r['max_reward']:>+8.4f}"
        )
        print(row)

    print(f"{'='*72}")

    # Component breakdown (all normalized)
    print(f"\n  Component Breakdown — mean_step values (normalized, per-step):")
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
        default=["random", "rule", "llm"],
        choices=list(_AGENT_MAP),
        help="Agents to evaluate (default: random rule llm)",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy", "medium", "hard", "enron"],
        choices=["easy", "medium", "hard", "enron", "all"],
        help="Difficulties to evaluate (default: easy medium hard enron)",
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
