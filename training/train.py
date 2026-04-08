"""SmartInboxRL — Training Loop.

Provides a Trainer class that runs RL training episodes, logs progress,
saves agent state, and can compare agent baselines.

Usage
-----
    python training/train.py --agent simple_rl --episodes 100
    python training/train.py --agent simple_rl --episodes 100 --difficulty easy
    python training/train.py --compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.inbox_env import InboxEnv

DATA_DIR = PROJECT_ROOT / "data"


class Trainer:
    """Training loop for SmartInboxRL agents."""

    def __init__(
        self,
        agent_type: str = "simple_rl",
        episodes: int = 100,
        difficulty: str = "all",
        seed: int = 42,
    ):
        self.agent_type = agent_type
        self.episodes = episodes
        self.difficulty = difficulty
        self.seed = seed

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Instantiate agent
        self.agent = self._make_agent(agent_type)

    @staticmethod
    def _make_agent(agent_type: str, for_eval: bool = False) -> Any:
        """Import and instantiate the correct agent."""
        if agent_type == "simple_rl":
            from agents.simple_rl_agent import SimpleRLAgent
            return SimpleRLAgent()
        elif agent_type == "rl_llm":
            from agents.rl_agent import RLAgent
            return RLAgent()
        elif agent_type == "random":
            from agents.random_agent import RandomAgent
            return RandomAgent()
        elif agent_type == "rule":
            from agents.rule_agent import RuleAgent
            return RuleAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        env = InboxEnv(difficulty=self.difficulty, seed=self.seed)

        print(f"\n{'='*60}")
        print(f"  SmartInboxRL Training")
        print(f"  Agent:      {self.agent_type}")
        print(f"  Episodes:   {self.episodes}")
        print(f"  Difficulty: {self.difficulty}")
        print(f"  Seed:       {self.seed}")
        print(f"{'='*60}\n")

        for episode in range(self.episodes):
            obs, _ = env.reset()
            episode_rewards: List[float] = []
            prev_obs = obs
            done = False

            while not done:
                # Convert observation for agent
                obs_for_agent = obs.to_dict() if hasattr(obs, "to_dict") else obs
                action = self.agent.act(obs_for_agent)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Critical: call the agent's update method for learning
                if hasattr(self.agent, "update"):
                    self.agent.update(prev_obs, action, reward, next_obs, done)

                episode_rewards.append(float(reward))
                prev_obs = next_obs
                obs = next_obs

            avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0

            # Episode-end hook (e.g. epsilon decay)
            if hasattr(self.agent, "on_episode_end"):
                self.agent.on_episode_end(episode, avg_reward)

            # Periodic save and logging
            if episode % 10 == 0:
                self._save_agent()
                self._log_progress(episode, avg_reward)

        # Final save
        self._save_agent()

        # Log final episode if not already logged
        if (self.episodes - 1) % 10 != 0:
            # Run one more metric calc for the final state
            self._log_progress(self.episodes - 1, avg_reward)

        self._print_training_summary()

    # ------------------------------------------------------------------
    # Agent save helper
    # ------------------------------------------------------------------

    def _save_agent(self) -> None:
        """Save agent state to disk."""
        if hasattr(self.agent, "save_state"):
            self.agent.save_state()
        elif hasattr(self.agent, "save"):
            self.agent.save()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_progress(self, episode: int, avg_reward: float) -> None:
        """Append a progress entry to data/training_log.json."""
        log_path = DATA_DIR / "training_log.json"

        # Load existing log
        log: List[dict] = []
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    log = json.load(f)
                if not isinstance(log, list):
                    log = []
            except (json.JSONDecodeError, Exception):
                log = []

        entry = {
            "episode": episode,
            "avg_reward": round(avg_reward, 6),
            "epsilon": round(getattr(self.agent, "epsilon", 0.0), 6),
            "timestamp": datetime.now().isoformat(),
            "agent_type": self.agent_type,
        }
        log.append(entry)

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        eps_val = getattr(self.agent, "epsilon", 0)
        print(
            f"  Episode {episode:4d} | Reward: {avg_reward:.4f} | "
            f"Epsilon: {eps_val:.3f}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_training_summary(self) -> None:
        """Print a summary comparing early vs. late episode performance."""
        log_path = DATA_DIR / "training_log.json"

        if not log_path.exists():
            print("\n[!] No training log found.")
            return

        try:
            with open(log_path, "r") as f:
                log = json.load(f)
        except Exception:
            print("\n[!] Could not read training log.")
            return

        if not log or not isinstance(log, list):
            print("\n[!] Training log is empty.")
            return

        # Filter to current run's entries
        rewards = [entry["avg_reward"] for entry in log if "avg_reward" in entry]

        if len(rewards) < 2:
            print("\n[!] Not enough data points for comparison.")
            return

        n = min(10, len(rewards) // 2)
        first_n = rewards[:n]
        last_n = rewards[-n:]

        first_avg = float(np.mean(first_n))
        last_avg = float(np.mean(last_n))
        delta = last_avg - first_avg

        print(f"\n{'='*60}")
        print(f"  Training Summary")
        print(f"{'='*60}")
        print(f"  First {n} episodes avg reward:  {first_avg:.4f}")
        print(f"  Last  {n} episodes avg reward:  {last_avg:.4f}")
        print(f"  Improvement: {'+' if delta >= 0 else ''}{delta:.4f}")
        print()

        if delta > 0:
            print("  [OK] Agent is learning!")
        else:
            print("  [!] No improvement detected. Try more episodes.")

        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Compare agents
    # ------------------------------------------------------------------

    def compare_agents(self) -> None:
        """Run 20 eval episodes for each agent and print comparison."""
        eval_episodes = 5
        agents_to_compare = ["random", "rule", "simple_rl"]

        # Check if rl_llm can work (has API key)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            agents_to_compare.append("rl_llm")

        results: dict[str, dict] = {}

        print(f"\n{'='*60}")
        print(f"  Agent Comparison ({eval_episodes} episodes each)")
        print(f"{'='*60}\n")

        for agent_name in agents_to_compare:
            try:
                agent = self._make_agent(agent_name, for_eval=True)
                env = InboxEnv(difficulty=self.difficulty, seed=self.seed)

                ep_rewards: List[float] = []

                for ep in range(eval_episodes):
                    obs, _ = env.reset()
                    step_rewards: List[float] = []
                    done = False

                    while not done:
                        obs_dict = obs.to_dict() if hasattr(obs, "to_dict") else obs
                        action = agent.act(obs_dict)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        step_rewards.append(float(reward))

                    ep_avg = float(np.mean(step_rewards)) if step_rewards else 0.0
                    ep_rewards.append(ep_avg)

                arr = np.array(ep_rewards)
                results[agent_name] = {
                    "avg": float(arr.mean()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "std": float(arr.std()),
                }
                print(f"  [OK] {agent_name} evaluated")

            except Exception as e:
                print(f"  [FAIL] {agent_name} failed: {e}")
                results[agent_name] = {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        # Print ASCII table
        print(f"\n  {'Agent':<15} {'Avg Reward':>12} {'Min':>8} {'Max':>8} {'Std':>8}")
        print(f"  {'-'*15} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

        for name, stats in sorted(results.items(), key=lambda x: x[1]["avg"], reverse=True):
            print(
                f"  {name:<15} {stats['avg']:>12.4f} {stats['min']:>8.4f} "
                f"{stats['max']:>8.4f} {stats['std']:>8.4f}"
            )

        print()

        # Save to JSON
        comparison_path = DATA_DIR / "comparison_results.json"
        with open(comparison_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {comparison_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SmartInboxRL Training")
    parser.add_argument(
        "--agent",
        default="simple_rl",
        choices=["simple_rl", "rl_llm", "random", "rule"],
        help="Agent type to train",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--difficulty",
        default="all",
        choices=["easy", "medium", "hard", "all"],
        help="Task difficulty",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all agents instead of training",
    )

    args = parser.parse_args()

    trainer = Trainer(
        agent_type=args.agent,
        episodes=args.episodes,
        difficulty=args.difficulty,
        seed=args.seed,
    )

    if args.compare:
        trainer.compare_agents()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
