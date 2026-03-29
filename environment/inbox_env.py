"""SmartInboxRL — Gymnasium-compatible inbox management environment.

The agent receives an email observation and must produce a structured
action containing: detected intents, priority, action type, and response text.
A composite reward evaluates each component independently.
"""

from __future__ import annotations

import json
from typing import Any

import gymnasium as gym
import numpy as np

from environment.action_space import validate_action
from environment.email_loader import EmailLoader
from environment.state import EpisodeState


class InboxEnv(gym.Env):
    """Realistic inbox management environment.

    Parameters
    ----------
    difficulty : str
        ``"easy"`` | ``"medium"`` | ``"hard"`` | ``"all"``
    noise_intensity : float
        Noise injection strength (0.0–1.0).
    max_steps : int | None
        Maximum emails per episode.  ``None`` = all tasks in the pool.
    history_window : int
        How many recent interactions to include in observations.
    seed : int | None
        Reproducibility seed.
    """

    metadata = {"render_modes": ["human", "json"]}

    def __init__(
        self,
        difficulty: str = "all",
        noise_intensity: float = 0.3,
        max_steps: int | None = None,
        history_window: int = 3,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.difficulty = difficulty
        self.noise_intensity = noise_intensity
        self.max_steps = max_steps
        self.history_window = history_window
        self.render_mode = render_mode

        self._loader = EmailLoader(
            difficulty=difficulty,
            noise_intensity=noise_intensity,
            shuffle=True,
            max_emails_per_episode=max_steps,
            seed=seed,
        )

        self._state: EpisodeState | None = None
        self._reward_engine: Any = None  # lazy-loaded from rewards module

        # Gymnasium spaces — we use Dict spaces for structured I/O
        # but the actual content is free-form text, so we keep them
        # as generic spaces.  Agents interact through dicts directly.
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        emails = self._loader.get_episode_emails()
        self._state = EpisodeState(
            emails=emails,
            difficulty=self.difficulty,
        )

        obs = self._build_obs()
        info = {
            "total_emails": self._state.num_emails,
            "difficulty": self.difficulty,
        }
        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one step: process agent action for the current email.

        Parameters
        ----------
        action : dict
            Must have keys: ``intents``, ``priority``, ``action``, ``response``.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step()")

        # Validate / normalise action
        clean_action = validate_action(action)

        email = self._state.current_email
        assert email is not None

        # Compute reward
        reward, breakdown = self._compute_reward(email, clean_action)

        # accumulate component breakdown
        for k, v in breakdown.items():
            self._state.reward_breakdown[k] = (
                self._state.reward_breakdown.get(k, 0.0) + v
            )

        # Record step
        self._state.record(
            email_id=email.get("id", f"step_{self._state.current_step}"),
            action=clean_action["action"],
            priority=clean_action["priority"],
            intents=clean_action["intents"],
            response=clean_action["response"],
            reward=reward,
        )

        terminated = self._state.done
        truncated = False
        obs = self._build_obs() if not terminated else {}

        info = {
            "step": self._state.current_step,
            "step_reward": reward,
            "reward_breakdown": breakdown,
            "episode_reward": self._state.total_reward,
            "remaining": self._state.remaining,
            "email_id": email.get("id"),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> str | None:
        """Render current state to stdout (human mode) or return JSON."""
        if self._state is None:
            return None

        email = self._state.current_email
        if email is None:
            summary = (
                f"Episode done — total reward: "
                f"{self._state.total_reward:.4f} "
                f"over {self._state.current_step} steps"
            )
        else:
            summary = (
                f"Step {self._state.current_step + 1}/"
                f"{self._state.num_emails} | "
                f"Difficulty: {email.get('difficulty', '?')} | "
                f"Subject: {email.get('subject', '')[:60]}"
            )

        if self.render_mode == "human":
            print(summary)
        elif self.render_mode == "json":
            return json.dumps(
                {"summary": summary, "state": self._state.__dict__},
                default=str,
            )
        return summary

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict[str, Any]:
        """Construct the observation dict for the current step."""
        assert self._state is not None
        email = self._state.current_email

        if email is None:
            return {}

        return {
            "email": {
                "id": email.get("id", ""),
                "subject": email.get("subject", ""),
                "body": email.get("body", ""),
                "sender": email.get("sender", ""),
            },
            "history": self._state.recent_history(self.history_window),
            "step": self._state.current_step,
            "total_steps": self._state.num_emails,
            "difficulty": email.get("difficulty", self.difficulty),
        }

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self, email: dict[str, Any], action: dict[str, Any]
    ) -> tuple[float, dict[str, float]]:
        """Compute composite reward for a single step.

        Lazy-loads the reward engine to avoid circular imports.
        """
        engine = self._get_reward_engine()
        return engine.compute(
            email=email,
            action=action,
            action_log=self._state.action_log if self._state else [],
        )

    def _get_reward_engine(self):
        """Lazy-load the reward engine."""
        if self._reward_engine is None:
            from rewards.reward_engine import RewardEngine

            self._reward_engine = RewardEngine()
        return self._reward_engine

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_episode_summary(self) -> dict[str, Any]:
        """Return a full summary of the completed episode."""
        if self._state is None:
            return {}
        return {
            "total_reward": round(self._state.total_reward, 4),
            "steps": self._state.current_step,
            "reward_breakdown": {
                k: round(v, 4) for k, v in self._state.reward_breakdown.items()
            },
            "history": [
                {
                    "email_id": h.email_id,
                    "action": h.action,
                    "priority": h.priority,
                    "intents": h.intents,
                    "reward": round(h.reward, 4),
                }
                for h in self._state.history
            ],
        }
