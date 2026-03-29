"""Episode state tracker for SmartInboxRL.

Maintains per-episode context: current email, interaction history,
step counter, and action log (used for anti-cheating penalties).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HistoryEntry:
    """A single interaction record stored in episode history."""
    step: int
    email_id: str
    action: str
    priority: str
    intents: list[str]
    response: str
    reward: float


@dataclass
class EpisodeState:
    """Mutable state container for a single episode.

    Attributes
    ----------
    emails : list[dict]
        The ordered list of email tasks for this episode.
    current_step : int
        Zero-indexed step counter.
    history : list[HistoryEntry]
        Previous interactions in this episode.
    action_log : list[str]
        Flat list of action-type strings, used for repetition penalty.
    difficulty : str
        Episode difficulty tier: easy / medium / hard / mixed.
    done : bool
        Whether the episode has terminated.
    total_reward : float
        Accumulated episode reward.
    reward_breakdown : dict[str, float]
        Running totals for each reward component.
    """

    emails: list[dict] = field(default_factory=list)
    current_step: int = 0
    history: list[HistoryEntry] = field(default_factory=list)
    action_log: list[str] = field(default_factory=list)
    difficulty: str = "mixed"
    done: bool = False
    total_reward: float = 0.0
    reward_breakdown: dict[str, float] = field(
        default_factory=lambda: {
            "intent": 0.0,
            "priority": 0.0,
            "action": 0.0,
            "response": 0.0,
            "penalty": 0.0,
        }
    )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def current_email(self) -> dict[str, Any] | None:
        """Return the email for the current step, or ``None`` if exhausted."""
        if self.current_step < len(self.emails):
            return self.emails[self.current_step]
        return None

    @property
    def num_emails(self) -> int:
        return len(self.emails)

    @property
    def remaining(self) -> int:
        return max(0, len(self.emails) - self.current_step)

    def record(
        self,
        email_id: str,
        action: str,
        priority: str,
        intents: list[str],
        response: str,
        reward: float,
    ) -> None:
        """Append an interaction to history and advance the step counter."""
        entry = HistoryEntry(
            step=self.current_step,
            email_id=email_id,
            action=action,
            priority=priority,
            intents=intents,
            response=response,
            reward=reward,
        )
        self.history.append(entry)
        self.action_log.append(action)
        self.total_reward += reward
        self.current_step += 1

        if self.current_step >= len(self.emails):
            self.done = True

    def recent_history(self, n: int = 3) -> list[dict[str, Any]]:
        """Return the last *n* interactions as plain dicts (for observations)."""
        entries = self.history[-n:] if self.history else []
        return [
            {
                "step": e.step,
                "email_id": e.email_id,
                "action": e.action,
                "priority": e.priority,
                "intents": e.intents,
                "response_snippet": e.response[:120],
                "reward": round(e.reward, 4),
            }
            for e in entries
        ]

    def clone(self) -> "EpisodeState":
        """Return a deep copy (useful for env wrappers or branching)."""
        return copy.deepcopy(self)
