"""Pydantic v2 typed models for SmartInboxRL — OpenEnv compliance."""

from __future__ import annotations

from typing import Any, Dict, List, Literal
from pydantic import BaseModel


# Difficulty can be any string to support enron, all, interactive, etc.
# OpenEnv spec uses easy/medium/hard but we extend str for backwards compat.
SUPPORTED_DIFFICULTIES = {"easy", "medium", "hard", "enron", "all", "interactive", "mixed"}


class EmailObservation(BaseModel):
    """Structured observation returned by InboxEnv.reset() and step()."""
    email: str                  # email body text
    email_subject: str = ""     # email subject
    email_sender: str = ""      # email sender
    email_id: str = ""          # email id
    history: List[dict]         # last N interaction records
    step: int                   # current step in episode
    total_steps: int = 0        # total steps in episode
    difficulty: str             # task difficulty (easy/medium/hard/enron/all/…)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style .get() for backwards compatibility with agents."""
        if key == "email":
            return {
                "id": self.email_id,
                "subject": self.email_subject,
                "sender": self.email_sender,
                "body": self.email,
            }
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Dict-style [] access for backwards compatibility."""
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def to_dict(self) -> dict:
        """Convert to the legacy dict format agents expect."""
        return {
            "email": {
                "id": self.email_id,
                "subject": self.email_subject,
                "sender": self.email_sender,
                "body": self.email,
            },
            "history": self.history,
            "step": self.step,
            "total_steps": self.total_steps,
            "difficulty": self.difficulty,
        }


class EmailAction(BaseModel):
    """Structured action produced by an agent."""
    intents: List[str]
    priority: Literal["low", "medium", "high", "critical"]
    action: Literal["reply", "ignore", "escalate", "forward"]
    response: str


class EmailReward(BaseModel):
    """Detailed per-step reward with component breakdown."""
    total_score: float       # normalized to [0.0, 1.0] per OpenEnv spec
    intent_score: float
    priority_score: float
    action_score: float
    response_score: float
    breakdown: Dict[str, Any]


class EpisodeState(BaseModel):
    """Snapshot of the current environment state."""
    current_email: str
    step: int
    difficulty: str
    done: bool
    history: List[dict]
    total_reward: float
