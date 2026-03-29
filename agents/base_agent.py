"""Abstract base class for SmartInboxRL agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Interface that every SmartInboxRL agent must implement."""

    name: str = "base"

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Given an observation, return an action dict.

        The action dict must contain:
          - intents:  list[str]
          - priority: str   (low / medium / high / critical)
          - action:   str   (reply / ignore / escalate / forward)
          - response: str
        """
        ...

    def reset(self) -> None:
        """Called at the start of each episode (optional override)."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} ({self.name})>"
