"""Uniform-random baseline agent for SmartInboxRL.

Produces random intents, priorities, actions, and lorem ipsum responses.
Establishes the absolute performance floor.
"""

from __future__ import annotations

import random
from typing import Any

from agents.base_agent import BaseAgent
from environment.action_space import (
    VALID_INTENTS,
    VALID_PRIORITIES,
    ActionType,
)


_RANDOM_RESPONSES = [
    "I'll look into this and get back to you.",
    "Noted, thanks for sharing.",
    "Let me check and follow up.",
    "Can we discuss this later?",
    "I appreciate the update.",
    "",
    "OK",
    "Sure thing!",
    "Will do.",
    "Thanks for letting me know.",
]


class RandomAgent(BaseAgent):
    """Uniform-random agent — performance floor baseline."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        num_intents = self._rng.randint(1, 3)
        intents = self._rng.sample(VALID_INTENTS, k=min(num_intents, len(VALID_INTENTS)))
        priority = self._rng.choice(VALID_PRIORITIES)
        action = self._rng.choice(list(ActionType)).value
        response = self._rng.choice(_RANDOM_RESPONSES)

        return {
            "intents": list(intents),
            "priority": priority,
            "action": action,
            "response": response,
        }
