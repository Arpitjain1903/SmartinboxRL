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
    "Subject: Re: Follow up\n\nHi,\n\nI'll look into this and get back to you.\n\nBest regards,\nAI Assistant",
    "Subject: Re: Update received\n\nHi,\n\nNoted, thanks for sharing this with me.\n\nBest regards,\nAI Assistant",
    "Subject: Re: Checking on this\n\nHi,\n\nLet me check on this and follow up.\n\nBest regards,\nAI Assistant",
    "Subject: Re: Discussion needed\n\nHi,\n\nCan we discuss this later today?\n\nBest regards,\nAI Assistant",
    "Subject: Re: Update\n\nHi,\n\nI appreciate the update.\n\nBest regards,\nAI Assistant",
    "",
    "Subject: Re: Acknowledged\n\nHi,\n\nOK, understood.\n\nBest regards,\nAI Assistant",
    "Subject: Re: Confirmed\n\nHi,\n\nSure thing!\n\nBest regards,\nAI Assistant",
    "Subject: Re: Action Item\n\nHi,\n\nWill do.\n\nBest regards,\nAI Assistant",
    "Subject: Re: Thank you\n\nHi,\n\nThanks for letting me know.\n\nBest regards,\nAI Assistant",
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
