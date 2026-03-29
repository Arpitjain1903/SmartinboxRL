"""Action space definitions for SmartInboxRL.

The agent's action is a structured dict with four components:
  - intents:  list[str]  — detected email intents
  - priority: str        — urgency level
  - action:   str        — what to do with the email
  - response: str        — generated reply text (empty string allowed for IGNORE)
"""

from __future__ import annotations

from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Discrete actions the agent can take on an email."""
    REPLY = "reply"
    IGNORE = "ignore"
    ESCALATE = "escalate"
    FORWARD = "forward"


VALID_ACTIONS: set[str] = {a.value for a in ActionType}

VALID_PRIORITIES: tuple[str, ...] = ("low", "medium", "high", "critical")

VALID_INTENTS: tuple[str, ...] = (
    "meeting_request",
    "task_assignment",
    "information_sharing",
    "question",
    "feedback_request",
    "social",
    "spam",
    "complaint",
    "follow_up",
    "scheduling",
    "approval_request",
    "introduction",
    "urgent_request",
    "newsletter",
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_action(action: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise an agent action dict.

    Returns the cleaned action dict or raises ``ValueError`` on bad input.
    """
    if not isinstance(action, dict):
        raise TypeError(f"Action must be a dict, got {type(action).__name__}")

    # --- intents ---
    intents = action.get("intents", [])
    if isinstance(intents, str):
        intents = [intents]
    intents = [i.strip().lower() for i in intents if isinstance(i, str) and i.strip()]
    if not intents:
        raise ValueError("Action must contain at least one intent")

    # --- priority ---
    priority = str(action.get("priority", "medium")).strip().lower()
    if priority not in VALID_PRIORITIES:
        raise ValueError(
            f"Invalid priority '{priority}'. Must be one of {VALID_PRIORITIES}"
        )

    # --- action type ---
    action_type = str(action.get("action", "reply")).strip().lower()
    if action_type not in VALID_ACTIONS:
        raise ValueError(
            f"Invalid action '{action_type}'. Must be one of {VALID_ACTIONS}"
        )

    # --- response ---
    response = str(action.get("response", ""))

    return {
        "intents": intents,
        "priority": priority,
        "action": action_type,
        "response": response,
    }
