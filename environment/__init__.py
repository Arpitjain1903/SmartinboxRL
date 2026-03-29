"""SmartInboxRL Environment — Gymnasium-compatible inbox management environment."""

from environment.inbox_env import InboxEnv
from environment.action_space import ActionType, VALID_PRIORITIES, VALID_INTENTS
from environment.state import EpisodeState
from environment.email_loader import EmailLoader

__all__ = [
    "InboxEnv",
    "ActionType",
    "VALID_PRIORITIES",
    "VALID_INTENTS",
    "EpisodeState",
    "EmailLoader",
]
