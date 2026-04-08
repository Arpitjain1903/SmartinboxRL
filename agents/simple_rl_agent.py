"""Simple tabular Q-learning RL agent for SmartInboxRL.

Discretises email observations into a compact state key and maintains
a Q-table mapping (state, action) → expected reward.  Supports:
  - Epsilon-greedy exploration during training
  - Saving / loading Q-table, state index, and training log from disk
  - Graceful fallback to keyword heuristics when no training data exists

No external dependencies beyond numpy (+ stdlib).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from agents.base_agent import BaseAgent
from environment.action_space import VALID_INTENTS, VALID_PRIORITIES, VALID_ACTIONS


# ---------------------------------------------------------------------------
# Action space — every possible (action_type, priority) combination
# ---------------------------------------------------------------------------

ACTION_TYPES = ["reply", "ignore", "escalate", "forward"]
PRIORITIES = ["low", "medium", "high", "critical"]

# Flat list of all discrete actions the agent can pick from
ACTION_LIST: List[Tuple[str, str]] = [
    (a, p) for a in ACTION_TYPES for p in PRIORITIES
]  # 16 combos, but we trim to useful ones:
# Actually keep the full 16 — the Q-table will learn which combos are useful.
# But for a cleaner mapping we can use a smaller set if desired.
# Let's keep a manageable 11 that the reward engine can reasonably score:
ACTION_LIST = [
    ("reply",    "low"),
    ("reply",    "medium"),
    ("reply",    "high"),
    ("reply",    "critical"),
    ("ignore",   "low"),
    ("ignore",   "medium"),
    ("escalate", "high"),
    ("escalate", "critical"),
    ("forward",  "low"),
    ("forward",  "medium"),
    ("forward",  "high"),
]

NUM_ACTIONS = len(ACTION_LIST)

# Template responses keyed by action type
RESPONSE_TEMPLATES: Dict[str, str] = {
    "reply": "Subject: Re: Your Message\n\nHi,\n\nThank you for your email. I've reviewed the contents and will follow up with a detailed response shortly.\n\nBest regards,\nAI Assistant",
    "ignore": "",
    "escalate": "Subject: Escalation: Immediate Attention Required\n\nHi team,\n\nI'm escalating this email for immediate review. Please prioritise accordingly.\n\nBest regards,\nAI Assistant",
    "forward": "Subject: Fwd: Please Review\n\nHi team,\n\nForwarding this email to the relevant team members for review and action.\n\nBest regards,\nAI Assistant",
}

# Intent mappings for keyword rules fallback
URGENCY_KEYWORDS = {"urgent", "asap", "deadline", "p0", "critical"}
QUESTION_STARTERS = {"who", "what", "when", "where", "why", "how", "can", "could", "would", "should", "is", "are", "do", "does", "did"}

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# State discretisation helpers
# ---------------------------------------------------------------------------

def _discretise_email_length(email_text: str) -> str:
    word_count = len(email_text.split())
    if word_count < 50:
        return "short"
    elif word_count <= 150:
        return "medium"
    else:
        return "long"


def _has_urgency(email_text: str) -> bool:
    text_lower = email_text.lower()
    return any(kw in text_lower for kw in URGENCY_KEYWORDS)


def _has_question(email_text: str) -> bool:
    text_lower = email_text.lower().strip()
    if "?" in email_text:
        return True
    first_word = text_lower.split()[0] if text_lower.split() else ""
    return first_word in QUESTION_STARTERS


def _sender_type(sender: str) -> str:
    sender_lower = sender.lower()
    if "@company" in sender_lower or "@internal" in sender_lower:
        return "internal"
    return "external"


def observation_to_state_key(observation: dict) -> str:
    """Convert an observation dict to a discrete state key string."""
    email = observation.get("email", {})
    email_text = f"{email.get('subject', '')} {email.get('body', '')}"
    sender = email.get("sender", "")

    length = _discretise_email_length(email_text)
    urgency = _has_urgency(email_text)
    question = _has_question(email_text)
    s_type = _sender_type(sender)

    return f"{length}_{urgency}_{question}_{s_type}"


def _keyword_intents(text: str) -> List[str]:
    """Simple keyword-based intent detection for fallback."""
    text_l = text.lower()
    intents = []

    spam_kw = {"congratulations", "winner", "lottery", "prize", "click here",
               "limited time", "act now", "free", "unsubscribe", "viagra"}
    if sum(1 for s in spam_kw if s in text_l) >= 2:
        intents.append("spam")

    meeting_kw = {"meeting", "standup", "sync", "call", "conference",
                  "schedule", "calendar", "agenda"}
    if any(s in text_l for s in meeting_kw):
        intents.append("meeting_request")

    task_kw = {"assign", "please do", "can you", "need you to",
               "action item", "todo", "task", "responsible"}
    if any(s in text_l for s in task_kw):
        intents.append("task_assignment")

    if "?" in text_l:
        intents.append("question")

    escalation_kw = {"escalate", "incident", "outage", "down", "breach",
                     "security", "phishing", "vulnerability"}
    if sum(1 for s in escalation_kw if s in text_l) >= 2:
        intents.append("urgent_request")

    if any(w in text_l for w in ("follow up", "following up", "per our last")):
        intents.append("follow_up")

    if not intents:
        intents.append("information_sharing")

    return intents


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SimpleRLAgent(BaseAgent):
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    name = "simple_rl"

    def __init__(
        self,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = random.Random(seed)

        # Try to load saved Q-table from disk
        self.loaded_from_disk = False
        self.q_table: Any = {}       # will be dict or np.ndarray
        self.state_index: Dict[str, int] = {}

        try:
            q_path = DATA_DIR / "q_table.npy"
            si_path = DATA_DIR / "q_table_states.json"
            log_path = DATA_DIR / "training_log.json"

            if q_path.exists() and si_path.exists():
                self.q_table = np.load(str(q_path))
                with open(si_path, "r") as f:
                    self.state_index = json.load(f)
                self.loaded_from_disk = True

                # Restore epsilon from training log if available
                if log_path.exists():
                    with open(log_path, "r") as f:
                        log = json.load(f)
                    if log and isinstance(log, list) and len(log) > 0:
                        last_eps = log[-1].get("epsilon")
                        if last_eps is not None:
                            self.epsilon = float(last_eps)
        except Exception:
            # Any error → start fresh
            self.q_table = {}
            self.state_index = {}
            self.loaded_from_disk = False

    # ------------------------------------------------------------------
    # act()
    # ------------------------------------------------------------------

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Select an action given an observation.

        If a trained Q-table is loaded and the state has been seen,
        exploit the learned policy. Otherwise, fall back to epsilon-greedy
        exploration or keyword heuristics.
        """
        state_key = observation_to_state_key(observation)
        email = observation.get("email", {})
        email_text = f"{email.get('subject', '')} {email.get('body', '')}"

        # --- Exploitation path: use loaded Q-table ---
        if (
            self.loaded_from_disk
            and isinstance(self.q_table, np.ndarray)
            and state_key in self.state_index
        ):
            # Epsilon-greedy: small chance of random even during inference
            if self._rng.random() < self.epsilon:
                action_idx = self._rng.randint(0, NUM_ACTIONS - 1)
            else:
                state_idx = self.state_index[state_key]
                action_idx = int(np.argmax(self.q_table[state_idx]))

            action_type, priority = ACTION_LIST[action_idx]
            intents = _keyword_intents(email_text)
            response = RESPONSE_TEMPLATES.get(action_type, "")

            return {
                "intents": intents,
                "priority": priority,
                "action": action_type,
                "response": response,
            }

        # --- Exploration / fallback path ---
        if self._rng.random() < self.epsilon:
            # Pure random
            action_idx = self._rng.randint(0, NUM_ACTIONS - 1)
            action_type, priority = ACTION_LIST[action_idx]
            intents = self._rng.sample(
                list(VALID_INTENTS),
                k=min(self._rng.randint(1, 3), len(VALID_INTENTS)),
            )
        else:
            # Simple keyword heuristic fallback
            action_type, priority, intents = self._keyword_fallback(email_text)

        response = RESPONSE_TEMPLATES.get(action_type, "")

        return {
            "intents": list(intents),
            "priority": priority,
            "action": action_type,
            "response": response,
        }

    # ------------------------------------------------------------------
    # Learning methods (called by Trainer)
    # ------------------------------------------------------------------

    def _ensure_state(self, state_key: str) -> int:
        """Ensure state_key exists in our Q-table, return its index."""
        if isinstance(self.q_table, np.ndarray):
            if state_key not in self.state_index:
                idx = len(self.state_index)
                self.state_index[state_key] = idx
                new_row = np.zeros((1, NUM_ACTIONS))
                self.q_table = np.vstack([self.q_table, new_row])
                return idx
            return self.state_index[state_key]
        else:
            # dict mode (fresh start)
            if state_key not in self.state_index:
                idx = len(self.state_index)
                self.state_index[state_key] = idx
            return self.state_index[state_key]

    def _action_to_idx(self, action: dict) -> int:
        """Convert an action dict to the closest ACTION_LIST index."""
        a_type = action.get("action", "reply")
        a_prio = action.get("priority", "medium")
        target = (a_type, a_prio)
        if target in ACTION_LIST:
            return ACTION_LIST.index(target)
        # Find closest match (same action type, any priority)
        for i, (at, ap) in enumerate(ACTION_LIST):
            if at == a_type:
                return i
        return 0  # default

    def update(
        self,
        obs: dict,
        action: dict,
        reward: float,
        next_obs: dict,
        done: bool,
    ) -> None:
        """Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]"""
        state_key = observation_to_state_key(
            obs.to_dict() if hasattr(obs, "to_dict") else obs
        )
        next_state_key = observation_to_state_key(
            next_obs.to_dict() if hasattr(next_obs, "to_dict") else (next_obs if isinstance(next_obs, dict) else {})
        )

        # Convert dict-based Q-table to numpy on first update
        if not isinstance(self.q_table, np.ndarray):
            n_states = max(len(self.state_index), 1)
            self.q_table = np.zeros((n_states, NUM_ACTIONS))

        s_idx = self._ensure_state(state_key)
        a_idx = self._action_to_idx(action if isinstance(action, dict) else {})

        if done:
            td_target = reward
        else:
            ns_idx = self._ensure_state(next_state_key)
            td_target = reward + self.gamma * float(np.max(self.q_table[ns_idx]))

        self.q_table[s_idx, a_idx] += self.lr * (td_target - self.q_table[s_idx, a_idx])

    def on_episode_end(self, episode: int, avg_reward: float) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_state(self) -> None:
        """Persist Q-table, state index, and training log to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if isinstance(self.q_table, np.ndarray):
            np.save(str(DATA_DIR / "q_table.npy"), self.q_table)

        with open(DATA_DIR / "q_table_states.json", "w") as f:
            json.dump(self.state_index, f, indent=2)

    def reset(self) -> None:
        """No-op — preserved for BaseAgent interface."""
        pass

    # ------------------------------------------------------------------
    # Keyword fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_fallback(text: str) -> Tuple[str, str, List[str]]:
        """Use simple keyword rules to decide action, priority, intents."""
        text_l = text.lower()
        intents = _keyword_intents(text)

        # Priority
        urgency = sum(1 for kw in URGENCY_KEYWORDS if kw in text_l)
        if urgency >= 2:
            priority = "critical"
        elif urgency >= 1:
            priority = "high"
        elif "?" in text_l:
            priority = "medium"
        else:
            priority = "low"

        # Action
        if "spam" in intents:
            action = "ignore"
        elif "urgent_request" in intents or priority == "critical":
            action = "escalate"
        elif "meeting_request" in intents or "question" in intents:
            action = "reply"
        else:
            action = "reply"

        return action, priority, intents
