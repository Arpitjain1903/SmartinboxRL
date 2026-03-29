"""Anti-cheating penalty system for SmartInboxRL.

Applies structured penalties that prevent reward hacking:
  - Repeated consecutive identical actions
  - Ignoring critical/high-priority emails
  - Trivial / low-effort generated responses
"""

from __future__ import annotations

from typing import Any


class PenaltySystem:
    """Compute penalties for a given step.

    Parameters
    ----------
    repeat_penalty : float
        Penalty per consecutive repeated action (negative).
    critical_ignore_penalty : float
        Penalty for ignoring a critical/high-priority email.
    trivial_response_penalty : float
        Penalty for a low-effort response on an email that requires one.
    min_response_length : int
        Minimum character count for a non-trivial response.
    """

    def __init__(
        self,
        repeat_penalty: float = -0.20,
        critical_ignore_penalty: float = -0.50,
        trivial_response_penalty: float = -0.30,
        min_response_length: int = 20,
    ):
        self.repeat_penalty = repeat_penalty
        self.critical_ignore_penalty = critical_ignore_penalty
        self.trivial_response_penalty = trivial_response_penalty
        self.min_response_length = min_response_length

    def compute(
        self,
        email: dict[str, Any],
        action: dict[str, Any],
        action_log: list[str],
    ) -> float:
        """Return the total penalty (≤ 0) for a single step."""
        total = 0.0

        total += self._check_repeat(action, action_log)
        total += self._check_critical_ignore(email, action)
        total += self._check_trivial_response(email, action)

        return total

    # ------------------------------------------------------------------
    # Individual penalty checks
    # ------------------------------------------------------------------

    def _check_repeat(
        self, action: dict[str, Any], action_log: list[str]
    ) -> float:
        """Penalise consecutive identical actions."""
        if not action_log:
            return 0.0

        act = action["action"]
        # Count how many consecutive trailing entries match
        streak = 0
        for prev in reversed(action_log):
            if prev == act:
                streak += 1
            else:
                break

        if streak >= 2:
            return self.repeat_penalty * (streak - 1)
        return 0.0

    def _check_critical_ignore(
        self, email: dict[str, Any], action: dict[str, Any]
    ) -> float:
        """Penalise ignoring emails that have high or critical gold priority."""
        if action["action"] != "ignore":
            return 0.0

        gold_priority = email.get("gold_priority", "low")
        if gold_priority in ("high", "critical"):
            return self.critical_ignore_penalty
        return 0.0

    def _check_trivial_response(
        self, email: dict[str, Any], action: dict[str, Any]
    ) -> float:
        """Penalise trivial responses when a substantive reply is expected."""
        # Only applies when the agent chose to reply
        if action["action"] != "reply":
            return 0.0

        gold_action = email.get("gold_action", "reply")
        if gold_action == "ignore":
            # No response expected, don't penalise
            return 0.0

        response = action.get("response", "")
        if len(response.strip()) < self.min_response_length:
            return self.trivial_response_penalty

        # Check for generic non-answers
        low = response.strip().lower()
        generic_phrases = {
            "ok", "okay", "sure", "thanks", "got it",
            "noted", "acknowledged", "will do", "k", "fine",
        }
        if low in generic_phrases:
            return self.trivial_response_penalty

        return 0.0
