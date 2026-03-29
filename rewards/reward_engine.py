"""Composite reward engine for SmartInboxRL.

Combines four evaluation components:
  - Intent understanding  (30%)  — F1 between predicted & gold intents
  - Priority correctness  (20%)  — exact match
  - Action decision       (20%)  — exact match
  - Response quality      (30%)  — embedding cosine similarity

Plus structured penalties from the penalty system.
"""

from __future__ import annotations

from typing import Any

from rewards.embedding_scorer import EmbeddingScorer
from rewards.penalty_system import PenaltySystem


class RewardEngine:
    """Compute the composite reward for a single step.

    Parameters
    ----------
    weights : dict[str, float] | None
        Override default component weights.
    """

    DEFAULT_WEIGHTS = {
        "intent": 0.30,
        "priority": 0.20,
        "action": 0.20,
        "response": 0.30,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        penalty_system: PenaltySystem | None = None,
        embedding_scorer: EmbeddingScorer | None = None,
    ):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        self._penalty = penalty_system or PenaltySystem()
        self._scorer = embedding_scorer or EmbeddingScorer()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        email: dict[str, Any],
        action: dict[str, Any],
        action_log: list[str] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Return ``(total_reward, component_breakdown)``.

        The breakdown dict has keys: intent, priority, action, response, penalty.
        """
        action_log = action_log or []

        # --- Individual component scores (each ∈ [0, 1]) ---
        intent_score = self._score_intents(
            predicted=action["intents"],
            gold=email.get("gold_intents", []),
        )
        priority_score = self._score_priority(
            predicted=action["priority"],
            gold=email.get("gold_priority", "medium"),
        )
        action_score = self._score_action(
            predicted=action["action"],
            gold=email.get("gold_action", "reply"),
        )
        response_score = self._score_response(
            predicted=action.get("response", ""),
            gold=email.get("gold_response", ""),
            action_type=action["action"],
        )

        # Weighted sum
        weighted = (
            self.weights["intent"] * intent_score
            + self.weights["priority"] * priority_score
            + self.weights["action"] * action_score
            + self.weights["response"] * response_score
        )

        # Penalties
        penalty = self._penalty.compute(email, action, action_log)

        total = max(-1.0, min(1.0, weighted + penalty))

        breakdown = {
            "intent": round(intent_score, 4),
            "priority": round(priority_score, 4),
            "action": round(action_score, 4),
            "response": round(response_score, 4),
            "penalty": round(penalty, 4),
            "total": round(total, 4),
        }

        return total, breakdown

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_intents(
        predicted: list[str], gold: list[str]
    ) -> float:
        """F1 score between predicted and gold intent labels."""
        if not gold:
            return 1.0 if not predicted else 0.5  # no gold = accept anything

        pred_set = set(predicted)
        gold_set = set(gold)

        if not pred_set:
            return 0.0

        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gold_set) if gold_set else 0.0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    @staticmethod
    def _score_priority(predicted: str, gold: str) -> float:
        """Exact match → 1.0, one-step off → 0.4, else → 0.0."""
        levels = ("low", "medium", "high", "critical")
        if predicted == gold:
            return 1.0
        try:
            diff = abs(levels.index(predicted) - levels.index(gold))
        except ValueError:
            return 0.0
        if diff == 1:
            return 0.4
        return 0.0

    @staticmethod
    def _score_action(predicted: str, gold: str) -> float:
        """Exact match → 1.0, partial credit for related actions → 0.3."""
        if predicted == gold:
            return 1.0

        # partial credit: escalate ↔ forward are related
        related = {
            ("escalate", "forward"),
            ("forward", "escalate"),
            ("reply", "escalate"),
            ("escalate", "reply"),
        }
        if (predicted, gold) in related:
            return 0.3

        return 0.0

    def _score_response(
        self,
        predicted: str,
        gold: str,
        action_type: str,
    ) -> float:
        """Semantic similarity between predicted and gold response.

        If the gold action is ``ignore`` (empty response expected),
        an empty predicted response scores 1.0.
        """
        # If no response expected, reward silence
        if not gold or gold.strip() == "":
            if not predicted or predicted.strip() == "":
                return 1.0
            # Penalty for generating noise when none was needed
            return 0.3

        # If agent chose to ignore but a response was expected
        if action_type == "ignore" and gold.strip():
            return 0.0

        if not predicted or predicted.strip() == "":
            return 0.0

        return self._scorer.score(predicted, gold)
