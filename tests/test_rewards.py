"""Tests for rewards/ — RewardEngine, PenaltySystem, EmbeddingScorer."""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rewards.reward_engine import RewardEngine
from rewards.penalty_system import PenaltySystem
from rewards.embedding_scorer import EmbeddingScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _email(
    gold_intents=None,
    gold_priority="medium",
    gold_action="reply",
    gold_response="Thank you for reaching out.",
):
    return {
        "id": "test_001",
        "subject": "Test email",
        "body": "Hello there",
        "sender": "test@example.com",
        "gold_intents": gold_intents or ["information_sharing"],
        "gold_priority": gold_priority,
        "gold_action": gold_action,
        "gold_response": gold_response,
    }


def _action(
    intents=None,
    priority="medium",
    action="reply",
    response="Thank you for reaching out.",
):
    return {
        "intents": intents or ["information_sharing"],
        "priority": priority,
        "action": action,
        "response": response,
    }


# ===========================================================================
# PenaltySystem
# ===========================================================================

class TestPenaltySystem:
    def setup_method(self):
        self.ps = PenaltySystem()

    # --- repeat penalty ---

    def test_no_penalty_no_history(self):
        penalty = self.ps.compute(_email(), _action(), action_log=[])
        assert penalty == 0.0

    def test_no_penalty_single_repeat(self):
        # only 1 prior entry — streak = 1, no penalty
        penalty = self.ps.compute(_email(), _action(action="reply"), action_log=["reply"])
        assert penalty == 0.0

    def test_repeat_penalty_two_consecutive(self):
        # streak = 2  →  penalty = -0.20 × (2-1) = -0.20
        log = ["reply", "reply"]
        penalty = self.ps._check_repeat(_action(action="reply"), log)
        assert penalty == pytest.approx(-0.20)

    def test_repeat_penalty_three_consecutive(self):
        # streak = 3  →  penalty = -0.20 × 2 = -0.40
        log = ["reply", "reply", "reply"]
        penalty = self.ps._check_repeat(_action(action="reply"), log)
        assert penalty == pytest.approx(-0.40)

    def test_repeat_streak_broken_by_different_action(self):
        log = ["ignore", "reply", "reply"]
        # streak for "reply" = 2 → this step is "reply" → streak = 2 → penalty = -0.20
        penalty = self.ps._check_repeat(_action(action="reply"), log)
        assert penalty == pytest.approx(-0.20)

    def test_no_repeat_penalty_when_streak_broken(self):
        log = ["reply", "ignore"]
        # last action is "ignore", now we take "reply" → streak = 0
        penalty = self.ps._check_repeat(_action(action="reply"), log)
        assert penalty == 0.0

    # --- critical ignore penalty ---

    def test_critical_ignore_penalty(self):
        email = _email(gold_priority="critical")
        act = _action(action="ignore")
        penalty = self.ps._check_critical_ignore(email, act)
        assert penalty == pytest.approx(-0.50)

    def test_high_ignore_penalty(self):
        email = _email(gold_priority="high")
        act = _action(action="ignore")
        penalty = self.ps._check_critical_ignore(email, act)
        assert pytest.approx(-0.50) == penalty

    def test_no_critical_penalty_for_low_priority(self):
        email = _email(gold_priority="low")
        act = _action(action="ignore")
        penalty = self.ps._check_critical_ignore(email, act)
        assert penalty == 0.0

    def test_no_critical_penalty_when_not_ignoring(self):
        email = _email(gold_priority="critical")
        act = _action(action="reply")
        penalty = self.ps._check_critical_ignore(email, act)
        assert penalty == 0.0

    # --- trivial response penalty ---

    def test_trivial_response_too_short(self):
        email = _email(gold_action="reply")
        act = _action(action="reply", response="ok")
        penalty = self.ps._check_trivial_response(email, act)
        assert penalty == pytest.approx(-0.30)

    def test_trivial_response_generic_phrase(self):
        for phrase in ["ok", "okay", "sure", "thanks", "noted", "acknowledged", "will do", "k", "fine"]:
            act = _action(action="reply", response=phrase)
            penalty = self.ps._check_trivial_response(_email(gold_action="reply"), act)
            assert penalty == pytest.approx(-0.30), f"Expected penalty for '{phrase}'"

    def test_no_trivial_penalty_for_ignore_action(self):
        act = _action(action="ignore", response="")
        penalty = self.ps._check_trivial_response(_email(), act)
        assert penalty == 0.0

    def test_no_trivial_penalty_for_substantial_response(self):
        act = _action(action="reply", response="Thank you for reaching out. I'll review this and respond shortly.")
        penalty = self.ps._check_trivial_response(_email(gold_action="reply"), act)
        assert penalty == 0.0


# ===========================================================================
# RewardEngine — component scorers
# ===========================================================================

class TestRewardEngineIntentScoring:
    def setup_method(self):
        self.engine = RewardEngine()

    def test_perfect_intent_match(self):
        score = self.engine._score_intents(["spam"], ["spam"])
        assert score > 0.99  # strictly < 1 but effectively perfect

    def test_zero_intent_match(self):
        score = self.engine._score_intents(["spam"], ["meeting_request"])
        assert score < 0.01  # strictly > 0 but effectively zero

    def test_partial_intent_f1(self):
        # predicted: spam, meeting_request  |  gold: spam
        # precision = 1/2, recall = 1/1  →  F1 = 2*(0.5*1)/(0.5+1) = 0.667
        score = self.engine._score_intents(["spam", "meeting_request"], ["spam"])
        assert pytest.approx(score, abs=0.01) == 2 / 3

    def test_empty_gold_accepts_any(self):
        score = self.engine._score_intents(["spam"], [])
        assert score == pytest.approx(0.5)

    def test_empty_predicted_returns_zero(self):
        score = self.engine._score_intents([], ["spam"])
        assert score < 0.01  # strictly > 0 but effectively zero

    def test_empty_gold_and_empty_pred_returns_one(self):
        score = self.engine._score_intents([], [])
        assert score > 0.99  # strictly < 1 but effectively perfect


class TestRewardEnginePriorityScoring:
    def setup_method(self):
        self.engine = RewardEngine()

    def test_exact_match_priority(self):
        assert self.engine._score_priority("medium", "medium") > 0.99

    def test_one_step_off_priority(self):
        assert self.engine._score_priority("high", "medium") == pytest.approx(0.4)
        assert self.engine._score_priority("low", "medium") == pytest.approx(0.4)

    def test_two_step_off_priority(self):
        assert self.engine._score_priority("critical", "low") < 0.01
        assert self.engine._score_priority("low", "high") < 0.01

    def test_invalid_priority_returns_zero(self):
        assert self.engine._score_priority("extreme", "high") < 0.01


class TestRewardEngineActionScoring:
    def setup_method(self):
        self.engine = RewardEngine()

    def test_exact_action_match(self):
        assert self.engine._score_action("reply", "reply") > 0.99
        assert self.engine._score_action("ignore", "ignore") > 0.99
        assert self.engine._score_action("escalate", "escalate") > 0.99
        assert self.engine._score_action("forward", "forward") > 0.99

    def test_partial_credit_related_actions(self):
        assert self.engine._score_action("escalate", "forward") == pytest.approx(0.3)
        assert self.engine._score_action("forward", "escalate") == pytest.approx(0.3)
        assert self.engine._score_action("reply", "escalate") == pytest.approx(0.3)
        assert self.engine._score_action("escalate", "reply") == pytest.approx(0.3)

    def test_zero_for_unrelated_actions(self):
        assert self.engine._score_action("ignore", "reply") < 0.01
        assert self.engine._score_action("reply", "ignore") < 0.01


class TestRewardEngineResponseScoring:
    def setup_method(self):
        # Use heuristic scorer (no ML model needed in tests)
        scorer = EmbeddingScorer()
        scorer._model = "UNAVAILABLE"   # force heuristic path
        self.engine = RewardEngine(embedding_scorer=scorer)

    def test_empty_gold_and_pred_scores_one(self):
        score = self.engine._score_response("", "", "ignore")
        assert score > 0.99  # strictly < 1 but effectively perfect

    def test_empty_pred_when_gold_expects_response_scores_zero(self):
        score = self.engine._score_response("", "Thank you for reaching out.", "reply")
        assert score < 0.01  # strictly > 0 but effectively zero

    def test_ignore_action_when_gold_response_expected_scores_zero(self):
        score = self.engine._score_response("some reply", "Thank you.", "ignore")
        assert score < 0.01  # strictly > 0 but effectively zero

    def test_noisy_response_when_none_expected_scores_low(self):
        # gold is empty, but agent generates text → penalised
        score = self.engine._score_response("Some unwanted text here.", "", "reply")
        assert score == pytest.approx(0.3)

    def test_similar_response_scores_nonzero(self):
        score = self.engine._score_response(
            "Thank you for reaching out.", "Thank you for your message.", "reply"
        )
        assert score > 0.0


# ===========================================================================
# RewardEngine — full composite reward
# ===========================================================================

class TestRewardEngineComposite:
    def setup_method(self):
        scorer = EmbeddingScorer()
        scorer._model = "UNAVAILABLE"
        self.engine = RewardEngine(embedding_scorer=scorer)

    def test_perfect_action_returns_high_reward(self):
        email = _email(
            gold_intents=["information_sharing"],
            gold_priority="medium",
            gold_action="reply",
            gold_response="Thank you for the update.",
        )
        action = _action(
            intents=["information_sharing"],
            priority="medium",
            action="reply",
            response="Thank you for the update.",
        )
        reward, breakdown = self.engine.compute(email, action)
        # Intent, priority, action all perfect → high reward (response via heuristic)
        assert reward > 0.6
        assert breakdown["intent"] > 0.99
        assert breakdown["priority"] > 0.99
        assert breakdown["action"] > 0.99
        assert breakdown["penalty"] == pytest.approx(0.0)

    def test_completely_wrong_action_returns_low_reward(self):
        email = _email(
            gold_intents=["meeting_request"],
            gold_priority="high",
            gold_action="escalate",
            gold_response="Escalating immediately.",
        )
        action = _action(
            intents=["spam"],       # wrong
            priority="low",         # wrong (3 steps off)
            action="ignore",        # wrong (unrelated)
            response="",            # wrong (should reply)
        )
        reward, breakdown = self.engine.compute(email, action, action_log=[])
        assert reward <= 0.3
        assert breakdown["intent"] < 0.01
        assert breakdown["priority"] < 0.01
        assert breakdown["action"] < 0.01

    def test_reward_clamped_to_minus_one_on_bad_action(self):
        email = _email(gold_priority="critical", gold_action="reply")
        # Ignoring critical email AND giving trivial response → stacked penalties
        action = _action(priority="low", action="ignore", response="ok")
        action_log = ["ignore", "ignore", "ignore"]   # also repeat penalty
        reward, _ = self.engine.compute(email, action, action_log=action_log)
        assert reward >= -1.0   # must not go below clamp

    def test_reward_clamped_to_plus_one(self):
        email = _email(
            gold_intents=["information_sharing"],
            gold_priority="medium",
            gold_action="reply",
            gold_response="Good morning.",
        )
        action = _action(
            intents=["information_sharing"],
            priority="medium",
            action="reply",
            response="Good morning.",
        )
        reward, _ = self.engine.compute(email, action)
        assert reward <= 1.0

    def test_breakdown_has_all_keys(self):
        email = _email()
        action = _action()
        _, breakdown = self.engine.compute(email, action)
        for key in ("intent", "priority", "action", "response", "penalty", "total"):
            assert key in breakdown

    def test_default_weights_sum_to_one(self):
        weights = RewardEngine.DEFAULT_WEIGHTS
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_custom_weights_applied(self):
        scorer = EmbeddingScorer()
        scorer._model = "UNAVAILABLE"
        engine = RewardEngine(weights={"intent": 0.5, "priority": 0.2, "action": 0.2, "response": 0.1},
                              embedding_scorer=scorer)
        assert engine.weights["intent"] == pytest.approx(0.5)
        assert engine.weights["response"] == pytest.approx(0.1)


# ===========================================================================
# EmbeddingScorer — heuristic fallback
# ===========================================================================

class TestEmbeddingScorerHeuristic:
    def setup_method(self):
        self.scorer = EmbeddingScorer()
        self.scorer._model = "UNAVAILABLE"   # force heuristic

    def test_identical_texts_score_close_to_one(self):
        score = self.scorer.score("Thank you for the update.", "Thank you for the update.")
        assert score > 0.9

    def test_completely_different_texts_score_low(self):
        score = self.scorer.score("pizza delivery tonight", "quarterly financial report")
        assert score < 0.3

    def test_empty_candidate_returns_zero(self):
        assert self.scorer.score("", "some reference") < 0.01

    def test_empty_reference_returns_zero(self):
        assert self.scorer.score("some candidate", "") < 0.01

    def test_score_bounded_zero_to_one(self):
        score = self.scorer.score("hello world", "hello there world")
        assert 0.0 <= score <= 1.0

    def test_multiple_references_takes_best(self):
        refs = ["bad reference xyz", "Thank you for the update."]
        score = self.scorer.score("Thank you for the update.", refs)
        # Should match the second reference well
        assert score > 0.5
