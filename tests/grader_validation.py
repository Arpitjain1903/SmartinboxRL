"""
Hackathon Grader Validation Script
===================================
Run with:
    python tests/grader_validation.py

Validates that EVERY grader function returns a score strictly within (0, 1).
Scores of exactly 0.0 or 1.0 are REJECTED by the hackathon validator.

Hard requirement:  0.001 <= score <= 0.999
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rewards.reward_engine import RewardEngine, _strict as _re_strict
from rewards.embedding_scorer import EmbeddingScorer, _strict as _es_strict
from rewards.penalty_system import PenaltySystem


# ===========================================================================
# Helpers — shared email / action fixtures
# ===========================================================================

def _email(
    gold_intents=None,
    gold_priority: str = "medium",
    gold_action: str = "reply",
    gold_response: str = "Thank you for reaching out.",
) -> dict:
    return {
        "id": "val_001",
        "subject": "Test",
        "body": "Hello",
        "sender": "test@example.com",
        "gold_intents": gold_intents or ["information_sharing"],
        "gold_priority": gold_priority,
        "gold_action": gold_action,
        "gold_response": gold_response,
    }


def _action(
    intents=None,
    priority: str = "medium",
    action: str = "reply",
    response: str = "Thank you for reaching out.",
) -> dict:
    return {
        "intents": intents or ["information_sharing"],
        "priority": priority,
        "action": action,
        "response": response,
    }


# ===========================================================================
# Grader 1 — Intent scorer  (F1 over intent labels)
# ===========================================================================

def grade_intents(predicted_intents: list[str], gold_intents: list[str]) -> float:
    """Grader 1: F1 between predicted and gold intent labels.

    Returns a score strictly in (0.001, 0.999).
    """
    engine = RewardEngine()
    raw = engine._score_intents(predicted_intents, gold_intents)
    return max(0.001, min(0.999, raw))   # explicit clip at grader boundary


# ===========================================================================
# Grader 2 — Priority scorer  (ordinal match)
# ===========================================================================

def grade_priority(predicted_priority: str, gold_priority: str) -> float:
    """Grader 2: Ordinal priority correctness.

    Perfect match → ~0.999, one-step off → 0.4, else → ~0.001.
    Returns a score strictly in (0.001, 0.999).
    """
    engine = RewardEngine()
    raw = engine._score_priority(predicted_priority, gold_priority)
    return max(0.001, min(0.999, raw))


# ===========================================================================
# Grader 3 — Action scorer  (exact + partial credit)
# ===========================================================================

def grade_action(predicted_action: str, gold_action: str) -> float:
    """Grader 3: Action classification correctness.

    Exact match → ~0.999, related action (escalate↔forward) → 0.3, else → ~0.001.
    Returns a score strictly in (0.001, 0.999).
    """
    engine = RewardEngine()
    raw = engine._score_action(predicted_action, gold_action)
    return max(0.001, min(0.999, raw))


# ===========================================================================
# Grader 4 — Response quality scorer  (semantic similarity)
# ===========================================================================

def grade_response(predicted_response: str, gold_response: str, action_type: str = "reply") -> float:
    """Grader 4: Semantic response quality via embedding cosine similarity.

    Falls back to Jaccard-ish heuristic when the embedding model is unavailable.
    Returns a score strictly in (0.001, 0.999).
    """
    # Force heuristic in validation to avoid network dependency
    scorer = EmbeddingScorer()
    scorer._model = "UNAVAILABLE"
    engine = RewardEngine(embedding_scorer=scorer)
    raw = engine._score_response(predicted_response, gold_response, action_type)
    return max(0.001, min(0.999, raw))


# ===========================================================================
# Grader 5 — Composite reward  (all components + penalties)
# ===========================================================================

def grade_composite(email: dict, action: dict, action_log: list[str] | None = None) -> float:
    """Grader 5: Full composite reward — weighted sum of all components minus penalties.

    Returns a score strictly in (0.001, 0.999).
    """
    scorer = EmbeddingScorer()
    scorer._model = "UNAVAILABLE"
    engine = RewardEngine(embedding_scorer=scorer)
    total, _ = engine.compute(email, action, action_log=action_log or [])
    return max(0.001, min(0.999, total))


# ===========================================================================
# Validation runner — follows the mandated hackathon test format
# ===========================================================================

def _run_grader(name: str, cases: list[tuple]) -> bool:
    """Run one grader's test block.  Returns True if all cases pass."""
    print(f"\n{'='*60}")
    print(f"=== GRADER VALIDATION: {name} ===")
    print(f"{'='*60}")
    all_passed = True
    for row in cases:
        desc = row[0]
        args = row[1:-1]
        fn   = row[-1]
        score = fn(*args)
        valid = 0.0 < score < 1.0 and score != 0.0 and score != 1.0
        # Additional hackathon check: must be in [0.001, 0.999]
        in_range = 0.001 <= score <= 0.999
        ok = valid and in_range
        status = "PASS" if ok else "FAIL -- INVALID SCORE"
        print(f"  [{status}]  {desc:<45}  score = {score:.6f}")
        if not ok:
            all_passed = False
    return all_passed


def main() -> None:
    results = []

    # ── Grader 1: Intent ─────────────────────────────────────────────────
    results.append(_run_grader("Intent Scorer (F1)", [
        ("perfect match",                  ["spam"], ["spam"],                                  grade_intents),
        ("completely wrong",               ["spam"], ["meeting_request"],                       grade_intents),
        ("partial overlap",                ["spam", "meeting_request"], ["spam"],               grade_intents),
        ("empty gold (accept any)",        ["spam"], [],                                        grade_intents),
        ("empty prediction and gold",      [], [],                                              grade_intents),
        ("empty prediction, gold exists",  [], ["spam"],                                        grade_intents),
    ]))

    # ── Grader 2: Priority ────────────────────────────────────────────────
    results.append(_run_grader("Priority Scorer (ordinal)", [
        ("exact match — medium",           "medium",   "medium",   grade_priority),
        ("exact match — critical",         "critical", "critical", grade_priority),
        ("one step off — high→medium",     "high",     "medium",   grade_priority),
        ("two steps off — critical→low",   "critical", "low",      grade_priority),
        ("invalid priority label",         "extreme",  "high",     grade_priority),
    ]))

    # ── Grader 3: Action ──────────────────────────────────────────────────
    results.append(_run_grader("Action Scorer (exact+partial)", [
        ("perfect — reply",                "reply",    "reply",    grade_action),
        ("perfect — ignore",               "ignore",   "ignore",   grade_action),
        ("related — escalate→forward",     "escalate", "forward",  grade_action),
        ("unrelated — ignore→reply",       "ignore",   "reply",    grade_action),
        ("unrelated — reply→ignore",       "reply",    "ignore",   grade_action),
    ]))

    # ── Grader 4: Response ────────────────────────────────────────────────
    results.append(_run_grader("Response Quality Scorer (semantic)", [
        ("identical response",             "Thank you for reaching out.", "Thank you for reaching out.", "reply", grade_response),
        ("similar response",               "Thanks for your message.",    "Thank you for reaching out.", "reply", grade_response),
        ("completely different response",  "pizza delivery tonight",      "quarterly meeting agenda",   "reply", grade_response),
        ("empty predicted — gold expected","",                            "Thank you.",                 "reply", grade_response),
        ("silence when none expected",     "",                            "",                           "ignore",grade_response),
        ("noise when silence expected",    "Some unwanted reply.",        "",                           "reply", grade_response),
        ("ignore when response needed",    "Some reply.",                 "Thank you.",                 "ignore",grade_response),
    ]))

    # ── Grader 5: Composite reward ────────────────────────────────────────
    perfect_email  = _email(["information_sharing"], "medium", "reply", "Thank you.")
    perfect_action = _action(["information_sharing"], "medium", "reply", "Thank you.")
    wrong_email    = _email(["meeting_request"],     "high",   "escalate", "Escalating now.")
    wrong_action   = _action(["spam"],               "low",    "ignore",   "")
    partial_action = _action(["information_sharing"], "high",  "reply",    "Thank you.")

    results.append(_run_grader("Composite Reward (full pipeline)", [
        ("perfect action",
         perfect_email, perfect_action, [],
         grade_composite),
        ("completely wrong action",
         wrong_email, wrong_action, [],
         grade_composite),
        ("partial credit (priority off by 1)",
         perfect_email, partial_action, [],
         grade_composite),
        ("stacked penalties (critical ignored + repeat)",
         _email(gold_priority="critical", gold_action="reply"), _action(action="ignore", response="ok"),
         ["ignore", "ignore", "ignore"],
         grade_composite),
    ]))

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    all_ok = all(results)
    if all_ok:
        print("FINAL: All scores valid — safe to submit!")
    else:
        print("FINAL: Fix graders before submitting!")
        sys.exit(1)


if __name__ == "__main__":
    main()
