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

from rewards.reward_engine import RewardEngine, safe_score, _strict as _re_strict
from rewards.embedding_scorer import EmbeddingScorer, safe_score as _es_safe_score, _strict as _es_strict
from rewards.penalty_system import PenaltySystem


# ===========================================================================
# Helpers - shared email / action fixtures
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
# Grader 1 - Intent scorer (F1 over intent labels)
# ===========================================================================

def grade_intents(predicted_intents: list[str], gold_intents: list[str]) -> float:
    """Grader 1: F1 between predicted and gold intent labels.

    Returns a score strictly in (0.001, 0.999).
    """
    engine = RewardEngine()
    raw = engine._score_intents(predicted_intents, gold_intents)
    return safe_score(raw)   # explicit clip with None/NaN safety


# ===========================================================================
# Grader 2 - Priority scorer (ordinal match)
# ===========================================================================

def grade_priority(predicted_priority: str, gold_priority: str) -> float:
    """Grader 2: Ordinal priority correctness.

    Perfect match -> ~0.999, one-step off -> 0.4, else -> ~0.001.
    Returns a score strictly in (0.001, 0.999).
    """
    engine = RewardEngine()
    raw = engine._score_priority(predicted_priority, gold_priority)
    return safe_score(raw)


# ===========================================================================
# Grader 3 - Action scorer (exact + partial credit)
# ===========================================================================

def grade_action(predicted_action: str, gold_action: str) -> float:
    """Grader 3: Action classification correctness.

    Exact match -> ~0.999, related action (escalate <-> forward) -> 0.3, else -> ~0.001.
    Returns a score strictly in (0.001, 0.999).
    """
    engine = RewardEngine()
    raw = engine._score_action(predicted_action, gold_action)
    return safe_score(raw)


# ===========================================================================
# Grader 4 - Response scorer (semantic similarity)
# ===========================================================================

def grade_response(predicted: str, gold: str, action_type: str) -> float:
    """Grader 4: Semantic similarity of drafted response.

    Calculated via cosine similarity or heuristic fallback.
    Returns a score strictly in (0.001, 0.999).
    """
    engine = RewardEngine()
    raw = engine._score_response(predicted, gold, action_type)
    return safe_score(raw)


# ===========================================================================
# Grader 5 - Composite Reward (full pipeline)
# ===========================================================================

def grade_composite(email: dict, action: dict, log: list[str]) -> float:
    """Grader 5: Full composite reward for a single step.

    Returns the weighted combination of the above scores, minus penalties.
    Final total is strictly clipped to (0.001, 0.999).
    """
    engine = RewardEngine()
    score, _ = engine.compute(email, action, action_log=log)
    return score


# ===========================================================================
# Validation Runner
# ===========================================================================

def _run_grader(name: str, cases: list) -> bool:
    print(f"\n{'='*60}")
    print(f"=== GRADER VALIDATION: {name} ===")
    print(f"{'='*60}")
    
    all_ok = True
    for case in cases:
        desc = case[0]
        func = case[-1]
        args = case[1:-1]
        
        try:
            score = func(*args)
            
            # Boundary check: Strictly in (0, 1) AND in [0.01, 0.99]
            if score < 0.01 or score > 0.99:
                status = "FAIL"
                all_ok = False
            else:
                status = "PASS"
            
            print(f"  [{status}]  {desc:<45}  score = {score:.6f}")
        except Exception as e:
            print(f"  [ERROR] {desc:<45}  {e}")
            all_ok = False
            
    return all_ok


def main():
    results = []
    engine = RewardEngine()

    # -- Grader 1: Intents ------------------------------------------------
    results.append(_run_grader("Intent Scorer (F1)", [
        ("perfect match",                  ["spam"], ["spam"],                                  grade_intents),
        ("completely wrong",               ["spam"], ["meeting_request"],                       grade_intents),
        ("partial overlap",                ["spam", "meeting_request"], ["spam"],               grade_intents),
        ("empty gold (accept any)",        ["spam"], [],                                        grade_intents),
        ("empty prediction and gold",      [], [],                                              grade_intents),
        ("empty prediction, gold exists",  [], ["spam"],                                        grade_intents),
    ]))

    # -- Grader 2: Priority ------------------------------------------------
    results.append(_run_grader("Priority Scorer (ordinal)", [
        ("exact match - medium",           "medium",   "medium",   grade_priority),
        ("exact match - critical",         "critical", "critical", grade_priority),
        ("one step off - high to medium",  "high",     "medium",   grade_priority),
        ("two steps off - critical to low","critical", "low",      grade_priority),
        ("invalid priority label",         "extreme",  "high",     grade_priority),
    ]))

    # -- Grader 3: Action --------------------------------------------------
    results.append(_run_grader("Action Scorer (exact+partial)", [
        ("perfect - reply",                "reply",    "reply",    grade_action),
        ("perfect - ignore",               "ignore",   "ignore",   grade_action),
        ("related - escalate to forward",  "escalate", "forward",  grade_action),
        ("unrelated - ignore to reply",    "ignore",   "reply",    grade_action),
        ("unrelated - reply to ignore",    "reply",    "ignore",   grade_action),
    ]))

    # -- Grader 4: Response ------------------------------------------------
    results.append(_run_grader("Response Quality Scorer (semantic)", [
        ("identical response",             "Thank you for reaching out.", "Thank you for reaching out.", "reply", grade_response),
        ("similar response",               "Thanks for your message.",    "Thank you for reaching out.", "reply", grade_response),
        ("completely different response",  "pizza delivery tonight",      "quarterly meeting agenda",   "reply", grade_response),
        ("empty predicted - gold expected","",                            "Thank you.",                 "reply", grade_response),
        ("silence when none expected",     "",                            "",                           "ignore",grade_response),
        ("noise when silence expected",    "Some unwanted reply.",        "",                           "reply", grade_response),
        ("ignore when response needed",    "Some reply.",                 "Thank you.",                 "ignore",grade_response),
    ]))

    # -- Grader 5: Composite reward ----------------------------------------
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

    # -- Summary ------------------------------------------------------------
    print("\n" + "="*60)
    all_ok = all(results)
    if all_ok:
        print("FINAL: All scores valid - safe to submit!")
    else:
        print("FINAL: Fix graders before submitting!")
        sys.exit(1)


if __name__ == "__main__":
    main()
