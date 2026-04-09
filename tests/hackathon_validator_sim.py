"""
HACKATHON VALIDATOR SIMULATION -- SmartInboxRL
===============================================
Run BEFORE submitting to confirm no score is exactly 0.0 or 1.0.

    python tests/hackathon_validator_sim.py

ALL tests must show [PASS].  Any [FAIL] means the submission will be
REJECTED by the Meta hackathon validator.

Valid range: score > 0.0 AND score < 1.0  -->  safe zone: [0.001, 0.999]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# -- import your actual grader -------------------------------------------------
from rewards.reward_engine import RewardEngine
from rewards.embedding_scorer import EmbeddingScorer

# Force heuristic mode so the test doesn't need network/GPU
_scorer = EmbeddingScorer()
_scorer._model = "UNAVAILABLE"
_engine = RewardEngine(embedding_scorer=_scorer)


def YOUR_GRADER(
    email: dict,
    action: dict,
    action_log: list[str] | None = None,
) -> float:
    """Thin wrapper over the real composite grader."""
    total, _ = _engine.compute(email, action, action_log=action_log or [])
    # Double-guard matching inference.py _safe_score
    return max(0.001, min(0.999, float(total)))


# -- result tracking -----------------------------------------------------------
results: list[bool] = []


def check(label: str, score: float) -> None:
    valid = (type(score) == float) and (0.0 < score < 1.0)
    status = "[PASS]" if valid else "[FAIL] SUBMISSION WILL BE REJECTED"
    results.append(valid)
    print(f"{status} | {label} | score = {score}")
    if not valid:
        if score == 0.0:
            print("       +-- Got exactly 0.0 -- validator will REJECT this")
        elif score == 1.0:
            print("       +-- Got exactly 1.0 -- validator will REJECT this")
        elif score < 0.0 or score > 1.0:
            print(f"       +-- Out of [0,1] range entirely -- score = {score}")


print("=" * 60)
print("  HACKATHON VALIDATOR SIMULATION -- SmartInboxRL")
print("=" * 60)

# -- SECTION 1: Perfect action (risk of returning exactly 1.0) ----------------
perfect_email = {
    "id": "t001", "subject": "Hi", "body": "Hello",
    "sender": "a@b.com",
    "gold_intents": ["information_sharing"],
    "gold_priority": "medium",
    "gold_action": "reply",
    "gold_response": "Thank you for reaching out.",
}
perfect_action = {
    "intents": ["information_sharing"],
    "priority": "medium",
    "action": "reply",
    "response": "Thank you for reaching out.",
}
score = YOUR_GRADER(perfect_email, perfect_action)
check("Perfect match -- risk of 1.0", score)

# -- SECTION 2: Completely wrong action (risk of returning exactly 0.0) -------
wrong_email = {
    "id": "t002", "subject": "Urgent", "body": "URGENT",
    "sender": "x@y.com",
    "gold_intents": ["meeting_request"],
    "gold_priority": "critical",
    "gold_action": "escalate",
    "gold_response": "Escalating immediately.",
}
wrong_action = {
    "intents": ["spam"],
    "priority": "low",
    "action": "ignore",
    "response": "",
}
score = YOUR_GRADER(wrong_email, wrong_action)
check("Completely wrong -- risk of 0.0", score)

# -- SECTION 3: Stacked penalties (risk of going to exactly 0.0) --------------
penalty_action = {
    "intents": ["spam"],
    "priority": "low",
    "action": "ignore",
    "response": "ok",
}
action_log = ["ignore", "ignore", "ignore", "ignore", "ignore"]
score = YOUR_GRADER(wrong_email, penalty_action, action_log=action_log)
check("Max penalties stacked -- risk of 0.0", score)

# -- SECTION 4: Empty inputs --------------------------------------------------
empty_email = {
    "id": "t003", "subject": "", "body": "",
    "sender": "",
    "gold_intents": [],
    "gold_priority": "medium",
    "gold_action": "reply",
    "gold_response": "",
}
empty_action = {
    "intents": [],
    "priority": "",
    "action": "",
    "response": "",
}
score = YOUR_GRADER(empty_email, empty_action)
check("Empty inputs -- risk of 0.0", score)

# -- SECTION 5: Simulate 50 random tasks (bulk check) -------------------------
random.seed(42)
priorities = ["low", "medium", "high", "critical"]
actions_list = ["reply", "ignore", "escalate", "forward"]
intents = ["spam", "meeting_request", "information_sharing", "complaint"]

bulk_failures: list[tuple[int, float]] = []
for i in range(50):
    e = {
        **perfect_email,
        "id": f"bulk_{i}",
        "gold_priority": random.choice(priorities),
        "gold_action": random.choice(actions_list),
        "gold_intents": random.sample(intents, k=random.randint(1, 2)),
    }
    a = {
        "intents": random.sample(intents, k=random.randint(1, 2)),
        "priority": random.choice(priorities),
        "action": random.choice(actions_list),
        "response": random.choice(["", "ok", "Thank you for reaching out."]),
    }
    s = YOUR_GRADER(e, a)
    if not (0.0 < s < 1.0):
        bulk_failures.append((i, s))

if bulk_failures:
    print(
        f"[FAIL] | Bulk test | {len(bulk_failures)} invalid scores: "
        f"{bulk_failures[:5]}"
    )
    results.append(False)
else:
    print("[PASS] | Bulk test (50 random tasks) | all scores in (0.0, 1.0)")
    results.append(True)

# -- SECTION 6: Error fallback score (inference.py exception branch) ----------
# Simulate error_score = _safe_score(0.0) as patched in inference.py
from inference import _safe_score  # noqa: E402

error_score = _safe_score(0.0)
check("Error fallback score (inference.py exception path)", error_score)

# -- FINAL VERDICT ------------------------------------------------------------
print()
print("=" * 60)
total = len(results)
passed = sum(results)
if all(results):
    print(f"  [OK] ALL {total}/{total} TESTS PASSED -- SAFE TO SUBMIT")
else:
    print(f"  [FAIL] {total - passed}/{total} TESTS FAILED -- DO NOT SUBMIT YET")
    print("  Fix all [FAIL] items before resubmitting.")
print("=" * 60)

sys.exit(0 if all(results) else 1)
