import pytest
import numpy as np
from rewards.reward_engine import RewardEngine
from rewards.embedding_scorer import EmbeddingScorer

def check_bound(score, grader_class, method_name, inputs):
    assert 0.0 < score < 1.0, f"SCORE OUT OF BOUNDS: {score} from {grader_class}.{method_name} with inputs {inputs}"

def test_engine_compute_bounds():
    engine = RewardEngine()
    grader_class = "RewardEngine"
    method_name = "compute"

    # Edge-case inputs
    emails = [
        {"id": "1", "gold_intents": [], "gold_priority": "medium", "gold_action": "reply", "gold_response": ""},
        {},
        {"gold_intents": ["spam"], "gold_priority": "critical", "gold_action": "escalate", "gold_response": "x" * 1000},
        None,
    ]
    actions = [
        {"intents": [], "priority": "medium", "action": "reply", "response": ""},
        {"intents": [], "priority": "low", "action": "ignore", "response": ""},
        {"intents": ["spam"], "priority": "low", "action": "ignore", "response": "y" * 1000},
        None,
    ]

    for email in emails:
        for action in actions:
            # Bypass None directly since the environment creates dicts
            e = email if email is not None else {}
            a = action if action is not None else {"intents": [], "priority": "low", "action": "ignore", "response": ""}
            score, _ = engine.compute(e, a, [])
            check_bound(score, grader_class, method_name, f"email={e}, action={a}")

def test_engine_intent_bounds():
    engine = RewardEngine()
    edge_cases = [
        ([], []),
        (["spam"], ["spam"]),
        (["spam"], ["info"]),
        ([], ["spam"]),
        (["spam"], []),
        (None, None),
    ]
    for p, g in edge_cases:
        p_safe = p if p is not None else []
        g_safe = g if g is not None else []
        score = engine._score_intents(p_safe, g_safe)
        check_bound(score, "RewardEngine", "_score_intents", f"p={p}, g={g}")

def test_engine_priority_bounds():
    engine = RewardEngine()
    edge_cases = [
        ("low", "low"),
        ("low", "critical"),
        ("critical", "low"),
        ("", ""),
        ("unknown", "low"),
        (None, None),
    ]
    for p, g in edge_cases:
        score = engine._score_priority(str(p), str(g))
        check_bound(score, "RewardEngine", "_score_priority", f"p={p}, g={g}")

def test_engine_action_bounds():
    engine = RewardEngine()
    edge_cases = [
        ("reply", "reply"),
        ("reply", "ignore"),
        ("escalate", "forward"), # related
        ("", ""),
        ("unknown", "reply"),
        (None, None),
    ]
    for p, g in edge_cases:
        score = engine._score_action(str(p), str(g))
        check_bound(score, "RewardEngine", "_score_action", f"p={p}, g={g}")

def test_embedding_scorer_bounds():
    scorer = EmbeddingScorer()
    edge_cases = [
        ("", [""]),
        ("exact match", ["exact match"]),
        ("completely different", ["totally unrelated string text here"]),
        ("a", ["b", "c", "d"]),
        ("", []),
    ]
    for p, g in edge_cases:
        score = scorer.score(p, g)
        check_bound(score, "EmbeddingScorer", "score", f"p={p}, g={g}")

def test_extreme_numerical_bounds():
    from utils import safe_score

    extreme_inputs = [
        0, 1, -1, 0.0, 1.0, -0.0, 
        99999999999, -99999999999,
        np.inf, -np.inf, np.nan,
        None
    ]

    for val in extreme_inputs:
        score = safe_score(val)
        assert 0.0 < score < 1.0, f"SCORE OUT OF BOUNDS: {score} from safe_score with input {val}"
