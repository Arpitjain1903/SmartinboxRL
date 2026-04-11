import pytest
from rewards.reward_engine import RewardEngine
from rewards.embedding_scorer import EmbeddingScorer
from rewards.penalty_system import PenaltySystem

def test_score_bounds():
    # Initialize components
    scorer = EmbeddingScorer()
    penalty = PenaltySystem()
    engine = RewardEngine(embedding_scorer=scorer, penalty_system=penalty)
    
    # 1. Edge-case inputs: Empty/Null logic
    empty_email = {}
    empty_action = {"intents": [], "priority": "medium", "action": "reply", "response": ""}
    
    # 2. Perfect inputs
    perfect_email = {
        "gold_intents": ["spam"],
        "gold_priority": "high",
        "gold_action": "escalate",
        "gold_response": "This is perfect."
    }
    perfect_action = {
        "intents": ["spam"],
        "priority": "high",
        "action": "escalate",
        "response": "This is perfect."
    }
    
    # 3. All-wrong inputs
    wrong_email = {
        "gold_intents": ["meeting_request"],
        "gold_priority": "low",
        "gold_action": "ignore",
        "gold_response": ""
    }
    wrong_action = {
        "intents": ["spam"],
        "priority": "critical",
        "action": "reply",
        "response": "Some weird response."
    }
    
    cases = [
        ("Empty", empty_email, empty_action),
        ("Perfect", perfect_email, perfect_action),
        ("All-wrong", wrong_email, wrong_action),
        ("None-Edge", {"gold_intents": None}, {"intents": None, "priority": None, "action": None, "response": None}),
    ]
    
    for case_name, em, ac in cases:
        score, breakdown = engine.compute(em, ac, action_log=[])
        
        # Assert overall score is bounded strictly between 0 and 1
        assert 0.0 < score < 1.0, f"SCORE OUT OF BOUNDS: {score} from RewardEngine in case {case_name}"
        
        # Component scores must also be bounded between 0 and 1
        # Penalty is an addend, but intent/priority/action/response must be in (0,1)
        for comp in ["intent", "priority", "action", "response"]:
            if comp in breakdown:
                comp_score = breakdown[comp]
                assert 0.0 < comp_score < 1.0, f"{comp.upper()} SCORE OUT OF BOUNDS: {comp_score} in case {case_name}"

def test_embedding_scorer_bounds():
    scorer = EmbeddingScorer()
    # Test identical, completely wrong, and missing strings
    test_cases = [
        ("Identical", "hello world", "hello world"),
        ("Completely wrong", "pizza", "car"),
        ("Empty candidate", "", "hello"),
        ("Empty reference", "hello", ""),
        ("Both Empty", "", ""),
    ]
    
    for name, cand, ref in test_cases:
        score = scorer.score(cand, ref)
        assert 0.0 < score < 1.0, f"SCORE OUT OF BOUNDS: {score} from EmbeddingScorer in case {name}"
