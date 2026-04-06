"""Tests for environment/action_space.py — validate_action() and constants."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.action_space import validate_action, VALID_ACTIONS, VALID_PRIORITIES, VALID_INTENTS


# ---------------------------------------------------------------------------
# validate_action — happy paths
# ---------------------------------------------------------------------------

def test_validate_action_minimal_valid():
    action = {"intents": ["spam"], "priority": "low", "action": "ignore", "response": ""}
    result = validate_action(action)
    assert result == {"intents": ["spam"], "priority": "low", "action": "ignore", "response": ""}


def test_validate_action_normalizes_case():
    action = {"intents": ["SPAM"], "priority": "HIGH", "action": "REPLY", "response": "OK"}
    result = validate_action(action)
    assert result["priority"] == "high"
    assert result["action"] == "reply"
    assert "spam" in result["intents"]


def test_validate_action_strips_whitespace():
    action = {"intents": ["  spam  "], "priority": " medium ", "action": " reply ", "response": "Hi"}
    result = validate_action(action)
    assert result["intents"] == ["spam"]
    assert result["priority"] == "medium"
    assert result["action"] == "reply"


def test_validate_action_intents_as_string_becomes_list():
    action = {"intents": "spam", "priority": "low", "action": "ignore", "response": ""}
    result = validate_action(action)
    assert result["intents"] == ["spam"]


def test_validate_action_multiple_intents():
    action = {
        "intents": ["spam", "meeting_request"],
        "priority": "high",
        "action": "escalate",
        "response": "Escalating.",
    }
    result = validate_action(action)
    assert set(result["intents"]) == {"spam", "meeting_request"}


def test_validate_action_all_valid_actions():
    for act in VALID_ACTIONS:
        result = validate_action({"intents": ["spam"], "priority": "low", "action": act, "response": ""})
        assert result["action"] == act


def test_validate_action_all_valid_priorities():
    for p in VALID_PRIORITIES:
        result = validate_action({"intents": ["spam"], "priority": p, "action": "ignore", "response": ""})
        assert result["priority"] == p


# ---------------------------------------------------------------------------
# validate_action — error paths
# ---------------------------------------------------------------------------

def test_validate_action_not_dict_raises_type_error():
    with pytest.raises(TypeError):
        validate_action("reply")


def test_validate_action_empty_intents_raises():
    with pytest.raises(ValueError, match="intent"):
        validate_action({"intents": [], "priority": "low", "action": "reply", "response": ""})


def test_validate_action_blank_intent_strings_raises():
    with pytest.raises(ValueError, match="intent"):
        validate_action({"intents": ["  ", ""], "priority": "low", "action": "reply", "response": ""})


def test_validate_action_bad_priority_raises():
    with pytest.raises(ValueError, match="priority"):
        validate_action({"intents": ["spam"], "priority": "extreme", "action": "reply", "response": ""})


def test_validate_action_bad_action_raises():
    with pytest.raises(ValueError, match="action"):
        validate_action({"intents": ["spam"], "priority": "low", "action": "delete", "response": ""})


def test_validate_action_missing_keys_uses_defaults():
    # Missing priority and action → defaults applied
    result = validate_action({"intents": ["spam"]})
    assert result["priority"] == "medium"
    assert result["action"] == "reply"
    assert result["response"] == ""


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

def test_valid_intents_count():
    assert len(VALID_INTENTS) == 14


def test_valid_actions_set():
    assert VALID_ACTIONS == {"reply", "ignore", "escalate", "forward"}


def test_valid_priorities_tuple():
    assert VALID_PRIORITIES == ("low", "medium", "high", "critical")
