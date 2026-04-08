"""Rule-based keyword heuristic agent for SmartInboxRL.

A simple deterministic agent that scans email text for keyword signals
to decide intents, priority, action, and canned responses.  Establishes
a non-ML performance baseline.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Keyword → signal maps
# ---------------------------------------------------------------------------

_SPAM_SIGNALS = {
    "congratulations", "winner", "lottery", "prize", "click here",
    "limited time", "act now", "free", "unsubscribe", "viagra",
    "no prescription", "discount", "deal", "offer expires",
}

_URGENCY_SIGNALS = {
    "urgent", "asap", "immediately", "critical", "emergency",
    "right away", "blocking", "deadline", "cob today", "eod",
    "time-sensitive", "high priority",
}

_MEETING_SIGNALS = {
    "meeting", "standup", "sync", "call", "conference", "1:1",
    "schedule", "calendar", "agenda",
}

_TASK_SIGNALS = {
    "assign", "please do", "can you", "need you to", "action item",
    "todo", "task", "own this", "take care of", "responsible",
}

_QUESTION_SIGNALS = {"?", "can you", "could you", "do you", "what", "how", "when", "where", "who"}

_ESCALATION_SIGNALS = {
    "escalate", "incident", "outage", "down", "breach", "security",
    "phishing", "vulnerability", "compliance", "audit", "cve-",
}


def _count_signals(text: str, signals: set[str]) -> int:
    text_l = text.lower()
    return sum(1 for s in signals if s in text_l)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RuleAgent(BaseAgent):
    """Keyword-heuristic agent producing deterministic decisions."""

    name = "rule"

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        email = observation.get("email", {})
        subject = email.get("subject", "")
        body = email.get("body", "")
        full_text = f"{subject} {body}"

        intents = self._detect_intents(full_text)
        priority = self._detect_priority(full_text)
        action, response = self._decide_action(full_text, intents, priority)

        return {
            "intents": intents,
            "priority": priority,
            "action": action,
            "response": response,
        }

    # ---- intent detection ----

    @staticmethod
    def _detect_intents(text: str) -> list[str]:
        intents: list[str] = []

        if _count_signals(text, _SPAM_SIGNALS) >= 2:
            intents.append("spam")
        if _count_signals(text, _MEETING_SIGNALS) >= 1:
            intents.append("meeting_request")
        if _count_signals(text, _TASK_SIGNALS) >= 1:
            intents.append("task_assignment")
        if _count_signals(text, _QUESTION_SIGNALS) >= 1:
            intents.append("question")
        if _count_signals(text, _ESCALATION_SIGNALS) >= 2:
            intents.append("urgent_request")
        if any(w in text.lower() for w in ("follow up", "following up", "per our last")):
            intents.append("follow_up")

        # fallback
        if not intents:
            intents.append("information_sharing")

        return intents

    # ---- priority ----

    @staticmethod
    def _detect_priority(text: str) -> str:
        urgency = _count_signals(text, _URGENCY_SIGNALS)
        escalation = _count_signals(text, _ESCALATION_SIGNALS)

        if urgency >= 2 or escalation >= 2:
            return "critical"
        if urgency >= 1 or escalation >= 1:
            return "high"
        if "?" in text:
            return "medium"
        return "low"

    # ---- action decision ----

    @staticmethod
    def _decide_action(
        text: str, intents: list[str], priority: str
    ) -> tuple[str, str]:
        text_l = text.lower()

        # Spam → ignore
        if "spam" in intents:
            return "ignore", ""

        # Security / incident → escalate
        if _count_signals(text_l, _ESCALATION_SIGNALS) >= 2:
            return (
                "escalate",
                "Subject: Escalation: Immediate Attention Required\n\nHi team,\n\nEscalating this to the appropriate team for immediate attention.\n\nBest regards,\nAI Assistant",
            )

        # Critical priority → escalate
        if priority == "critical":
            return (
                "escalate",
                "Subject: Escalation: Critical Priority\n\nHi team,\n\nThis looks critical. I'm escalating this now and will follow up shortly.\n\nBest regards,\nAI Assistant",
            )

        # Newsletter / FYI → forward or ignore
        keywords_fwd = {"fyi", "newsletter", "digest", "forwarding"}
        if any(k in text_l for k in keywords_fwd):
            return (
                "forward",
                "Subject: Fwd: FYI / Newsletter\n\nHi team,\n\nForwarding this to the relevant team members.\n\nBest regards,\nAI Assistant",
            )

        # Questions or tasks → reply
        if "question" in intents or "task_assignment" in intents:
            return (
                "reply",
                "Subject: Re: Your Request\n\nHi there,\n\nThanks for reaching out. I'll review this and get back to you with a detailed response shortly.\n\nBest regards,\nAI Assistant",
            )

        # Meeting → reply
        if "meeting_request" in intents:
            return (
                "reply",
                "Subject: Re: Meeting Request\n\nHi there,\n\nThanks for the invite. I'll confirm my availability and get back to you.\n\nBest regards,\nAI Assistant",
            )

        # Default → reply
        return (
            "reply",
            "Subject: Re: Your Message\n\nHi there,\n\nThank you for the message. I'll review this and follow up accordingly.\n\nBest regards,\nAI Assistant",
        )
