"""LLM-backed baseline agent for SmartInboxRL.

Uses any OpenAI-compatible API endpoint (configured via environment variables)
to extract intents, priority, action, and response from emails.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from agents.base_agent import BaseAgent
from environment.action_space import VALID_INTENTS, VALID_PRIORITIES, VALID_ACTIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an intelligent email assistant. Given an email, analyse it and respond \
with a JSON object containing exactly these keys:

1. "intents" — a list of intent labels detected in the email.
   Valid labels: {intents}

2. "priority" — the urgency level of this email.
   Valid values: {priorities}

3. "action" — what should be done with this email.
   Valid values: {actions}

4. "response" — your reply text. If the action is "ignore", \
set this to an empty string.

Return ONLY valid JSON. No explanation, no markdown fences.
""".format(
    intents=", ".join(VALID_INTENTS),
    priorities=", ".join(VALID_PRIORITIES),
    actions=", ".join(VALID_ACTIONS),
)


def _build_user_prompt(observation: dict[str, Any]) -> str:
    email = observation.get("email", {})
    history = observation.get("history", [])

    parts = [
        f"From: {email.get('sender', 'unknown')}",
        f"Subject: {email.get('subject', '(no subject)')}",
        f"\n{email.get('body', '')}",
    ]

    if history:
        parts.append("\n--- Recent interaction history ---")
        for h in history[-3:]:
            parts.append(
                f"Step {h.get('step')}: action={h.get('action')}, "
                f"priority={h.get('priority')}, intents={h.get('intents')}"
            )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class LLMAgent(BaseAgent):
    """Agent backed by an OpenAI-compatible LLM API.

    Environment variables
    ---------------------
    API_BASE_URL : str
        API endpoint (default: ``https://api.openai.com/v1``).
    MODEL_NAME : str
        Model identifier (default: ``gpt-4o-mini``).
    OPENAI_API_KEY : str
        API bearer token.
    """

    name = "llm"

    def __init__(
        self,
        api_base: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
    ):
        self.api_base = api_base or os.getenv(
            "API_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy-initialise the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    base_url=self.api_base,
                    api_key=self.api_key,
                )
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._client

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        user_prompt = _build_user_prompt(observation)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM API call failed: %s", exc)
            return self._fallback(observation)

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Best-effort JSON extraction from the LLM output."""
        try:
            # First try finding a json code block
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
            if match:
                cleaned = match.group(1)
            else:
                # If no code block, try to find {...}
                match = re.search(r"(\{.*\})", raw, re.DOTALL)
                if match:
                    cleaned = match.group(1)
                else:
                    cleaned = raw

            cleaned = cleaned.strip()
            data = json.loads(cleaned)
        except Exception as exc:
            logger.error("JSON parsing failed: %s", exc)
            return LLMAgent._fallback_action()

        # Validate / normalise
        intents = data.get("intents", ["information_sharing"])
        if isinstance(intents, str):
            intents = [intents]

        priority = str(data.get("priority", "medium")).lower()
        if priority not in VALID_PRIORITIES:
            priority = "medium"

        action = str(data.get("action", "reply")).lower()
        if action not in VALID_ACTIONS:
            action = "reply"

        response = str(data.get("response", ""))

        return {
            "intents": intents,
            "priority": priority,
            "action": action,
            "response": response,
        }

    @staticmethod
    def _fallback_action() -> dict[str, Any]:
        return {
            "intents": ["information_sharing"],
            "priority": "medium",
            "action": "reply",
            "response": "Thank you for your email. I'll review this and follow up.",
        }

    def _fallback(self, observation: dict[str, Any]) -> dict[str, Any]:
        logger.warning("Using fallback action for step %s", observation.get("step"))
        return self._fallback_action()
