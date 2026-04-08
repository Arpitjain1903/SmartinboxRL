"""LLM-backed RL agent that learns from experience via SQLite memory.

Extends the base LLM agent pattern with:
  - SQLite-backed strategy memory (loaded at init from data/agent_memory.db)
  - Dynamic system prompt enriched with high-reward / low-reward patterns
  - Experience tracking and pattern extraction
  - Graceful degradation when DB or API key is missing

Requires OPENAI_API_KEY for LLM calls; falls back to rule-based if missing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from environment.action_space import VALID_INTENTS, VALID_PRIORITIES, VALID_ACTIONS

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "agent_memory.db"
STATE_PATH = DATA_DIR / "agent_state.json"


# ---------------------------------------------------------------------------
# Base prompt (used when no experience data exists)
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
You are an intelligent email assistant. Given an email, analyse it and respond \
with a JSON object containing exactly these keys:

1. "intents" — a list of intent labels detected in the email.
   Valid labels: {intents}

2. "priority" — the urgency level of this email.
   Valid values: {priorities}

3. "action" — what should be done with this email.
   Valid values: {actions}

4. "response" — your reply text. If the action is "ignore", \
set this to an empty string. IMPORTANT: If replying, the response MUST be a \
professional valid email format including: a Subject line, a Salutation, the \
message body, and a formal Closing (e.g. "Best regards, AI Assistant"). \
Use \\n characters to format the email lines properly within the JSON string.

Return ONLY valid JSON. No explanation, no markdown fences.
""".format(
    intents=", ".join(VALID_INTENTS),
    priorities=", ".join(VALID_PRIORITIES),
    actions=", ".join(VALID_ACTIONS),
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RLAgent(BaseAgent):
    """LLM agent enriched with learned patterns from SQLite experience DB."""

    name = "rl_llm"

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

        # ── Load learned patterns from SQLite ──
        self.top_patterns: List[Dict[str, Any]] = []
        self.bad_patterns: List[Dict[str, Any]] = []
        self.total_experiences: int = 0
        self.overall_avg_reward: float = 0.0
        self.learned_patterns: Dict[str, Any] = {}

        try:
            if DB_PATH.exists():
                conn = sqlite3.connect(str(DB_PATH))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Check that strategy_memory table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_memory'"
                )
                if cursor.fetchone():
                    # Top performing patterns
                    cursor.execute("""
                        SELECT pattern, best_action, best_priority, avg_reward, times_seen
                        FROM strategy_memory
                        WHERE times_seen >= 3
                        ORDER BY avg_reward DESC
                        LIMIT 10
                    """)
                    self.top_patterns = [dict(row) for row in cursor.fetchall()]

                    # Poorly performing patterns to avoid
                    cursor.execute("""
                        SELECT pattern, best_action, avg_reward
                        FROM strategy_memory
                        WHERE avg_reward < 0.50
                        ORDER BY avg_reward ASC
                        LIMIT 5
                    """)
                    self.bad_patterns = [dict(row) for row in cursor.fetchall()]

                # Experience counts
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='experiences'"
                )
                if cursor.fetchone():
                    cursor.execute("SELECT COUNT(*), AVG(total_reward) FROM experiences")
                    row = cursor.fetchone()
                    if row:
                        self.total_experiences = int(row[0] or 0)
                        self.overall_avg_reward = float(row[1] or 0.0)

                conn.close()
        except Exception as e:
            logger.warning("Could not load agent memory DB: %s", e)
            self.top_patterns = []
            self.bad_patterns = []
            self.total_experiences = 0
            self.overall_avg_reward = 0.0

        # Load agent_state.json for any additional patterns
        try:
            if STATE_PATH.exists():
                with open(STATE_PATH, "r") as f:
                    state_data = json.load(f)
                self.learned_patterns = state_data.get("learned_patterns", {})
        except Exception:
            self.learned_patterns = {}

    # ------------------------------------------------------------------
    # Dynamic system prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build system prompt enriched with learned patterns."""
        if self.total_experiences == 0:
            return _BASE_SYSTEM_PROMPT

        # Build enriched prompt with learned patterns
        parts = [
            f"You are an email triage expert. You have learned from {self.total_experiences} emails.",
            f"Current performance: {self.overall_avg_reward:.2f}/1.00",
            "",
        ]

        if self.top_patterns:
            parts.append("LEARNED HIGH-REWARD PATTERNS — follow these:")
            for p in self.top_patterns:
                parts.append(
                    f"- When email contains '{p['pattern']}' → action='{p['best_action']}', "
                    f"priority='{p.get('best_priority', 'medium')}' "
                    f"(worked {p['times_seen']}x, avg reward={p['avg_reward']:.2f})"
                )
            parts.append("")

        if self.bad_patterns:
            parts.append("LEARNED MISTAKES — avoid these:")
            for p in self.bad_patterns:
                parts.append(
                    f"- When email contains '{p['pattern']}' → DO NOT use action='{p['best_action']}' "
                    f"(avg reward={p['avg_reward']:.2f} — this performs poorly)"
                )
            parts.append("")

        parts.append(
            'Respond ONLY with valid JSON:\n'
            '{"intents": [...], "priority": "...", "action": "...", "response": "..."}'
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # OpenAI client
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # act()
    # ------------------------------------------------------------------

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Produce an action using the LLM with experience-enriched prompt."""
        # If no API key, use rule-based fallback
        if not self.api_key:
            return self._rule_fallback(observation)

        user_prompt = self._build_user_prompt(observation)
        system_prompt = self._build_system_prompt()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM API call failed: %s", exc)
            return self._rule_fallback(observation)

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Learning methods (called by Trainer)
    # ------------------------------------------------------------------

    def update(
        self,
        obs: Any,
        action: dict,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Store experience in SQLite for future prompt enrichment."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            # Create tables if they don't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    email_text TEXT,
                    action_type TEXT,
                    priority TEXT,
                    intents TEXT,
                    total_reward REAL,
                    episode INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_memory (
                    pattern TEXT PRIMARY KEY,
                    best_action TEXT,
                    best_priority TEXT,
                    avg_reward REAL,
                    times_seen INTEGER DEFAULT 0
                )
            """)

            # Extract email info
            obs_dict = obs.to_dict() if hasattr(obs, "to_dict") else (obs if isinstance(obs, dict) else {})
            email = obs_dict.get("email", {})
            email_text = f"{email.get('subject', '')} {email.get('body', '')}"
            action_dict = action if isinstance(action, dict) else {}

            # Store experience
            cursor.execute(
                """INSERT INTO experiences (timestamp, email_text, action_type, priority, intents, total_reward)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    email_text[:500],
                    action_dict.get("action", "reply"),
                    action_dict.get("priority", "medium"),
                    json.dumps(action_dict.get("intents", [])),
                    reward,
                ),
            )

            # Extract keywords/patterns and update strategy_memory
            patterns = self._extract_patterns(email_text)
            for pattern in patterns:
                cursor.execute(
                    "SELECT avg_reward, times_seen FROM strategy_memory WHERE pattern = ?",
                    (pattern,),
                )
                row = cursor.fetchone()
                if row:
                    old_avg, count = row
                    new_count = count + 1
                    new_avg = (old_avg * count + reward) / new_count
                    # Update best action if this reward is higher
                    if reward > old_avg:
                        cursor.execute(
                            """UPDATE strategy_memory
                               SET avg_reward = ?, times_seen = ?,
                                   best_action = ?, best_priority = ?
                               WHERE pattern = ?""",
                            (new_avg, new_count,
                             action_dict.get("action", "reply"),
                             action_dict.get("priority", "medium"),
                             pattern),
                        )
                    else:
                        cursor.execute(
                            "UPDATE strategy_memory SET avg_reward = ?, times_seen = ? WHERE pattern = ?",
                            (new_avg, new_count, pattern),
                        )
                else:
                    cursor.execute(
                        """INSERT INTO strategy_memory (pattern, best_action, best_priority, avg_reward, times_seen)
                           VALUES (?, ?, ?, ?, 1)""",
                        (
                            pattern,
                            action_dict.get("action", "reply"),
                            action_dict.get("priority", "medium"),
                            reward,
                        ),
                    )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to update agent memory: %s", e)

    def on_episode_end(self, episode: int, avg_reward: float) -> None:
        """Reload patterns from DB so the next episode benefits immediately."""
        try:
            if DB_PATH.exists():
                conn = sqlite3.connect(str(DB_PATH))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_memory'"
                )
                if cursor.fetchone():
                    cursor.execute("""
                        SELECT pattern, best_action, best_priority, avg_reward, times_seen
                        FROM strategy_memory WHERE times_seen >= 3
                        ORDER BY avg_reward DESC LIMIT 10
                    """)
                    self.top_patterns = [dict(row) for row in cursor.fetchall()]

                    cursor.execute("""
                        SELECT pattern, best_action, avg_reward
                        FROM strategy_memory WHERE avg_reward < 0.50
                        ORDER BY avg_reward ASC LIMIT 5
                    """)
                    self.bad_patterns = [dict(row) for row in cursor.fetchall()]

                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='experiences'"
                )
                if cursor.fetchone():
                    cursor.execute("SELECT COUNT(*), AVG(total_reward) FROM experiences")
                    row = cursor.fetchone()
                    if row:
                        self.total_experiences = int(row[0] or 0)
                        self.overall_avg_reward = float(row[1] or 0.0)

                conn.close()
        except Exception as e:
            logger.warning("Could not reload patterns: %s", e)

    def save_state(self) -> None:
        """Save agent state metadata to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "total_experiences": self.total_experiences,
            "overall_avg_reward": self.overall_avg_reward,
            "learned_patterns": self.learned_patterns,
            "last_saved": datetime.now().isoformat(),
        }
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)

    def reset(self) -> None:
        """No-op for BaseAgent interface."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_patterns(text: str) -> List[str]:
        """Extract keyword patterns from email text for strategy memory."""
        text_l = text.lower()
        patterns = []

        keyword_groups = {
            "urgent": ["urgent", "asap", "deadline", "critical", "emergency"],
            "meeting": ["meeting", "schedule", "calendar", "sync", "call"],
            "task": ["assign", "task", "please do", "action item", "todo"],
            "question": ["?", "how", "what", "when", "where", "why"],
            "spam": ["free", "winner", "prize", "click here", "unsubscribe"],
            "follow_up": ["follow up", "following up", "reminder"],
            "security": ["security", "breach", "incident", "phishing", "outage"],
        }

        for pattern_name, keywords in keyword_groups.items():
            if any(kw in text_l for kw in keywords):
                patterns.append(pattern_name)

        if not patterns:
            patterns.append("general")

        return patterns

    @staticmethod
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

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Best-effort JSON extraction from the LLM output."""
        try:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
            if match:
                cleaned = match.group(1)
            else:
                match = re.search(r"(\{.*\})", raw, re.DOTALL)
                cleaned = match.group(1) if match else raw

            cleaned = cleaned.strip()
            data = json.loads(cleaned)
        except Exception:
            return RLAgent._fallback_action()

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
            "response": "Subject: Re: Your Message\n\nHi,\n\nThank you for your email. I'll review this and follow up.\n\nBest regards,\nAI Assistant",
        }

    def _rule_fallback(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Rule-based fallback when API key is missing."""
        email = observation.get("email", {})
        text = f"{email.get('subject', '')} {email.get('body', '')}".lower()

        intents = []
        spam_kw = {"congratulations", "winner", "lottery", "prize", "click here",
                    "free", "unsubscribe", "viagra"}
        if sum(1 for s in spam_kw if s in text) >= 2:
            intents.append("spam")
        if any(s in text for s in {"meeting", "schedule", "calendar", "sync"}):
            intents.append("meeting_request")
        if any(s in text for s in {"task", "assign", "please do", "action item"}):
            intents.append("task_assignment")
        if "?" in text:
            intents.append("question")
        if not intents:
            intents.append("information_sharing")

        urgency_kw = {"urgent", "asap", "deadline", "critical", "emergency"}
        urgency = sum(1 for kw in urgency_kw if kw in text)

        if urgency >= 2:
            priority = "critical"
        elif urgency >= 1:
            priority = "high"
        elif "?" in text:
            priority = "medium"
        else:
            priority = "low"

        if "spam" in intents:
            action = "ignore"
            response = ""
        elif priority == "critical":
            action = "escalate"
            response = "Subject: Escalation: Critical Priority\n\nHi team,\n\nEscalating this for immediate attention.\n\nBest regards,\nAI Assistant"
        elif "question" in intents or "task_assignment" in intents:
            action = "reply"
            response = "Subject: Re: Your Request\n\nHi,\n\nThanks for reaching out. I'll review and respond shortly.\n\nBest regards,\nAI Assistant"
        else:
            action = "reply"
            response = "Subject: Re: Your Message\n\nHi,\n\nThank you for the message. I'll follow up accordingly.\n\nBest regards,\nAI Assistant"

        return {
            "intents": intents,
            "priority": priority,
            "action": action,
            "response": response,
        }
