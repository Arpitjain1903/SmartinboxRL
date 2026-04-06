"""Email loader and noise-injection pipeline for SmartInboxRL.

Loads curated email tasks (easy / medium / hard) from JSON files,
optionally injects noise (typos, shorthand, casing), and serves
them to the environment as ordered or shuffled batches.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TASK_DIR = _DATA_DIR / "tasks"
_NOISE_PATH = _DATA_DIR / "noise_profiles.json"


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

class NoiseInjector:
    """Applies realistic perturbations to email text."""

    def __init__(self, noise_path: Path = _NOISE_PATH):
        self._profiles: dict[str, Any] = {}
        if noise_path.exists():
            with open(noise_path, "r", encoding="utf-8") as f:
                self._profiles = json.load(f)

        self.typo_map: dict[str, str] = self._profiles.get("typos", {})
        self.shorthand_map: dict[str, str] = self._profiles.get("shorthand", {})
        self.casing_rules: list[str] = self._profiles.get("casing_rules", [])

    # ---- public API ----

    def inject(self, text: str, intensity: float = 0.3) -> str:
        """Return *text* with noise proportional to *intensity* ∈ [0, 1]."""
        if intensity <= 0 or not text:
            return text
        text = self._apply_shorthand(text, intensity)
        text = self._apply_typos(text, intensity)
        text = self._apply_casing(text, intensity)
        return text

    # ---- private helpers ----

    def _apply_shorthand(self, text: str, intensity: float) -> str:
        for full, short in self.shorthand_map.items():
            if random.random() < intensity:
                pattern = re.compile(re.escape(full), re.IGNORECASE)
                text = pattern.sub(short, text)
        return text

    def _apply_typos(self, text: str, intensity: float) -> str:
        words = text.split()
        for i, word in enumerate(words):
            lower = word.lower()
            if lower in self.typo_map and random.random() < intensity * 0.5:
                words[i] = self.typo_map[lower]
        return " ".join(words)

    def _apply_casing(self, text: str, intensity: float) -> str:
        if "random_upper" in self.casing_rules and random.random() < intensity * 0.2:
            lines = text.split("\n")
            idx = random.randint(0, max(0, len(lines) - 1))
            lines[idx] = lines[idx].upper()
            text = "\n".join(lines)
        return text


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class EmailLoader:
    """Loads and serves email tasks for episodes.

    Parameters
    ----------
    difficulty : str
        ``"easy"`` | ``"medium"`` | ``"hard"`` | ``"enron"`` | ``"all"``
    noise_intensity : float
        0.0 = no noise, 1.0 = heavy noise.  Applied only at load time.
    shuffle : bool
        Whether to shuffle tasks within each episode reset.
    max_emails_per_episode : int | None
        Cap episode length.  ``None`` = use all available tasks.
    seed : int | None
        Fixed seed for reproducibility.
    """

    _DIFFICULTIES = ("easy", "medium", "hard", "enron")

    def __init__(
        self,
        difficulty: str = "all",
        noise_intensity: float = 0.3,
        shuffle: bool = True,
        max_emails_per_episode: int | None = None,
        seed: int | None = None,
    ):
        self.difficulty = difficulty.lower()
        self.noise_intensity = noise_intensity
        self.shuffle = shuffle
        self.max_emails = max_emails_per_episode
        self._rng = random.Random(seed)
        self._injector = NoiseInjector()

        self._tasks: list[dict[str, Any]] = self._load_tasks()

    # ---- public API ----

    def get_episode_emails(self, n: int | None = None) -> list[dict[str, Any]]:
        """Return a list of email-task dicts for one episode.

        Each dict contains at minimum:
          - id, subject, body, sender, difficulty
          - gold_intents, gold_priority, gold_action, gold_response
        """
        pool = list(self._tasks)
        if self.shuffle:
            self._rng.shuffle(pool)
        limit = n or self.max_emails or len(pool)
        batch = pool[:limit]

        # apply noise
        if self.noise_intensity > 0:
            batch = [self._noisy_copy(t) for t in batch]
        return batch

    @property
    def all_tasks(self) -> list[dict[str, Any]]:
        return list(self._tasks)

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    # ---- private helpers ----

    def _load_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        tiers = (
            list(self._DIFFICULTIES)
            if self.difficulty == "all"
            else [self.difficulty]
        )
        for tier in tiers:
            path = _TASK_DIR / f"{tier}_tasks.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    tier_tasks = json.load(f)
                    for t in tier_tasks:
                        t.setdefault("difficulty", tier)
                    tasks.extend(tier_tasks)
        return tasks

    def _noisy_copy(self, task: dict[str, Any]) -> dict[str, Any]:
        if task.get("difficulty") == "enron":
            return dict(task)
            
        noisy = dict(task)
        intensity = self.noise_intensity
        # harder tasks get more noise
        if task.get("difficulty") == "hard":
            intensity = min(1.0, intensity + 0.2)
        noisy["body"] = self._injector.inject(task.get("body", ""), intensity)
        noisy["subject"] = self._injector.inject(task.get("subject", ""), intensity * 0.5)
        return noisy
