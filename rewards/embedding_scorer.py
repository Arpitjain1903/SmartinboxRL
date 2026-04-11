"""Semantic response scorer using sentence-transformer embeddings.

Computes cosine similarity between a generated response and one or more
gold-reference responses.  Falls back to a simple length / keyword heuristic
when the embedding model is unavailable (e.g. in lightweight test runs).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# HARD REQUIREMENT: every grader score must be strictly between 0 and 1.
# Hackathon validator REJECTS exactly 0.0 or 1.0.  Valid range: (0.01, 0.99).
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def safe_score(value) -> float:
    """Clamp *value* into the open interval (0.01, 0.99).

    Handles None, NaN, bool, and any numeric type.
    NEVER returns exactly 0.0 or 1.0 — satisfies the hackathon hard requirement.
    """
    if value is None:
        return 0.5
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    if v != v:  # NaN check
        return 0.5
    return max(_SCORE_MIN, min(_SCORE_MAX, v))


# Keep _strict as an alias so existing imports don't break
_strict = safe_score

# ---------------------------------------------------------------------------
# Lazy model loader (avoids slow import on module load)
# ---------------------------------------------------------------------------

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    """Load the SentenceTransformer model once."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            import torch
            logger.info("Loading embedding model: %s", _MODEL_NAME)
            _model = SentenceTransformer(_MODEL_NAME)
            _model.eval()
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            logger.info("Embedding model loaded successfully")
        except Exception as exc:
            logger.warning(
                "Could not load embedding model (%s). "
                "Falling back to heuristic scorer.",
                exc,
            )
            _model = "UNAVAILABLE"
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EmbeddingScorer:
    """Score a candidate response against gold reference(s)."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure the lazy model is only set once per process
        if getattr(self, "_model", None) is None:
            self._model = None  # lazy, will be loaded on first use


    @property
    def model(self):
        if self._model is None:
            self._model = _get_model()
        return self._model

    @property
    def available(self) -> bool:
        return self.model not in (None, "UNAVAILABLE")

    # ---- main scoring method ----

    def score(
        self,
        candidate: str,
        references: str | Sequence[str],
    ) -> float:
        """Return a similarity score ∈ [0, 1].

        If the embedding model is available, uses cosine similarity.
        Otherwise falls back to a keyword-overlap heuristic.
        """
        if isinstance(references, str):
            references = [references]

        if not candidate or not any(references):
            return safe_score(0.0)

        if self.available:
            return self._embedding_score(candidate, references)
        return self._heuristic_score(candidate, references)

    # ---- embedding-based scoring ----

    def _embedding_score(
        self, candidate: str, references: list[str]
    ) -> float:
        embeddings = self.model.encode(
            [candidate] + list(references),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        cand_emb = embeddings[0]
        ref_embs = embeddings[1:]

        # cosine similarity (embeddings already normalized)
        similarities = [float(np.dot(cand_emb, ref)) for ref in ref_embs]
        best = max(similarities)

        # clamp to strict open interval (0, 1)
        return safe_score(best)

    # ---- fallback heuristic ----

    @staticmethod
    def _heuristic_score(candidate: str, references: list[str]) -> float:
        """Simple token-overlap heuristic (Jaccard-ish)."""
        cand_tokens = set(candidate.lower().split())
        if not cand_tokens:
            return _strict(0.0)

        best = 0.0
        for ref in references:
            ref_tokens = set(ref.lower().split())
            if not ref_tokens:
                continue
            overlap = len(cand_tokens & ref_tokens)
            union = len(cand_tokens | ref_tokens)
            best = max(best, overlap / union if union else 0.0)

        # scale to roughly match embedding range, then enforce strict (0, 1)
        return safe_score(min(1.0, best * 1.3))
