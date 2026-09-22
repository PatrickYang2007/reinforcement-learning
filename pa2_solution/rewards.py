"""Completion-only rewards for the TinyStories bear 2×2 experiments.

Train talks to rewards only through :class:`RewardFn`:
``completions[B] -> rewards[B]`` (finite, detached, no prompt credit).
"""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Protocol
import unicodedata

import torch

_BEAR = re.compile(r"(?<!\w)bears?(?!\w)", flags=re.IGNORECASE)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def bear_mentions(text: str) -> int:
    return len(_BEAR.findall(normalize(text)))


def extract_words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", normalize(text)))


class RewardFn(Protocol):
    name: str

    def __call__(self, completions: Sequence[str]) -> torch.Tensor:
        """Return shape ``[B]`` float32 rewards, detached and finite."""


class SparseBearReward:
    """Best lexical recipe: ``r = 1[c(y) >= 1]`` for whole-word bear/bears."""

    name = "sparse"

    def __call__(self, completions: Sequence[str]) -> torch.Tensor:
        # TODO 3.1: BEGIN sparse_reward
        values = [1.0 if bear_mentions(text) >= 1 else 0.0 for text in completions]
        # TODO 3.1: END sparse_reward
        return torch.tensor(values, dtype=torch.float32).detach()


class DenseStoryReward:
    """Whole-story MiniLM cosine similarity to a target string (default ``bear``).

    Embedder matches the assignment style:
    ``SentenceTransformer("all-MiniLM-L6-v2")``.
    """

    name = "dense"

    def __init__(
        self,
        target_text: str = "bear",
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ) -> None:
        if not target_text.strip():
            raise ValueError("target_text must be non-empty")
        self.target_text = target_text
        self.model_name = model_name
        self.device = device
        self._model = None
        self._target = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "dense reward needs sentence-transformers: pip install -e '.[semantic]'"
                ) from exc
            # Same constructor students see in the assignment notebook.
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        with torch.inference_mode():
            emb = self._load().encode(
                list(texts), convert_to_tensor=True, show_progress_bar=False
            )
        return torch.as_tensor(emb, dtype=torch.float32).detach()

    def __call__(self, completions: Sequence[str]) -> torch.Tensor:
        if not completions:
            return torch.empty(0, dtype=torch.float32)
        # Empty strings get score 0 without embedding.
        active = [i for i, t in enumerate(completions) if isinstance(t, str) and t.strip()]
        out = torch.zeros(len(completions), dtype=torch.float32)
        if not active:
            return out
        texts = [completions[i] for i in active]
        emb = self._encode(texts)
        # TODO 3.2: BEGIN dense_reward
        if self._target is None:
            # Encode the target once; later calls reuse this unit vector.
            target = self._encode([self.target_text])
            self._target = torch.nn.functional.normalize(target, dim=-1)[0]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        scores = (emb @ self._target.to(emb.device)).cpu()
        # TODO 3.2: END dense_reward
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("dense reward produced non-finite scores")
        out[torch.tensor(active, dtype=torch.long)] = scores
        return out.detach()


def build_reward(name: str, **kwargs) -> RewardFn:
    key = name.strip().lower()
    aliases = {
        "sparse": "sparse",
        "sparse_bear": "sparse",
        "dense": "dense",
        "story_semantic": "dense",
        "whole_story_cosine": "dense",
    }
    key = aliases.get(key, key)
    if key == "sparse":
        return SparseBearReward()
    if key == "dense":
        return DenseStoryReward(**kwargs)
    raise KeyError(f"unknown reward {name!r}; use sparse or dense")
