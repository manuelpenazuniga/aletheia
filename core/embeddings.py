"""Embedding contract + a deterministic offline embedder.

The core never calls Bedrock. It takes an :class:`Embedder` — the real one
(Amazon Titan, in ``adapters/bedrock_embedder.py``) is injected from the edges.
That keeps the core portable and lets tests run offline and reproducibly.

:class:`DeterministicEmbedder` is **not** a mock standing in for a real model in
any user-facing path: it is a real, documented, seeded feature-hashing embedder
used for unit tests and offline local development. Anything shown in the demo or
reported as a result uses Titan (the project plan §3.2).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from core.config import DEFAULT_EMBEDDING_DIM

# Unicode word characters, not just [a-z0-9], so non-English text (and digits in
# other scripts) tokenizes into distinct tokens instead of collapsing to the
# empty-token fallback. This embedder is an offline test/dev stand-in — the SRE
# corpus is English — but it should not silently give every non-ASCII string the
# same vector.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a fixed-dimension vector."""

    @property
    def dim(self) -> int:
        """Dimension of every vector produced. Must match AletheiaConfig.embedding_dim."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed one string."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed several strings, preserving order."""
        ...


class DeterministicEmbedder:
    """Seeded feature-hashing embedder: same text, same vector, every process.

    Tokens are hashed (BLAKE2b, not Python's randomized ``hash``) onto signed
    dimensions, then the vector is L2-normalised. Unlike a random-per-string
    embedding this preserves lexical similarity structure — texts sharing words
    land closer — which is what makes it usable for exercising retrieval ordering
    without a network call.
    """

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._seed = seed
        # BLAKE2b rejects a key longer than 64 bytes; a large seed rendered to a
        # string would overflow it. Derive a fixed-length key from the seed so any
        # int is accepted while staying deterministic.
        self._key = hashlib.blake2b(str(seed).encode(), digest_size=16).digest()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def seed(self) -> int:
        return self._seed

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            # An empty vector would be a silent retrieval failure; anchor it to a
            # fixed, non-zero direction instead.
            vec[self._seed % self._dim] = 1.0
            return vec

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=self._key).digest()
            value = int.from_bytes(digest, "big")
            index = value % self._dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[index] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:  # pragma: no cover - requires exact sign cancellation
            vec[self._seed % self._dim] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DeterministicEmbedder(dim={self._dim}, seed={self._seed})"
