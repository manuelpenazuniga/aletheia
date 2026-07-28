"""A tiny, dependency-free 2D projection of the vector memory, for the map view.

numpy is not a dependency here, so this uses a **landmark projection**: pick the
two most dissimilar memories as anchors, then place every memory (and any query) by
its real cosine similarity to each anchor. Similar memories land near each other, so
a semantic search visibly lights up a neighbourhood. Pure and deterministic — the
same vectors always project to the same coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _norm(v: float, lo: float, hi: float) -> float:
    if hi - lo < 1e-9:
        return 0.5
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


@dataclass(frozen=True, slots=True)
class Projection:
    """A fitted landmark projection. ``project`` maps a vector to (x, y) in [0, 1]²."""

    anchor_a: tuple[float, ...]
    anchor_b: tuple[float, ...]
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def project(self, vec: Sequence[float]) -> tuple[float, float]:
        return (
            _norm(cosine(vec, self.anchor_a), self.x_min, self.x_max),
            _norm(cosine(vec, self.anchor_b), self.y_min, self.y_max),
        )


def fit(vectors: list[Sequence[float]]) -> Projection:
    """Fit a landmark projection to a set of vectors.

    ``anchor_a`` is the most "outlying" vector (lowest total similarity to the rest);
    ``anchor_b`` is the vector least similar to ``anchor_a``. Every point is then
    (sim-to-a, sim-to-b), normalised to the point cloud's own range so the scatter
    fills the frame. Deterministic given the input order.
    """
    vs = [tuple(float(x) for x in v) for v in vectors if v]
    if len(vs) < 2:
        z = vs[0] if vs else (1.0,)
        return Projection(z, z, 0.0, 1.0, 0.0, 1.0)
    a = min(vs, key=lambda u: sum(cosine(u, w) for w in vs))
    b = min(vs, key=lambda u: cosine(u, a))
    xs = [cosine(v, a) for v in vs]
    ys = [cosine(v, b) for v in vs]
    return Projection(a, b, min(xs), max(xs), min(ys), max(ys))
