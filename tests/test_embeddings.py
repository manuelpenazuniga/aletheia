"""DeterministicEmbedder: reproducible offline vectors with real similarity structure."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from adapters.memory_inmem import cosine_distance
from core.embeddings import DeterministicEmbedder, Embedder


def test_satisfies_the_embedder_protocol():
    assert isinstance(DeterministicEmbedder(dim=8), Embedder)


def test_same_text_same_vector():
    embedder = DeterministicEmbedder(dim=32, seed=0)
    assert embedder.embed("disk full on node 3") == embedder.embed("disk full on node 3")


def test_determinism_survives_a_fresh_interpreter():
    """String hashing is randomised per process; the embedder must not be."""
    program = (
        "from core.embeddings import DeterministicEmbedder;"
        "print(DeterministicEmbedder(dim=16, seed=0).embed('database latency')[:4])"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    }
    assert len(runs) == 1


def test_seed_changes_the_projection():
    assert DeterministicEmbedder(dim=32, seed=0).embed("x") != DeterministicEmbedder(
        dim=32, seed=1
    ).embed("x")


def test_vectors_are_unit_length():
    vec = DeterministicEmbedder(dim=64).embed("cert expired on the api gateway")
    assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0)


def test_similar_texts_are_closer_than_unrelated_ones():
    """Retrieval tests would be meaningless with a structure-free embedding."""
    embedder = DeterministicEmbedder(dim=256, seed=0)
    query = embedder.embed("database latency spike on the primary shard")
    related = embedder.embed("database latency spike on shard four")
    unrelated = embedder.embed("tls certificate expired on the gateway")

    assert cosine_distance(query, related) < cosine_distance(query, unrelated)


def test_empty_text_yields_a_usable_non_zero_vector():
    """A zero vector would be a silent retrieval failure."""
    vec = DeterministicEmbedder(dim=8, seed=3).embed("   ")
    assert any(v != 0.0 for v in vec)
    assert len(vec) == 8


def test_batch_preserves_order_and_dimension():
    embedder = DeterministicEmbedder(dim=16)
    texts = ["alpha", "beta", "gamma"]
    batch = embedder.embed_batch(texts)
    assert [len(v) for v in batch] == [16, 16, 16]
    assert batch == [embedder.embed(t) for t in texts]


def test_dimension_must_be_positive():
    with pytest.raises(ValueError):
        DeterministicEmbedder(dim=0)


def test_distinct_non_ascii_texts_get_distinct_vectors():
    """Non-English text must not all collapse to the empty-token fallback."""
    embedder = DeterministicEmbedder(dim=64, seed=0)
    assert embedder.embed("你好世界") != embedder.embed("مرحبا")
    assert embedder.embed("你好世界") != embedder.embed("   ")


def test_large_seed_does_not_overflow_the_hash_key():
    """A seed too large to be a BLAKE2b key must still produce a stable vector."""
    embedder = DeterministicEmbedder(dim=16, seed=10**40)
    a = embedder.embed("database latency")
    b = DeterministicEmbedder(dim=16, seed=10**40).embed("database latency")
    assert a == b
    assert math.isclose(sum(v * v for v in a), 1.0, rel_tol=1e-9)
