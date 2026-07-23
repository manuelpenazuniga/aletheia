"""Shared fixtures. Unit tests are offline, deterministic and fast."""

from __future__ import annotations

import itertools
import threading

import pytest

from adapters.memory_inmem import InMemoryAdapter
from core.embeddings import DeterministicEmbedder
from core.models import MemoryEvent

#: Small dimension for unit tests: the logic is dimension-agnostic and 8 floats
#: keep failures readable. Production dimension (1024) is exercised by smoke.py.
TEST_DIM = 8


@pytest.fixture
def embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder(dim=TEST_DIM, seed=0)


@pytest.fixture
def counter_ids():
    """Deterministic, thread-safe id factory, so assertions can name ids directly.

    Note it is shared by memory ids and provenance link ids, so ids interleave —
    tests should use the ids returned by writes rather than predicting them.
    """
    counter = itertools.count(1)
    lock = threading.Lock()

    def _next_id() -> str:
        with lock:
            return f"mem-{next(counter):04d}"

    return _next_id


@pytest.fixture
def adapter(counter_ids) -> InMemoryAdapter:
    store = InMemoryAdapter(embedding_dim=TEST_DIM, id_factory=counter_ids)
    store.register_agent("sre-1", "sre", "hash-1")
    store.register_agent("sre-2", "sre", "hash-2")
    return store


@pytest.fixture
def make_event(embedder):
    """Build a MemoryEvent with a real (deterministic) embedding of its content."""

    def _make(content: str, agent_id: str = "sre-1", **kwargs) -> MemoryEvent:
        kwargs.setdefault("embedding", embedder.embed(content))
        return MemoryEvent(agent_id=agent_id, content=content, **kwargs)

    return _make
