"""MemoryService (C1): write + budgeted retrieval over the storage contract."""

from __future__ import annotations

import pytest

from adapters.memory_inmem import InMemoryAdapter
from core.config import AletheiaConfig
from core.embeddings import DeterministicEmbedder
from core.memory import MemoryService
from core.models import MemoryKind

from .conftest import TEST_DIM


@pytest.fixture
def service(adapter):
    embedder = DeterministicEmbedder(dim=TEST_DIM, seed=0)
    config = AletheiaConfig(embedding_dim=TEST_DIM, retrieval_budget_tokens=250)
    return MemoryService(adapter, embedder, config)


def test_remember_embeds_and_persists(service, adapter):
    mem_id = service.remember("sre-1", "disk full on node 3")
    stored = adapter.get_memory(mem_id)
    assert stored.content == "disk full on node 3"
    assert stored.embedding is not None and len(stored.embedding) == TEST_DIM
    assert adapter.provenance_chain(mem_id)[0].mem_id == mem_id


def test_remember_computes_the_embedding_itself(service, adapter):
    """The caller never supplies a vector — the trusted side embeds."""
    a = service.remember("sre-1", "database latency spike on the primary shard")
    b = service.remember("sre-1", "database latency spike on the primary shard", importance=0.9)
    # Same content -> same (deterministic) embedding, regardless of other fields.
    assert adapter.get_memory(a).embedding == adapter.get_memory(b).embedding


def test_recall_returns_relevant_memories_first(service):
    service.remember("sre-1", "database latency spike on the primary shard")
    service.remember("sre-1", "tls certificate expired on the api gateway")

    hits = service.recall("sre-1", "why is the database so slow to read")

    assert hits
    assert "latency" in hits[0].content


def test_recall_respects_the_configured_budget(adapter):
    embedder = DeterministicEmbedder(dim=TEST_DIM, seed=0)
    config = AletheiaConfig(embedding_dim=TEST_DIM, retrieval_budget_tokens=30)
    service = MemoryService(adapter, embedder, config)
    for i in range(5):
        service.remember("sre-1", f"memory {i} about database latency and slow reads")

    hits = service.recall("sre-1", "database latency")

    # ~12-token memories under a 30-token budget: truncated below all five.
    assert sum(h.cost_tokens for h in hits) <= 30
    assert 0 < len(hits) < 5


def test_recall_budget_override_is_honoured(service):
    for i in range(3):
        service.remember("sre-1", f"memory {i} about database latency")
    assert service.recall("sre-1", "database latency", budget_tokens=1) == []


def test_recent_is_newest_first(service):
    for i in range(3):
        service.remember("sre-1", f"event {i}")
    recent = service.recent("sre-1", k=2)
    assert len(recent) == 2
    assert recent[0].created_at >= recent[1].created_at


def test_kind_is_passed_through(service, adapter):
    mem_id = service.remember("sre-1", "a generalisation", kind=MemoryKind.SEMANTIC)
    assert adapter.get_memory(mem_id).kind is MemoryKind.SEMANTIC


def test_embedder_dim_must_match_config():
    adapter = InMemoryAdapter(embedding_dim=TEST_DIM)
    wrong = DeterministicEmbedder(dim=TEST_DIM + 1, seed=0)
    with pytest.raises(ValueError, match="must agree"):
        MemoryService(adapter, wrong, AletheiaConfig(embedding_dim=TEST_DIM))
