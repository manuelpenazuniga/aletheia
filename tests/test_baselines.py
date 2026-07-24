"""The E1 comparison backends.

These tests prove the *harness logic*, never a published number: the naive store
CAN drop and diverge a write under simulated contention, while the serializable
oracle cannot. The lost-update and divergence proofs are single-threaded via the
deterministic ``interleave`` seam (guaranteed outcome, no sleep, no flakiness);
one tiny real-thread test asserts only the direction (oracle loses nothing).
"""

from __future__ import annotations

import threading

import pytest

from adapters.memory_inmem import InMemoryAdapter
from core.adapter import StorageAdapter
from core.config import DEFAULT_EMBEDDING_DIM
from core.embeddings import DeterministicEmbedder
from core.models import MemoryEvent
from experiments.baselines import (
    NaiveNonTransactionalStore,
    bl_single_fleet_size,
    build_bl_rag_memory_service,
    build_naive_backend,
    count_present,
    find_divergences,
)

DIM = 4
VEC = [1.0, 0.0, 0.0, 0.0]


def _store() -> NaiveNonTransactionalStore:
    store = build_naive_backend(embedding_dim=DIM)
    store.register_agent("a", "sre", "h")
    return store


def _event(content: str, mem_id: str) -> MemoryEvent:
    return MemoryEvent(agent_id="a", content=content, embedding=list(VEC), mem_id=mem_id)


# ------------------------------------------------------------------- contract
def test_naive_is_storage_adapter():
    assert isinstance(build_naive_backend(embedding_dim=DIM), StorageAdapter)


def test_naive_roundtrip_unstressed():
    """With no interleave/race set, the store behaves like an ordinary adapter."""
    store = _store()
    mid = store.write_episode("a", _event("disk full on node 3", "m1"))
    assert mid == "m1"

    recent = store.read_recent("a", 5)
    assert [m.mem_id for m in recent] == ["m1"]

    hits = store.query_semantic("a", list(VEC), k=5, budget_tokens=4000)
    assert [h.mem_id for h in hits] == ["m1"]
    assert not any(find_divergences(store))  # both directions empty
    assert count_present(store) == 1


def test_naive_query_semantic_skips_missing_vector():
    """A row whose vector diverged away must be skipped, not crash retrieval."""
    store = _store()
    store.write_episode("a", _event("has a vector", "m1"))
    # Simulate a diverged row: present in _rows/_order, absent from _vectors.
    store._vectors.pop("m1")
    hits = store.query_semantic("a", list(VEC), k=5, budget_tokens=4000)
    assert hits == []


# ---------------------------------------------------------------- lost update
def test_naive_loses_write_under_interleave():
    """Deterministic _order clobber: two acknowledged writes, one survives reads."""
    store = _store()

    def interleave(mem_id: str) -> None:
        # Fire once, at m1's window — after m1 snapshotted the (empty) _order but
        # before it writes its stale copy back. The nested m2 write commits into
        # _order; m1 then clobbers it. The nested call re-enters this seam with
        # mem_id == "m2", which we ignore.
        if mem_id == "m1":
            store.write_episode("a", _event("second", "m2"))

    store._interleave = interleave

    id1 = store.write_episode("a", _event("first", "m1"))
    assert id1 == "m1"

    # Both writes were acknowledged and both rows exist...
    assert {"m1", "m2"} <= set(store._rows)
    # ...but exactly one is reachable through _order (the other was clobbered).
    listed = [m.mem_id for m in store.list_memories()]
    assert listed == ["m1"]
    assert [m.mem_id for m in store.read_recent("a", 5)] == ["m1"]
    assert count_present(store) == 1
    assert len(store._rows) - count_present(store) == 1  # exactly one lost update


def test_serializable_never_loses_sequential():
    """The oracle enumerates both writes — it cannot drop an acknowledged write."""
    adapter = InMemoryAdapter(embedding_dim=DIM)
    adapter.register_agent("a", "sre", "h")
    adapter.write_episode("a", _event("first", "m1"))
    adapter.write_episode("a", _event("second", "m2"))
    listed = {m.mem_id for m in adapter.list_memories()}
    assert listed == {"m1", "m2"}


def test_serializable_never_loses_under_threads():
    """Real-thread variant: assert only the direction — zero lost on the oracle."""
    adapter = InMemoryAdapter(embedding_dim=DIM)
    adapter.register_agent("a", "sre", "h")
    n = 24
    barrier = threading.Barrier(n)

    def writer(i: int) -> None:
        barrier.wait()  # maximise overlap
        adapter.write_episode("a", _event(f"mem {i}", f"m{i:02d}"))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({m.mem_id for m in adapter.list_memories()}) == n  # nothing lost


# ------------------------------------------------------------------ divergence
def test_naive_can_diverge():
    """An abort injected between the row and the vector leaves a row with no vector."""
    store = _store()

    def boom(mem_id: str) -> None:
        raise RuntimeError("aborted mid-write")

    store._interleave = boom
    with pytest.raises(RuntimeError, match="aborted mid-write"):
        store.write_episode("a", _event("partial", "mx"))

    rows_without_vector, vectors_without_row = find_divergences(store)
    assert "mx" in rows_without_vector  # row exists, its vector never landed
    assert vectors_without_row == []


def test_inmem_no_divergence():
    """A completed oracle write always pairs its row with its embedding."""
    adapter = InMemoryAdapter(embedding_dim=DIM)
    adapter.register_agent("a", "sre", "h")
    for i in range(5):
        adapter.write_episode("a", _event(f"mem {i}", f"m{i}"))
    for mem in adapter.list_memories():
        assert mem.embedding is not None and len(mem.embedding) == DIM


# ------------------------------------------------------------- E3 / E6 baselines
def test_bl_rag_service_config():
    """The plain-RAG baseline runs a MemoryService with all four mechanisms off."""
    embedder = DeterministicEmbedder(dim=DEFAULT_EMBEDDING_DIM, seed=0)
    adapter = InMemoryAdapter(embedding_dim=DEFAULT_EMBEDDING_DIM)
    service = build_bl_rag_memory_service(adapter, embedder)
    cfg = service._config
    assert (
        cfg.enable_consolidation or cfg.enable_forgetting or cfg.enable_gossip or cfg.enable_immune
    ) is False
    assert cfg.labels["arm"] == "BL_rag"


def test_bl_single_fleet_size():
    assert bl_single_fleet_size() == 1
