"""Behavioural contract every :class:`core.adapter.StorageAdapter` must satisfy.

These functions are the shared, id-independent assertions that pin a backend to
the reference oracle (:class:`adapters.memory_inmem.InMemoryAdapter`). Each one
seeds identical events into BOTH a fresh oracle and the backend under test, then
asserts equality on projections that do not depend on generated ids or on
server-vs-client timestamps (content order, scores, counts, status transitions).

The oracle is literally the source of truth: if the backend disagrees with it on
any of these, the backend is wrong. Run against the InMemoryAdapter itself the
contract is trivially satisfied; run against the live CockroachDBAdapter it is
the integration test.

Not persisted by the CockroachDB backend (documented divergence): ``meta`` and
exact ``created_at`` values. The contract never asserts on either.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import pytest

from adapters.memory_inmem import InMemoryAdapter
from core.adapter import (
    AgentNotRegistered,
    DuplicateMemory,
    MemoryNotFound,
)
from core.embeddings import DeterministicEmbedder
from core.models import MemoryEvent, MemoryKind, MemoryStatus

#: float32 VECTOR storage in CockroachDB means distances computed on the stored
#: vectors differ from the oracle's float64 arithmetic by a small amount.
TOL = 2e-3


def _event(embedder: DeterministicEmbedder, agent_id: str, content: str, **kwargs) -> MemoryEvent:
    kwargs.setdefault("embedding", embedder.embed(content))
    return MemoryEvent(agent_id=agent_id, content=content, **kwargs)


def _write_both(backend, oracle, event_a: MemoryEvent, event_b: MemoryEvent) -> tuple[str, str]:
    """Write the *same logical* event to both stores, returning (backend_id, oracle_id).

    Two MemoryEvent instances are built (never share one across stores) so a store
    cannot mutate the other's copy.
    """
    return backend.write_episode(event_a.agent_id, event_a), oracle.write_episode(
        event_b.agent_id, event_b
    )


# ---------------------------------------------------------------------------
# Writes and provenance
# ---------------------------------------------------------------------------
def contract_write_creates_root_provenance(backend, embedder, agent_id: str) -> None:
    mem_id = backend.write_episode(agent_id, _event(embedder, agent_id, "a direct observation"))
    uuid.UUID(mem_id)  # a real UUID, not a placeholder id
    chain = backend.provenance_chain(mem_id)
    assert len(chain) == 1
    link = chain[0]
    assert link.hop_count == 0
    assert link.parent_mem is None
    assert link.is_root


def contract_write_rejections_leave_no_row(backend, embedder, agent_id: str) -> None:
    unknown = f"nobody-{uuid.uuid4().hex[:8]}"
    with pytest.raises(AgentNotRegistered):
        backend.write_episode(unknown, _event(embedder, unknown, "orphan write"))

    with pytest.raises(ValueError):
        backend.write_episode(agent_id, _event(embedder, "someone-else", "mismatched author"))

    with pytest.raises(ValueError):
        # A too-short embedding: models don't check dimension, the adapter does.
        backend.write_episode(
            agent_id, _event(embedder, agent_id, "wrong dimension", embedding=[0.1, 0.2, 0.3])
        )

    with pytest.raises(MemoryNotFound):
        backend.write_episode(
            agent_id,
            _event(embedder, agent_id, "child of nothing", parent_mem=str(uuid.uuid4())),
        )

    assert backend.stats(agent_id).active == 0


def contract_duplicate_mem_id(backend, embedder, agent_id: str) -> None:
    fixed = str(uuid.uuid4())
    backend.write_episode(agent_id, _event(embedder, agent_id, "first", mem_id=fixed))
    with pytest.raises(DuplicateMemory):
        backend.write_episode(agent_id, _event(embedder, agent_id, "second", mem_id=fixed))


# ---------------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------------
def contract_query_semantic_matches_oracle(backend, oracle, embedder, agent_id: str) -> None:
    contents = [
        "database read latency is spiking on the primary shard",
        "the deploy pipeline rolled back after a failed migration",
        "the tls certificate on the gateway expired overnight",
        "disk usage on the log volume crossed ninety five percent",
        "an oncall engineer acknowledged the paging alert",
    ]
    for c in contents:
        _write_both(backend, oracle, _event(embedder, agent_id, c), _event(embedder, agent_id, c))

    query = embedder.embed("p99 read latency is high on the database primary")
    b_hits = backend.query_semantic(agent_id, query, k=5, budget_tokens=100_000)
    o_hits = oracle.query_semantic(agent_id, query, k=5, budget_tokens=100_000)

    assert [h.content for h in b_hits] == [h.content for h in o_hits]
    for bh, oh in zip(b_hits, o_hits, strict=True):
        assert abs(bh.distance - oh.distance) < TOL, (bh.content, bh.distance, oh.distance)
        assert abs(bh.score - oh.score) < TOL


def contract_query_budget_truncation(backend, oracle, embedder, agent_id: str) -> None:
    # Distinct, increasing costs so a mid-list budget cut is unambiguous, and
    # distinct contents so distances (hence order) are unique in both stores.
    seeds = [
        ("closest match on database latency runbook", 30),
        ("second closest on database restart drain", 40),
        ("third on cache eviction thrash", 55),
        ("fourth on network partition healing", 70),
    ]
    ids_by_content: dict[str, tuple[str, str]] = {}
    for content, cost in seeds:
        b_id, o_id = _write_both(
            backend,
            oracle,
            _event(embedder, agent_id, content, cost_tokens=cost),
            _event(embedder, agent_id, content, cost_tokens=cost),
        )
        ids_by_content[content] = (b_id, o_id)

    query = embedder.embed("database latency runbook")
    budget = 100  # fits some, then the next hit overflows
    b_hits = backend.query_semantic(agent_id, query, k=4, budget_tokens=budget)
    o_hits = oracle.query_semantic(agent_id, query, k=4, budget_tokens=budget)

    assert [h.content for h in b_hits] == [h.content for h in o_hits]
    assert len(b_hits) < len(seeds)  # truncation actually happened

    returned = {h.content for h in b_hits}
    for content, (b_id, o_id) in ids_by_content.items():
        expected = 1 if content in returned else 0
        assert backend.get_memory(b_id).access_count == expected
        assert oracle.get_memory(o_id).access_count == expected


# ---------------------------------------------------------------------------
# Recency and status transitions
# ---------------------------------------------------------------------------
def contract_read_recent_excludes_non_active(backend, oracle, embedder, agent_id: str, sleep):
    ids: list[tuple[str, str]] = []
    for i in range(4):
        b_id, o_id = _write_both(
            backend,
            oracle,
            _event(embedder, agent_id, f"recent event number {i}"),
            _event(embedder, agent_id, f"recent event number {i}"),
        )
        ids.append((b_id, o_id))
        sleep()  # distinct created_at so newest-first order is unambiguous

    # Supersede the first, quarantine the second: both must drop out of recency.
    new_b, new_o = _write_both(
        backend,
        oracle,
        _event(embedder, agent_id, "replacement memory"),
        _event(embedder, agent_id, "replacement memory"),
    )
    backend.supersede(ids[0][0], new_b)
    oracle.supersede(ids[0][1], new_o)
    backend.quarantine(ids[1][0], "injection_pattern", "unit", {"k": "v"})
    oracle.quarantine(ids[1][1], "injection_pattern", "unit", {"k": "v"})

    b_recent = backend.read_recent(agent_id, k=10)
    o_recent = oracle.read_recent(agent_id, k=10)
    assert [m.content for m in b_recent] == [m.content for m in o_recent]
    assert "recent event number 0" not in {m.content for m in b_recent}
    assert "recent event number 1" not in {m.content for m in b_recent}


def contract_supersede_freshness(backend, embedder, agent_id: str) -> None:
    old_id = backend.write_episode(
        agent_id, _event(embedder, agent_id, "the old runbook says bounce the node")
    )
    new_id = backend.write_episode(
        agent_id, _event(embedder, agent_id, "the new runbook says drain then restart")
    )
    backend.supersede(old_id, new_id)

    query = embedder.embed("runbook for the node")
    hit_ids = {
        h.mem_id for h in backend.query_semantic(agent_id, query, k=10, budget_tokens=100_000)
    }
    assert old_id not in hit_ids
    assert old_id not in {m.mem_id for m in backend.read_recent(agent_id, k=10)}

    snapshot = backend.get_memory(old_id)
    assert snapshot.status == MemoryStatus.SUPERSEDED
    assert snapshot.superseded_by == new_id


def contract_quarantine(backend, embedder, agent_id: str) -> None:
    mem_id = backend.write_episode(agent_id, _event(embedder, agent_id, "suspicious claim"))

    before = len(backend.quarantine_log())
    with pytest.raises(ValueError):
        backend.quarantine(mem_id, "not_a_real_reason", "unit", {})
    assert backend.get_memory(mem_id).status == MemoryStatus.ACTIVE
    assert len(backend.quarantine_log()) == before  # invalid reason wrote nothing

    payload = {"detector": "provenance", "score": 0.9, "nested": {"a": [1, 2, 3]}}
    backend.quarantine(mem_id, "bad_provenance", "immune-v1", payload)
    assert backend.get_memory(mem_id).status == MemoryStatus.QUARANTINED
    assert mem_id not in {
        h.mem_id
        for h in backend.query_semantic(
            agent_id, embedder.embed("suspicious claim"), k=10, budget_tokens=100_000
        )
    }
    mine = [r for r in backend.quarantine_log() if r.mem_id == mem_id]
    assert len(mine) == 1
    assert mine[0].payload == payload  # JSONB round-trips


def contract_archive(backend, embedder, agent_id: str, make_adapter: Callable[..., object]) -> None:
    # A raising callback must leave the memory active and commit nothing.
    boom_id = backend.write_episode(agent_id, _event(embedder, agent_id, "will fail to offload"))

    def boom(_ev):
        raise RuntimeError("s3 unavailable")

    raising = make_adapter(archive_callback=boom)
    with pytest.raises(RuntimeError):
        raising.archive(boom_id)
    assert backend.get_memory(boom_id).status == MemoryStatus.ACTIVE
    raising.close()

    # A succeeding callback receives the pre-flip snapshot, then status flips.
    ok_id = backend.write_episode(agent_id, _event(embedder, agent_id, "will offload cleanly"))
    captured: list = []
    ok = make_adapter(archive_callback=captured.append)
    ok.archive(ok_id)
    ok.close()

    assert len(captured) == 1
    assert captured[0].mem_id == ok_id
    assert captured[0].status == MemoryStatus.ACTIVE  # snapshot taken before the flip
    assert backend.get_memory(ok_id).status == MemoryStatus.ARCHIVED
    assert ok_id not in {
        h.mem_id
        for h in backend.query_semantic(
            agent_id, embedder.embed("offload"), k=10, budget_tokens=100_000
        )
    }


# ---------------------------------------------------------------------------
# Canonical facts and provenance chains
# ---------------------------------------------------------------------------
def contract_set_canonical_versions(backend, oracle, embedder, agent_id: str) -> None:
    src = backend.write_episode(agent_id, _event(embedder, agent_id, "source of the fact"))
    o_src = oracle.write_episode(agent_id, _event(embedder, agent_id, "source of the fact"))
    key = f"runbook:test:{uuid.uuid4().hex[:8]}"

    backend.set_canonical(key, "v1 content", src)
    oracle.set_canonical(key, "v1 content", o_src)
    assert backend.get_canonical(key).version == 1 == oracle.get_canonical(key).version

    backend.set_canonical(key, "v2 content", src)
    oracle.set_canonical(key, "v2 content", o_src)
    b_fact = backend.get_canonical(key)
    assert b_fact.version == 2 == oracle.get_canonical(key).version
    assert b_fact.content == "v2 content"

    with pytest.raises(MemoryNotFound):
        backend.set_canonical(key, "bad", str(uuid.uuid4()))


def contract_provenance_chain_multi_hop(backend, oracle, embedder, agent_id: str) -> None:
    def build(store) -> list[str]:
        ids: list[str] = []
        parent: str | None = None
        for hop in range(3):
            ev = _event(embedder, agent_id, f"claim at hop {hop}", parent_mem=parent, hop_count=hop)
            mem_id = store.write_episode(agent_id, ev)
            ids.append(mem_id)
            parent = mem_id
        return ids

    b_ids = build(backend)
    o_ids = build(oracle)

    b_chain = backend.provenance_chain(b_ids[-1])
    o_chain = oracle.provenance_chain(o_ids[-1])
    assert [link.hop_count for link in b_chain] == [link.hop_count for link in o_chain]
    assert [link.hop_count for link in b_chain] == [2, 1, 0]  # nearest first


# ---------------------------------------------------------------------------
# Aggregates and enumeration
# ---------------------------------------------------------------------------
def contract_stats_match_oracle(backend, oracle, embedder, agent_id: str) -> None:
    ids: list[tuple[str, str]] = []
    for i in range(5):
        ids.append(
            _write_both(
                backend,
                oracle,
                _event(embedder, agent_id, f"stat seed {i}", cost_tokens=10 + i),
                _event(embedder, agent_id, f"stat seed {i}", cost_tokens=10 + i),
            )
        )
    backend.quarantine(ids[0][0], "semantic_anomaly", "unit", {})
    oracle.quarantine(ids[0][1], "semantic_anomaly", "unit", {})
    backend.set_canonical("fact:stats", "canon", ids[1][0])
    oracle.set_canonical("fact:stats", "canon", ids[1][1])

    for scope in (agent_id, None):
        b = backend.stats(scope)
        o = oracle.stats(scope)
        assert (b.total_memories, b.active, b.quarantined) == (
            o.total_memories,
            o.active,
            o.quarantined,
        )
        assert b.total_cost_tokens == o.total_cost_tokens
        assert b.canonical_facts == o.canonical_facts


def contract_list_memories_match_oracle(backend, oracle, embedder, agent_id: str, sleep):
    ids: list[tuple[str, str]] = []
    for i in range(4):
        kind = MemoryKind.SEMANTIC if i % 2 else MemoryKind.EPISODIC
        ids.append(
            _write_both(
                backend,
                oracle,
                _event(embedder, agent_id, f"list seed {i}", kind=kind),
                _event(embedder, agent_id, f"list seed {i}", kind=kind),
            )
        )
        sleep()
    backend.supersede(ids[0][0], ids[1][0])
    oracle.supersede(ids[0][1], ids[1][1])

    for kwargs in (
        {},
        {"status": None},
        {"status": MemoryStatus.SUPERSEDED},
        {"kind": MemoryKind.SEMANTIC},
        {"agent_id": agent_id},
        {"status": None, "kind": MemoryKind.EPISODIC, "agent_id": agent_id},
    ):
        b = [m.content for m in backend.list_memories(**kwargs)]
        o = [m.content for m in oracle.list_memories(**kwargs)]
        assert b == o, kwargs


def make_reference_pair(dim: int = 8) -> tuple[InMemoryAdapter, DeterministicEmbedder]:
    """A self-check helper: the contract run against the oracle itself must pass."""
    embedder = DeterministicEmbedder(dim=dim, seed=0)
    store = InMemoryAdapter(embedding_dim=dim)
    return store, embedder
