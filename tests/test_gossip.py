"""C4 gossip tick — offline, deterministic tests over the InMemoryAdapter oracle.

Every test runs with a :class:`DeterministicEmbedder` and a fixed ``id_factory``,
so recipients and degraded content are reproducible with zero network. The tick
is exercised against the reference adapter; the CockroachDB adapter must match it
behaviourally under the integration suite.
"""

from __future__ import annotations

import itertools

import pytest

from adapters.memory_inmem import InMemoryAdapter
from core.config import AletheiaConfig
from core.embeddings import DeterministicEmbedder
from core.gossip import (
    GOSSIP_LOSS_PER_HOP,
    GOSSIP_MAX_LOSS,
    GossipSummary,
    degradation_loss,
    degrade_content,
    gossip_tick,
)
from core.models import MemoryEvent, MemoryKind

DIM = 8


def _id_factory():
    """Monotonic, reproducible ids so seeded runs are byte-for-byte comparable."""
    counter = itertools.count()
    return lambda: f"id{next(counter):04d}"


def _config(**overrides) -> AletheiaConfig:
    base = {"embedding_dim": DIM, "gossip_max_hops": 3, "seed": 0}
    base.update(overrides)
    return AletheiaConfig(**base)


def _build_fleet(adapter, embedder, agents):
    for agent in agents:
        adapter.register_agent(agent, "sre", "hash")


def _seed(
    adapter,
    embedder,
    agent,
    content,
    *,
    importance=0.8,
    kind=MemoryKind.EPISODIC,
    hop_count=0,
    meta=None,
) -> str:
    event = MemoryEvent(
        agent_id=agent,
        content=content,
        kind=kind,
        embedding=embedder.embed(content),
        importance=importance,
        hop_count=hop_count,
        meta=meta or {},
    )
    return adapter.write_episode(agent, event)


def _children_of(adapter, mem_id) -> list[MemoryEvent]:
    return [m for m in adapter.list_memories(status=None) if m.parent_mem == mem_id]


# --------------------------------------------------------------------------- disabled
def test_disabled_flag_is_a_pure_no_op():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b"])
    _seed(adapter, embedder, "a", "disk full on node one")
    _seed(adapter, embedder, "b", "latency spike on the api")

    before = adapter.stats(None).total_memories
    summary = gossip_tick(adapter, embedder, _config(enable_gossip=False))

    assert summary == GossipSummary(candidates=0, propagations=0, capped=0, peers=0, skipped=True)
    assert adapter.stats(None).total_memories == before


# ------------------------------------------------------------------- single propagation
def test_single_source_writes_a_degraded_child():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b", "c"])
    # b and c are peers (authors of active memory) but not episodic sources.
    src = _seed(adapter, embedder, "a", "restart the database node to clear the lock")
    _seed(adapter, embedder, "b", "peer note b", kind=MemoryKind.SEMANTIC)
    _seed(adapter, embedder, "c", "peer note c", kind=MemoryKind.SEMANTIC)

    summary = gossip_tick(adapter, embedder, _config(), fanout=1)

    children = _children_of(adapter, src)
    assert len(children) == 1
    assert summary.propagations == 1
    assert summary.peers == 3
    child = children[0]
    assert child.parent_mem == src
    assert child.hop_count == 1
    assert child.kind == MemoryKind.EPISODIC
    assert child.embedding is not None
    assert child.meta["gossip"] is True
    assert child.meta["simulated_degradation"] is True
    assert child.meta["gossip_source_mem"] == src
    assert child.meta["gossip_source_agent"] == "a"
    assert child.agent_id in {"b", "c"}


def test_child_content_is_no_longer_than_the_source():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b"])
    source_text = "check the cert expiry then rotate it and redeploy the ingress gateway now"
    src = _seed(adapter, embedder, "a", source_text)
    _seed(adapter, embedder, "b", "peer", kind=MemoryKind.SEMANTIC)

    gossip_tick(adapter, embedder, _config(), fanout=1)

    child = _children_of(adapter, src)[0]
    assert len(child.content.split()) <= len(source_text.split())


def test_child_importance_is_decayed_and_in_range():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b"])
    src = _seed(adapter, embedder, "a", "disk pressure alert on node three", importance=0.8)
    _seed(adapter, embedder, "b", "peer", kind=MemoryKind.SEMANTIC)

    gossip_tick(adapter, embedder, _config(), fanout=1)

    child = _children_of(adapter, src)[0]
    assert 0.0 <= child.importance <= 1.0
    assert child.importance < 0.8


# ------------------------------------------------------------------------- hop cap
def test_source_at_the_hop_cap_is_counted_and_not_propagated():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b"])
    # hop_count == gossip_max_hops -> a child would exceed the cap.
    capped = _seed(adapter, embedder, "a", "already three hops out", hop_count=3)
    _seed(adapter, embedder, "b", "peer", kind=MemoryKind.SEMANTIC)

    summary = gossip_tick(adapter, embedder, _config(gossip_max_hops=3), fanout=1)

    assert summary.capped == 1
    assert summary.candidates == 0
    assert summary.propagations == 0
    assert _children_of(adapter, capped) == []


def test_max_hops_zero_propagates_nothing():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b"])
    _seed(adapter, embedder, "a", "fresh observation")
    _seed(adapter, embedder, "b", "another observation")

    summary = gossip_tick(adapter, embedder, _config(gossip_max_hops=0), fanout=1)

    assert summary.propagations == 0
    assert summary.skipped is False
    assert summary.capped == 2


# ------------------------------------------------------------- convergence & idempotency
def test_repeated_ticks_converge_and_then_stop():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b", "c"])
    # max_hops=1 keeps children from cascading, isolating the source's own reach.
    src = _seed(adapter, embedder, "a", "the fix for db latency is to add an index")
    _seed(adapter, embedder, "b", "peer", kind=MemoryKind.SEMANTIC)
    _seed(adapter, embedder, "c", "peer", kind=MemoryKind.SEMANTIC)
    cfg = _config(gossip_max_hops=1)

    gossip_tick(adapter, embedder, cfg, fanout=1)
    assert len(_children_of(adapter, src)) == 1
    gossip_tick(adapter, embedder, cfg, fanout=1)
    assert len(_children_of(adapter, src)) == 2  # every peer now has exactly one copy
    recipients = {c.agent_id for c in _children_of(adapter, src)}
    assert recipients == {"b", "c"}

    total_before = adapter.stats(None).total_memories
    summary = gossip_tick(adapter, embedder, cfg, fanout=1)  # nothing left to send
    assert summary.propagations == 0
    assert adapter.stats(None).total_memories == total_before


def test_back_to_back_ticks_do_not_duplicate_an_edge():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b"])
    _seed(adapter, embedder, "a", "observation from a")
    _seed(adapter, embedder, "b", "observation from b")
    cfg = _config(gossip_max_hops=1)

    gossip_tick(adapter, embedder, cfg, fanout=1)
    after_first = adapter.stats(None).total_memories
    summary = gossip_tick(adapter, embedder, cfg, fanout=1)

    assert summary.propagations == 0
    assert adapter.stats(None).total_memories == after_first


# ------------------------------------------------------------------------- no peers
def test_single_agent_fleet_propagates_nothing():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["solo"])
    _seed(adapter, embedder, "solo", "a lonely observation")

    summary = gossip_tick(adapter, embedder, _config(), fanout=1)

    assert summary.propagations == 0
    assert summary.peers == 1
    assert summary.candidates == 1


# ---------------------------------------------------------------------- determinism
def test_two_identical_runs_produce_identical_output():
    def run():
        embedder = DeterministicEmbedder(dim=DIM)
        adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
        _build_fleet(adapter, embedder, ["a", "b", "c"])
        _seed(adapter, embedder, "a", "runbook step one restart the pod")
        _seed(adapter, embedder, "b", "runbook step two scale the deployment")
        _seed(adapter, embedder, "c", "runbook step three verify the health check")
        gossip_tick(adapter, embedder, _config(gossip_max_hops=1), fanout=1)
        return [
            (m.agent_id, m.parent_mem, m.content)
            for m in adapter.list_memories(status=None)
            if m.meta.get("gossip")
        ]

    assert run() == run()


# ------------------------------------------------------------------ candidate filtering
def test_only_active_episodic_memories_are_sources():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b", "c", "d"])

    semantic = _seed(adapter, embedder, "a", "a generalisation", kind=MemoryKind.SEMANTIC)
    canonical = _seed(adapter, embedder, "b", "a pointer", kind=MemoryKind.CANONICAL_REF)
    superseded = _seed(adapter, embedder, "c", "old truth")
    newer = _seed(adapter, embedder, "c", "new truth")
    adapter.supersede(superseded, newer)
    quarantined = _seed(adapter, embedder, "d", "poisoned claim")
    adapter.quarantine(quarantined, "injection_pattern", "test", {})

    gossip_tick(adapter, embedder, _config(gossip_max_hops=1), fanout=3)

    for excluded in (semantic, canonical, superseded, quarantined):
        assert _children_of(adapter, excluded) == []
    # The one active-episodic memory (newer) is the only legitimate source.
    assert len(_children_of(adapter, newer)) >= 1


# ------------------------------------------------------------------------- fanout > 1
def test_fanout_reaches_multiple_distinct_peers():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b", "c", "d"])
    src = _seed(adapter, embedder, "a", "spread this widely across the fleet")
    _seed(adapter, embedder, "b", "peer", kind=MemoryKind.SEMANTIC)
    _seed(adapter, embedder, "c", "peer", kind=MemoryKind.SEMANTIC)
    _seed(adapter, embedder, "d", "peer", kind=MemoryKind.SEMANTIC)

    gossip_tick(adapter, embedder, _config(gossip_max_hops=1), fanout=3)

    recipients = [c.agent_id for c in _children_of(adapter, src)]
    assert sorted(recipients) == ["b", "c", "d"]
    assert len(recipients) == len(set(recipients))  # distinct


# ----------------------------------------------------------------------- dim mismatch
def test_embedder_dim_mismatch_raises_before_any_write():
    embedder = DeterministicEmbedder(dim=4)  # != config.embedding_dim (8)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b"])
    # Seed with the correct dimension so the memory exists; the tick's guard fires.
    correct = DeterministicEmbedder(dim=DIM)
    _seed(adapter, correct, "a", "an observation")
    _seed(adapter, correct, "b", "another observation")
    before = adapter.stats(None).total_memories

    with pytest.raises(ValueError, match="embedder dimension"):
        gossip_tick(adapter, embedder, _config())
    assert adapter.stats(None).total_memories == before


# ------------------------------------------------------------------ no fact_key leak
def test_gossip_child_never_carries_a_fact_key():
    embedder = DeterministicEmbedder(dim=DIM)
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=_id_factory())
    _build_fleet(adapter, embedder, ["a", "b"])
    src = _seed(
        adapter,
        embedder,
        "a",
        "the canonical runbook fix for db latency",
        meta={"fact_key": "runbook:db-latency:fix"},
    )
    _seed(adapter, embedder, "b", "peer", kind=MemoryKind.SEMANTIC)

    gossip_tick(adapter, embedder, _config(), fanout=1)

    child = _children_of(adapter, src)[0]
    assert "fact_key" not in child.meta


# ---------------------------------------------------- pure-function degradation knobs
def test_degradation_loss_is_monotonic_and_capped():
    assert degradation_loss(1, 3) == pytest.approx(GOSSIP_LOSS_PER_HOP)
    assert degradation_loss(1, 3) <= degradation_loss(2, 3) <= degradation_loss(3, 3)
    assert degradation_loss(100, 3) == GOSSIP_MAX_LOSS
    assert degradation_loss(0, 3) == 0.0


def test_degrade_content_is_deterministic_and_never_empty():
    text = "rotate the certificate and redeploy the ingress before the outage widens"
    first = degrade_content(text, 3, seed=0)
    second = degrade_content(text, 3, seed=0)
    assert first == second
    assert first != ""
    assert len(first.split()) <= len(text.split())
    # A single word at maximum loss must still survive (non-empty invariant).
    assert degrade_content("solo", 100, seed=7) == "solo"
    assert degrade_content("", 2, seed=0) == ""
