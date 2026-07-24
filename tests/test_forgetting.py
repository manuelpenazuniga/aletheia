"""C3 metabolic forgetting — unit tests against the InMemoryAdapter oracle.

Everything runs offline and deterministic: a seeded embedder, a counter id
factory, and an injected ``now`` so recency is reproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from adapters.memory_inmem import InMemoryAdapter
from core.config import AletheiaConfig
from core.forgetting import (
    DEFAULT_RECENCY_HALFLIFE_SECONDS,
    ForgettingResult,
    memory_value,
    run_forgetting,
)
from core.models import MemoryEvent, MemoryStatus, QuarantineReason

from .conftest import TEST_DIM

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _store(archived: list[str] | None = None, id_factory=None) -> InMemoryAdapter:
    cb = (lambda mem: archived.append(mem.mem_id)) if archived is not None else None
    store = InMemoryAdapter(embedding_dim=TEST_DIM, id_factory=id_factory, archive_callback=cb)
    store.register_agent("sre-1", "sre", "hash")
    return store


def _write(
    store: InMemoryAdapter,
    content: str,
    *,
    cost_tokens: int = 100,
    importance: float = 0.5,
    access_count: int = 0,
    last_accessed: datetime = NOW,
    created_at: datetime = NOW,
    embedding=None,
) -> str:
    return store.write_episode(
        "sre-1",
        MemoryEvent(
            agent_id="sre-1",
            content=content,
            embedding=embedding,
            cost_tokens=cost_tokens,
            importance=importance,
            access_count=access_count,
            last_accessed=last_accessed,
            created_at=created_at,
        ),
    )


# ------------------------------------------------------------------ no-op paths


def test_disabled_is_a_pure_noop():
    """enable_forgetting=False: ran=False, zero adapter calls, nothing archived."""
    archived: list[str] = []
    store = _store(archived)
    for i in range(5):
        _write(store, f"m{i}", cost_tokens=100)
    cfg = AletheiaConfig(memory_budget_tokens=10).with_overrides(enable_forgetting=False)

    result = run_forgetting(store, cfg, now=NOW)

    assert result == ForgettingResult(
        ran=False, budget=10, total_before=0, total_after=0, archived_ids=[], reclaimed_tokens=0
    )
    assert archived == []
    assert store.stats(None).active == 5


def test_under_budget_runs_but_archives_nothing():
    store = _store()
    for i in range(3):
        _write(store, f"m{i}", cost_tokens=100)
    cfg = AletheiaConfig(memory_budget_tokens=1000)

    result = run_forgetting(store, cfg, now=NOW)

    assert result.ran is True
    assert result.archived_ids == []
    assert result.total_before == result.total_after == 300
    assert store.stats(None).active == 3


def test_empty_store_is_a_noop_that_ran():
    result = run_forgetting(_store(), AletheiaConfig(memory_budget_tokens=100), now=NOW)
    assert result.ran is True
    assert result.total_before == 0
    assert result.archived_count == 0


# ------------------------------------------------------------------ pruning


def test_over_budget_prunes_lowest_value_first_and_stops_early(counter_ids):
    """Archive the fewest, cheapest-by-value memories to get under budget."""
    archived: list[str] = []
    store = _store(archived, id_factory=counter_ids)
    # All cost 100. Vary value via importance so the order is known.
    hi = _write(store, "high", cost_tokens=100, importance=1.0)
    mid = _write(store, "mid", cost_tokens=100, importance=0.6)
    lo = _write(store, "low", cost_tokens=100, importance=0.2)
    lowest = _write(store, "lowest", cost_tokens=100, importance=0.05)
    # total_before = 400, budget 250 -> must drop 2 (lowest-value: lowest, lo).
    cfg = AletheiaConfig(memory_budget_tokens=250)

    result = run_forgetting(store, cfg, now=NOW)

    assert result.archived_ids == [lowest, lo]  # ascending value order
    assert archived == [lowest, lo]
    assert result.total_after == 200
    assert result.total_after <= cfg.memory_budget_tokens
    assert store.get_memory(hi).status is MemoryStatus.ACTIVE
    assert store.get_memory(mid).status is MemoryStatus.ACTIVE
    assert store.get_memory(lo).status is MemoryStatus.ARCHIVED
    assert store.get_memory(lowest).status is MemoryStatus.ARCHIVED


def test_budget_accounting_is_consistent():
    store = _store()
    for i in range(6):
        _write(store, f"m{i}", cost_tokens=100, importance=0.1 * (i + 1))
    cfg = AletheiaConfig(memory_budget_tokens=250)

    r = run_forgetting(store, cfg, now=NOW)

    reclaimed = sum(store.get_memory(mid).cost_tokens for mid in r.archived_ids)
    assert r.reclaimed_tokens == reclaimed
    assert r.total_before - r.reclaimed_tokens == r.total_after
    assert r.total_after <= cfg.memory_budget_tokens


def test_high_value_memory_survives_while_low_value_is_shed():
    store = _store()
    keeper = _write(
        store, "keeper", cost_tokens=100, importance=1.0, access_count=50, last_accessed=NOW
    )
    for i in range(5):
        _write(
            store,
            f"junk{i}",
            cost_tokens=100,
            importance=0.05,
            access_count=0,
            last_accessed=NOW - timedelta(days=60),
        )
    cfg = AletheiaConfig(memory_budget_tokens=150)

    r = run_forgetting(store, cfg, now=NOW)

    assert keeper not in r.archived_ids
    assert store.get_memory(keeper).status is MemoryStatus.ACTIVE


def test_recency_drives_forgetting():
    """Two memories identical except last_accessed: the stale one goes first."""
    store = _store()
    fresh = _write(store, "fresh", cost_tokens=100, importance=0.5, last_accessed=NOW)
    stale = _write(
        store, "stale", cost_tokens=100, importance=0.5, last_accessed=NOW - timedelta(days=30)
    )
    cfg = AletheiaConfig(memory_budget_tokens=100)  # must drop exactly one

    r = run_forgetting(store, cfg, now=NOW)

    assert r.archived_ids == [stale]
    assert store.get_memory(fresh).status is MemoryStatus.ACTIVE


def test_usage_drives_forgetting():
    store = _store()
    used = _write(store, "used", cost_tokens=100, importance=0.5, access_count=20)
    unused = _write(store, "unused", cost_tokens=100, importance=0.5, access_count=0)
    cfg = AletheiaConfig(memory_budget_tokens=100)

    r = run_forgetting(store, cfg, now=NOW)

    assert r.archived_ids == [unused]
    assert store.get_memory(used).status is MemoryStatus.ACTIVE


def test_zero_importance_forgotten_first():
    store = _store()
    zero = _write(store, "zero", cost_tokens=100, importance=0.0, access_count=100)
    stale_low = _write(
        store,
        "stale-low",
        cost_tokens=100,
        importance=0.1,
        access_count=0,
        last_accessed=NOW - timedelta(days=365),
    )
    cfg = AletheiaConfig(memory_budget_tokens=100)

    r = run_forgetting(store, cfg, now=NOW)

    assert r.archived_ids == [zero]
    assert store.get_memory(stale_low).status is MemoryStatus.ACTIVE


def test_deterministic_tie_break_by_mem_id(counter_ids):
    """Identical value/last_accessed/created_at -> archived in mem_id order, stably."""
    store = _store(id_factory=counter_ids)
    ids = [
        _write(store, f"tie{i}", cost_tokens=100, importance=0.5, access_count=0) for i in range(3)
    ]
    cfg = AletheiaConfig(memory_budget_tokens=100)  # drop 2 of 3

    r1 = run_forgetting(store, cfg, now=NOW)

    assert r1.archived_ids == sorted(ids)[:2]

    # Re-run over the now-under-budget store: no further work.
    r2 = run_forgetting(store, cfg, now=NOW)
    assert r2.archived_ids == []


def test_no_embedding_memory_is_counted_and_archivable():
    store = _store()
    no_vec = _write(store, "no vector", cost_tokens=100, importance=0.05, embedding=None)
    _write(store, "keeper", cost_tokens=100, importance=1.0, access_count=10)
    cfg = AletheiaConfig(memory_budget_tokens=100)

    r = run_forgetting(store, cfg, now=NOW)

    assert r.total_before == 200
    assert r.archived_ids == [no_vec]


def test_archive_offload_runs_before_status_flip():
    seen: list[tuple[str, MemoryStatus]] = []
    store = InMemoryAdapter(embedding_dim=TEST_DIM)
    store.register_agent("sre-1", "sre", "hash")
    mem_id = _write(store, "offload me", cost_tokens=100, importance=0.0)

    def callback(mem):
        # At callback time the row is still ACTIVE (flip happens after).
        seen.append((mem.mem_id, store.get_memory(mem.mem_id).status))

    store._archive_callback = callback
    cfg = AletheiaConfig(memory_budget_tokens=10)

    run_forgetting(store, cfg, now=NOW)

    assert seen == [(mem_id, MemoryStatus.ACTIVE)]
    assert store.get_memory(mem_id).status is MemoryStatus.ARCHIVED


def test_only_active_memories_are_counted_and_archived():
    store = _store()
    active_low = _write(store, "active-low", cost_tokens=100, importance=0.1)
    superseded = _write(store, "superseded", cost_tokens=1000, importance=0.9)
    quarantined = _write(store, "quarantined", cost_tokens=1000, importance=0.9)
    survivor = _write(store, "survivor", cost_tokens=100, importance=1.0)
    store.supersede(superseded, survivor)
    store.quarantine(quarantined, QuarantineReason.BAD_PROVENANCE, "hmac", {})
    cfg = AletheiaConfig(memory_budget_tokens=150)

    r = run_forgetting(store, cfg, now=NOW)

    # Only the two ACTIVE memories are in the denominator (200), not the 2000
    # locked away in superseded/quarantined.
    assert r.total_before == 200
    assert r.archived_ids == [active_low]
    assert survivor not in r.archived_ids
    assert superseded not in r.archived_ids and quarantined not in r.archived_ids


def test_idempotent_rerun_archives_nothing_the_second_time():
    store = _store()
    for i in range(5):
        _write(store, f"m{i}", cost_tokens=100, importance=0.1 * (i + 1))
    cfg = AletheiaConfig(memory_budget_tokens=250)

    first = run_forgetting(store, cfg, now=NOW)
    second = run_forgetting(store, cfg, now=NOW)

    assert first.archived_count > 0
    assert second.archived_ids == []
    assert second.total_before <= cfg.memory_budget_tokens


# ------------------------------------------------------------ memory_value unit


def test_memory_value_monotonic_in_importance():
    base = dict(agent_id="sre-1", content="x", cost_tokens=10, access_count=3, last_accessed=NOW)
    lo = memory_value(MemoryEvent(importance=0.2, **base), now=NOW)
    hi = memory_value(MemoryEvent(importance=0.8, **base), now=NOW)
    assert hi > lo


def test_memory_value_monotonic_in_access_count():
    base = dict(agent_id="sre-1", content="x", cost_tokens=10, importance=0.5, last_accessed=NOW)
    lo = memory_value(MemoryEvent(access_count=1, **base), now=NOW)
    hi = memory_value(MemoryEvent(access_count=100, **base), now=NOW)
    assert hi > lo


def test_memory_value_decreases_with_age():
    base = dict(agent_id="sre-1", content="x", cost_tokens=10, importance=0.5, access_count=3)
    fresh = memory_value(MemoryEvent(last_accessed=NOW, **base), now=NOW)
    old = memory_value(MemoryEvent(last_accessed=NOW - timedelta(days=30), **base), now=NOW)
    assert old < fresh


def test_memory_value_clamps_future_last_accessed():
    """Clock skew (last_accessed > now) must not decay nor blow up the exponent."""
    base = dict(agent_id="sre-1", content="x", cost_tokens=10, importance=0.5, access_count=3)
    future = memory_value(MemoryEvent(last_accessed=NOW + timedelta(days=5), **base), now=NOW)
    at_now = memory_value(MemoryEvent(last_accessed=NOW, **base), now=NOW)
    assert future == pytest.approx(at_now)


def test_memory_value_exact_at_one_halflife():
    ev = MemoryEvent(
        agent_id="sre-1",
        content="x",
        cost_tokens=10,
        importance=1.0,
        access_count=0,  # usage_factor == 1.0
        last_accessed=NOW - timedelta(seconds=DEFAULT_RECENCY_HALFLIFE_SECONDS),
    )
    # importance 1.0 * usage 1.0 * recency 0.5 == 0.5
    assert memory_value(ev, now=NOW) == pytest.approx(0.5)
