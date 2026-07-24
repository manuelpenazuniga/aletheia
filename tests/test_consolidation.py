"""C2 consolidation cycle — unit tests against the InMemoryAdapter oracle.

Deterministic and offline: fixed timestamps decide "newest", and a counter id
factory pins the created_at tie-break.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from adapters.memory_inmem import InMemoryAdapter
from core.config import AletheiaConfig
from core.consolidation import ConsolidationSummary, consolidate
from core.models import MemoryEvent, MemoryKind, MemoryStatus

from .conftest import TEST_DIM

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _store(id_factory=None) -> InMemoryAdapter:
    store = InMemoryAdapter(embedding_dim=TEST_DIM, id_factory=id_factory)
    store.register_agent("sre-1", "sre", "hash")
    return store


def _write(
    store: InMemoryAdapter,
    content: str,
    *,
    fact_key: str | None = None,
    kind: MemoryKind = MemoryKind.EPISODIC,
    created_at: datetime = T0,
    mem_id: str | None = None,
) -> str:
    meta = {"fact_key": fact_key} if fact_key is not None else {}
    return store.write_episode(
        "sre-1",
        MemoryEvent(
            agent_id="sre-1",
            content=content,
            kind=kind,
            created_at=created_at,
            mem_id=mem_id,
            meta=meta,
        ),
    )


# ------------------------------------------------------------------ core cases


def test_two_revisions_supersede_and_promote():
    store = _store()
    old = _write(store, "restart the pod", fact_key="rb:x", created_at=T0)
    new = _write(store, "drain then restart", fact_key="rb:x", created_at=T0 + timedelta(hours=1))

    summary = consolidate(store, AletheiaConfig())

    assert summary == ConsolidationSummary(groups=1, supersedes=1, canonical_updates=1)
    assert store.get_memory(old).status is MemoryStatus.SUPERSEDED
    assert store.get_memory(old).superseded_by == new
    assert store.get_memory(new).status is MemoryStatus.ACTIVE
    fact = store.get_canonical("rb:x")
    assert fact.content == "drain then restart"
    assert fact.version == 1
    assert fact.source_mem == new


def test_three_revisions_chain():
    store = _store()
    m0 = _write(store, "v0", fact_key="rb:y", created_at=T0)
    m1 = _write(store, "v1", fact_key="rb:y", created_at=T0 + timedelta(hours=1))
    newest = _write(store, "v2", fact_key="rb:y", created_at=T0 + timedelta(hours=2))

    summary = consolidate(store, AletheiaConfig())

    assert summary.supersedes == 2
    assert summary.canonical_updates == 1
    assert store.get_memory(m0).status is MemoryStatus.SUPERSEDED
    assert store.get_memory(m1).status is MemoryStatus.SUPERSEDED
    fact = store.get_canonical("rb:y")
    assert fact.content == "v2"
    assert fact.version == 1
    assert fact.source_mem == newest


def test_singleton_first_promotion():
    store = _store()
    only = _write(store, "sole fact", fact_key="rb:z")

    summary = consolidate(store, AletheiaConfig())

    assert summary == ConsolidationSummary(groups=1, supersedes=0, canonical_updates=1)
    fact = store.get_canonical("rb:z")
    assert fact.content == "sole fact"
    assert fact.source_mem == only


def test_no_fact_key_variants_are_ignored():
    store = _store()
    a = _write(store, "no key")  # meta has no fact_key
    b = store.write_episode(
        "sre-1", MemoryEvent(agent_id="sre-1", content="empty key", meta={"fact_key": ""})
    )
    c = store.write_episode(
        "sre-1", MemoryEvent(agent_id="sre-1", content="whitespace", meta={"fact_key": "   "})
    )
    d = store.write_episode(
        "sre-1", MemoryEvent(agent_id="sre-1", content="non-str", meta={"fact_key": 123})
    )

    summary = consolidate(store, AletheiaConfig())

    assert summary == ConsolidationSummary(groups=0, supersedes=0, canonical_updates=0)
    for mid in (a, b, c, d):
        assert store.get_memory(mid).status is MemoryStatus.ACTIVE


def test_idempotent_second_run_does_not_bump_version():
    store = _store()
    _write(store, "v0", fact_key="rb:x", created_at=T0)
    _write(store, "v1", fact_key="rb:x", created_at=T0 + timedelta(hours=1))

    first = consolidate(store, AletheiaConfig())
    second = consolidate(store, AletheiaConfig())

    assert first.supersedes == 1 and first.canonical_updates == 1
    # Second run: the survivor is a singleton group whose content matches canonical.
    assert second.groups == 1
    assert second.supersedes == 0
    assert second.canonical_updates == 0
    assert store.get_canonical("rb:x").version == 1


def test_created_at_tie_broken_by_larger_mem_id():
    store = _store()
    small = _write(store, "small-id", fact_key="rb:tie", created_at=T0, mem_id="mem-aaa")
    large = _write(store, "large-id", fact_key="rb:tie", created_at=T0, mem_id="mem-zzz")

    summary = consolidate(store, AletheiaConfig())

    assert summary.supersedes == 1
    assert store.get_memory(small).status is MemoryStatus.SUPERSEDED
    assert store.get_memory(large).status is MemoryStatus.ACTIVE
    assert store.get_canonical("rb:tie").content == "large-id"


def test_content_unchanged_does_not_bump_version():
    """A newer revision with identical content: supersede yes, version bump no."""
    store = _store()
    _write(store, "same text", fact_key="rb:x", created_at=T0)
    consolidate(store, AletheiaConfig())  # promotes v1
    _write(store, "same text", fact_key="rb:x", created_at=T0 + timedelta(hours=1))

    summary = consolidate(store, AletheiaConfig())

    assert summary.supersedes == 1  # older superseded by newer
    assert summary.canonical_updates == 0  # content identical -> no bump
    assert store.get_canonical("rb:x").version == 1


def test_disabled_flag_is_a_noop():
    store = _RecordingStore(embedding_dim=TEST_DIM)
    store.register_agent("sre-1", "sre", "hash")
    _write(store, "v0", fact_key="rb:x", created_at=T0)
    _write(store, "v1", fact_key="rb:x", created_at=T0 + timedelta(hours=1))
    cfg = AletheiaConfig().with_overrides(enable_consolidation=False)

    summary = consolidate(store, cfg)

    assert summary == ConsolidationSummary(
        groups=0, supersedes=0, canonical_updates=0, skipped=True
    )
    assert store.supersede_calls == 0
    assert store.set_canonical_calls == 0


def test_multiple_keys_aggregate():
    store = _store()
    _write(store, "x0", fact_key="k1", created_at=T0)
    _write(store, "x1", fact_key="k1", created_at=T0 + timedelta(hours=1))
    _write(store, "y0", fact_key="k2", created_at=T0)
    _write(store, "y1", fact_key="k2", created_at=T0 + timedelta(hours=1))

    summary = consolidate(store, AletheiaConfig())

    assert summary == ConsolidationSummary(groups=2, supersedes=2, canonical_updates=2)
    assert store.get_canonical("k1").content == "x1"
    assert store.get_canonical("k2").content == "y1"


def test_semantic_memory_with_fact_key_is_ignored():
    store = _store()
    _write(store, "semantic", fact_key="rb:s", kind=MemoryKind.SEMANTIC)

    summary = consolidate(store, AletheiaConfig())

    assert summary == ConsolidationSummary(groups=0, supersedes=0, canonical_updates=0)
    assert store.get_canonical("rb:s") is None


def test_superseded_excluded_from_next_cycle():
    store = _store()
    old = _write(store, "v0", fact_key="rb:x", created_at=T0)
    _write(store, "v1", fact_key="rb:x", created_at=T0 + timedelta(hours=1))
    consolidate(store, AletheiaConfig())

    assert store.get_memory(old).status is MemoryStatus.SUPERSEDED
    active_episodic = store.list_memories(status=MemoryStatus.ACTIVE, kind=MemoryKind.EPISODIC)
    assert old not in {m.mem_id for m in active_episodic}


def test_no_candidates_returns_zeros_not_skipped():
    summary = consolidate(_store(), AletheiaConfig())
    assert summary == ConsolidationSummary(
        groups=0, supersedes=0, canonical_updates=0, skipped=False
    )


class _RecordingStore(InMemoryAdapter):
    """Counts mutating calls, to prove the disabled flag touches nothing."""

    supersede_calls = 0
    set_canonical_calls = 0

    def supersede(self, old_mem_id: str, new_mem_id: str) -> None:
        self.supersede_calls += 1
        super().supersede(old_mem_id, new_mem_id)

    def set_canonical(self, key: str, content: str, source_mem: str) -> None:
        self.set_canonical_calls += 1
        super().set_canonical(key, content, source_mem)
