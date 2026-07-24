"""The comparison backends Aletheia is measured against (the project plan §8.2).

Three baselines live here:

* :class:`NaiveNonTransactionalStore` — a :class:`~core.adapter.StorageAdapter`
  whose state is split across independent structures with **no** unifying lock,
  so it can lose a write and diverge its vector from its row. This is exactly the
  defect experiment E1 measures against CockroachDB's serializable writes.
* :func:`build_bl_rag_memory_service` — the plain retrieve-and-answer baseline
  for E3: a :class:`~core.memory.MemoryService` configured from the ``BL_rag``
  arm (all four mechanisms off).
* :func:`bl_single_fleet_size` — the E6 single-agent baseline is nothing more
  than a fleet of size one; this thin helper names that.

.. warning::
   :class:`NaiveNonTransactionalStore` is **E1-only**. It is never the
   correctness oracle (that is :class:`~adapters.memory_inmem.InMemoryAdapter`)
   and never sources a published number. Its non-atomicity is *intentional and
   load-bearing* — "fixing" it with a lock would delete the very phenomenon E1
   exists to measure.

The metric helpers (:func:`find_divergences`, :func:`count_present`) are pure
functions over the store's state — no I/O — so they stay offline-testable.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from core.adapter import (
    AgentNotRegistered,
    DuplicateMemory,
    MemoryNotFound,
    StorageAdapter,
)
from core.config import DEFAULT_EMBEDDING_DIM
from core.embeddings import Embedder
from core.memory import MemoryService
from core.models import (
    CanonicalFact,
    MemoryEvent,
    MemoryHit,
    MemoryKind,
    MemoryStats,
    MemoryStatus,
    ProvenanceLink,
    QuarantineReason,
    QuarantineRecord,
    utcnow,
)
from experiments.arms import get_arm

#: Statuses excluded from retrieval/enumeration but reachable by id (mirrors the
#: oracle so the naive store is a faithful drop-in apart from its concurrency bug).
_NON_RETRIEVABLE = {MemoryStatus.SUPERSEDED, MemoryStatus.QUARANTINED, MemoryStatus.ARCHIVED}


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance in [0, 2]; a zero vector is defined as maximally distant."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 2.0
    return 1.0 - (dot / (norm_a * norm_b))


class NaiveNonTransactionalStore:
    """A deliberately non-atomic :class:`StorageAdapter` for experiment E1.

    Where :class:`~adapters.memory_inmem.InMemoryAdapter` guards every operation
    with a single ``RLock``, this store splits its state across four independent
    structures and touches them in separate, unsynchronised steps:

    * ``_rows``     — the operational memory row.
    * ``_vectors``  — the embedding, stored **separately** from the row.
    * ``_provenance`` — the provenance link.
    * ``_order``    — insertion order; the **only** thing reads enumerate.

    ``write_episode`` is non-atomic in two documented, load-bearing ways:

    **(A) Lost update.** ``_order`` is updated by read-copy-write-back with no
    lock: the writer snapshots ``self._order`` *by reference*, appends to a copy,
    then writes the copy back. Two writers that snapshot the same list each write
    back their own copy — the last writer wins and the other ``mem_id`` vanishes
    from ``_order``. ``write_episode`` returned that id (the write was
    "acknowledged"), yet reads never surface it: a countable lost update. The
    serializable oracle cannot do this — its whole op is under one lock and it
    never round-trips a plain-list snapshot.

    **(B) Vector-versus-row divergence.** ``_rows`` (step 1) and ``_vectors`` (step 4)
    are populated in separate steps with no covering lock, so a concurrent reader
    — or an abort injected at the mid-write window — can observe a row whose
    vector is absent (or vice versa). :func:`find_divergences` reports both
    directions at end-of-run.

    The mid-write window is the :meth:`_invoke_window` seam: ``interleave`` (a
    deterministic, thread-free test callback) makes the race reproducible in a
    single thread; ``race_delay`` (a real ``sleep``) widens the window for the
    threaded E1 runner. Both default to a no-op, so ordinary use is well-behaved.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        *,
        id_factory: Callable[[], str] | None = None,
        race_delay: float = 0.0,
        interleave: Callable[[str], None] | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self._new_id = id_factory or (lambda: str(uuid.uuid4()))
        self._race_delay = race_delay
        self._interleave = interleave

        # State split with NO unifying lock — the whole point of this store.
        self._agents: dict[str, dict[str, str]] = {}
        self._rows: dict[str, MemoryEvent] = {}
        self._vectors: dict[str, list[float]] = {}
        self._provenance: dict[str, ProvenanceLink] = {}
        self._order: list[str] = []
        self._canonical: dict[str, CanonicalFact] = {}
        self._quarantine_log: list[QuarantineRecord] = []

    # ------------------------------------------------------------------ window
    def _invoke_window(self, mem_id: str) -> None:
        """The mid-write race window. Callbacks must be re-entrancy aware: a
        nested write re-enters this seam, so an ``interleave`` callback that does
        a second write should fire its body for one id only."""
        if self._interleave is not None:
            self._interleave(mem_id)
        elif self._race_delay > 0.0:
            time.sleep(self._race_delay)

    # ------------------------------------------------------------------ agents
    def register_agent(self, agent_id: str, role: str, token_hash: str) -> None:
        if not agent_id:
            raise ValueError("agent_id is required")
        self._agents[agent_id] = {
            "role": role,
            "token_hash": token_hash,
            "created_at": self._agents.get(agent_id, {}).get("created_at") or utcnow(),
        }

    def _require_agent(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise AgentNotRegistered(f"agent not registered: {agent_id!r}")

    def _require_row(self, mem_id: str) -> MemoryEvent:
        try:
            return self._rows[mem_id]
        except KeyError:
            raise MemoryNotFound(f"no such memory: {mem_id!r}") from None

    def _check_dim(self, vec: Sequence[float]) -> None:
        if len(vec) != self.embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch: got {len(vec)}, store expects {self.embedding_dim}"
            )

    # ------------------------------------------------------------------ writes
    def write_episode(self, agent_id: str, event: MemoryEvent) -> str:
        """Non-atomic write — see the class docstring for the two failure modes."""
        if event.agent_id != agent_id:
            raise ValueError(
                f"agent_id mismatch: called with {agent_id!r}, event carries {event.agent_id!r}"
            )
        if event.embedding is not None:
            self._check_dim(event.embedding)
        self._require_agent(agent_id)

        mem_id = event.mem_id or self._new_id()
        # Check-then-act: a documented TOCTOU that rarely fires with uuid ids. The
        # PRIMARY lost-update mechanism is the _order clobber below, not this.
        if mem_id in self._rows:
            raise DuplicateMemory(f"memory already exists: {mem_id!r}")

        stored = replace(
            event,
            mem_id=mem_id,
            embedding=list(event.embedding) if event.embedding is not None else None,
            meta=dict(event.meta),
        )
        self._rows[mem_id] = stored  # step 1: row
        snapshot = self._order  # step 2a: READ current order (by reference)
        new_order = [*snapshot, mem_id]  # step 2b: COPY + append
        self._invoke_window(mem_id)  # <-- widened race window
        self._order = new_order  # step 2c: WRITE BACK the stale copy
        self._provenance[mem_id] = ProvenanceLink(  # step 3: provenance
            link_id=self._new_id(),
            mem_id=mem_id,
            parent_mem=event.parent_mem,
            agent_id=agent_id,
            signature=event.signature or "",
            hop_count=event.hop_count,
            created_at=stored.created_at,
        )
        self._vectors[mem_id] = (  # step 4: vector LAST, after another window point
            list(event.embedding) if event.embedding is not None else []
        )
        return mem_id

    def upsert_embedding(self, mem_id: str, vec: list[float], meta: dict) -> None:
        self._check_dim(vec)
        stored = self._require_row(mem_id)
        self._vectors[mem_id] = list(vec)
        if meta:
            stored.meta.update(meta)

    # ------------------------------------------------------------------- reads
    def _enumerate(self) -> list[MemoryEvent]:
        """Rows reachable via ``_order`` only — so a lost mem_id is invisible."""
        out: list[MemoryEvent] = []
        for mem_id in self._order:
            row = self._rows.get(mem_id)
            if row is not None:
                out.append(row)
        return out

    def read_recent(self, agent_id: str, k: int) -> list[MemoryEvent]:
        if k <= 0:
            return []
        self._require_agent(agent_id)
        own = [
            m
            for m in self._enumerate()
            if m.agent_id == agent_id and m.status not in _NON_RETRIEVABLE
        ]
        own.reverse()  # _order is insertion order (newest last) -> newest first
        return [self._copy(m) for m in own[:k]]

    def query_semantic(
        self, agent_id: str, vec: list[float], k: int, budget_tokens: int
    ) -> list[MemoryHit]:
        self._check_dim(vec)
        if k <= 0 or budget_tokens <= 0:
            return []
        self._require_agent(agent_id)

        candidates: list[tuple[float, MemoryEvent]] = []
        for mem in self._enumerate():
            if mem.status in _NON_RETRIEVABLE:
                continue
            embedding = self._vectors.get(mem.mem_id or "")
            if not embedding:
                # Divergence made visible to a reader: a row whose vector is
                # missing is skipped, not crashed on.
                continue
            candidates.append((_cosine_distance(vec, embedding), mem))

        candidates.sort(key=lambda pair: (pair[0], pair[1].mem_id or ""))

        hits: list[MemoryHit] = []
        spent = 0
        for distance, mem in candidates[:k]:
            cost = mem.cost_tokens or 0
            if spent + cost > budget_tokens:
                break
            spent += cost
            hits.append(
                MemoryHit(
                    mem_id=mem.mem_id or "",
                    agent_id=mem.agent_id,
                    content=mem.content,
                    kind=MemoryKind(mem.kind),
                    score=max(0.0, min(1.0, 1.0 - distance)),
                    distance=distance,
                    cost_tokens=cost,
                    importance=mem.importance,
                    status=MemoryStatus(mem.status),
                    created_at=mem.created_at,
                    hop_count=mem.hop_count,
                    meta=dict(mem.meta),
                )
            )
        return hits

    def get_memory(self, mem_id: str) -> MemoryEvent:
        return self._copy(self._require_row(mem_id))

    def list_memories(
        self,
        *,
        status: MemoryStatus | str | None = MemoryStatus.ACTIVE,
        kind: MemoryKind | str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryEvent]:
        want_status = None if status is None else MemoryStatus(status)
        want_kind = None if kind is None else MemoryKind(kind)
        matches = [
            m
            for m in self._enumerate()
            if (want_status is None or MemoryStatus(m.status) == want_status)
            and (want_kind is None or MemoryKind(m.kind) == want_kind)
            and (agent_id is None or m.agent_id == agent_id)
        ]
        matches.sort(key=lambda m: (m.created_at, m.mem_id or ""))
        return [self._copy(m) for m in matches]

    # ------------------------------------------------------ status transitions
    def supersede(self, old_mem_id: str, new_mem_id: str) -> None:
        if old_mem_id == new_mem_id:
            raise ValueError("a memory cannot supersede itself")
        old = self._require_row(old_mem_id)
        self._require_row(new_mem_id)
        old.status = MemoryStatus.SUPERSEDED
        old.superseded_by = new_mem_id

    def quarantine(self, mem_id: str, reason: str, detector: str, payload: dict) -> None:
        record = QuarantineRecord(
            q_id=self._new_id(),
            mem_id=mem_id,
            reason=QuarantineReason(reason),
            detector=detector,
            payload=dict(payload or {}),
        )
        stored = self._require_row(mem_id)
        stored.status = MemoryStatus.QUARANTINED
        self._quarantine_log.append(record)

    def archive(self, mem_id: str) -> None:
        self._require_row(mem_id).status = MemoryStatus.ARCHIVED

    # --------------------------------------------------------------- canonical
    def get_canonical(self, key: str) -> CanonicalFact | None:
        return self._canonical.get(key)

    def set_canonical(self, key: str, content: str, source_mem: str) -> None:
        if source_mem is not None:
            self._require_row(source_mem)
        current = self._canonical.get(key)
        self._canonical[key] = CanonicalFact(
            fact_key=key,
            content=content,
            version=(current.version + 1) if current else 1,
            source_mem=source_mem,
            updated_at=utcnow(),
        )

    # -------------------------------------------------------------- provenance
    def provenance_chain(self, mem_id: str) -> list[ProvenanceLink]:
        chain: list[ProvenanceLink] = []
        seen: set[str] = set()
        self._require_row(mem_id)
        cursor: str | None = mem_id
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            link = self._provenance.get(cursor)
            if link is None:
                break
            chain.append(link)
            cursor = link.parent_mem
        return chain

    # -------------------------------------------------------------------- misc
    def stats(self, agent_id: str | None = None) -> MemoryStats:
        scope = [m for m in self._rows.values() if agent_id is None or m.agent_id == agent_id]
        by_status: dict[MemoryStatus, int] = dict.fromkeys(MemoryStatus, 0)
        for mem in scope:
            by_status[MemoryStatus(mem.status)] += 1
        timestamps = sorted(m.created_at for m in scope)
        return MemoryStats(
            agent_id=agent_id,
            total_memories=len(scope),
            active=by_status[MemoryStatus.ACTIVE],
            superseded=by_status[MemoryStatus.SUPERSEDED],
            quarantined=by_status[MemoryStatus.QUARANTINED],
            archived=by_status[MemoryStatus.ARCHIVED],
            total_cost_tokens=sum(m.cost_tokens or 0 for m in scope),
            canonical_facts=len(self._canonical),
            oldest_created_at=timestamps[0] if timestamps else None,
            newest_created_at=timestamps[-1] if timestamps else None,
        )

    def close(self) -> None:
        """No resources to release; present so the protocol is uniform."""

    @staticmethod
    def _copy(mem: MemoryEvent) -> MemoryEvent:
        return replace(
            mem,
            embedding=list(mem.embedding) if mem.embedding is not None else None,
            meta=dict(mem.meta),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"NaiveNonTransactionalStore(rows={len(self._rows)}, order={len(self._order)}, "
            f"vectors={len(self._vectors)}, dim={self.embedding_dim})"
        )

    def __enter__(self) -> NaiveNonTransactionalStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# -------------------------------------------------------------- pure E1 metrics
def find_divergences(store: NaiveNonTransactionalStore) -> tuple[list[str], list[str]]:
    """The E1 divergence metric: ``(rows_without_vector, vectors_without_row)``.

    A row present in ``_rows`` whose ``_vectors`` entry is absent (or empty), and
    a vector present with no backing row — the two directions of vector-versus-row
    divergence a non-transactional write can leave behind. Pure, no I/O.
    """
    rows_without_vector = sorted(mem_id for mem_id in store._rows if not store._vectors.get(mem_id))
    vectors_without_row = sorted(mem_id for mem_id in store._vectors if mem_id not in store._rows)
    return rows_without_vector, vectors_without_row


def count_present(store: NaiveNonTransactionalStore) -> int:
    """Number of distinct rows reachable via ``_order`` — the lost-update ledger.

    A write whose ``mem_id`` was clobbered out of ``_order`` is counted in
    ``_rows`` but not here; ``len(_rows) - count_present`` is the lost count.
    """
    seen: set[str] = set()
    for mem_id in store._order:
        if mem_id in store._rows:
            seen.add(mem_id)
    return len(seen)


# ----------------------------------------------------------------- E3/E6 baselines
def build_naive_backend(
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    *,
    id_factory: Callable[[], str] | None = None,
    race_delay: float = 0.0,
    interleave: Callable[[str], None] | None = None,
) -> NaiveNonTransactionalStore:
    """Construct the E1 naive baseline store (see :class:`NaiveNonTransactionalStore`)."""
    return NaiveNonTransactionalStore(
        embedding_dim,
        id_factory=id_factory,
        race_delay=race_delay,
        interleave=interleave,
    )


def build_bl_rag_memory_service(adapter: StorageAdapter, embedder: Embedder) -> MemoryService:
    """The E3 plain-RAG baseline: a :class:`MemoryService` on the ``BL_rag`` arm.

    ``BL_rag`` turns off all four mechanisms — the service does nothing but embed,
    write and budget-retrieve. Same adapter and embedder as every other arm, so
    the only variable is the absent machinery.
    """
    config = get_arm("BL_rag").config()
    return MemoryService(adapter, embedder, config)


def bl_single_fleet_size() -> int:
    """The E6 single-agent baseline is a fleet of size one."""
    return 1
