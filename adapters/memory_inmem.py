"""In-memory reference implementation of :class:`core.adapter.StorageAdapter`.

Purpose: fast, dependency-free iteration on the memory logic, and the oracle the
CockroachDB adapter is tested against — both must produce the same observable
behaviour for the same operations.

Explicitly **not** a production backend and never used for published numbers: it
has no durability, no distribution and no real concurrency control beyond a
process-local lock. That is precisely the gap experiment E1 measures.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from core.adapter import AgentNotRegistered, MemoryNotFound
from core.config import DEFAULT_EMBEDDING_DIM
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

#: Statuses that are excluded from retrieval but still reachable by id for audit.
_NON_RETRIEVABLE = {MemoryStatus.SUPERSEDED, MemoryStatus.QUARANTINED, MemoryStatus.ARCHIVED}


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance in [0, 2], matching CockroachDB's ``<=>`` operator.

    A zero vector has no direction, so its distance to anything is defined as the
    maximum (2.0) rather than raising: a degenerate embedding must rank last, not
    crash a retrieval.
    """
    if len(a) != len(b):
        raise ValueError(f"vector dimension mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 2.0
    return 1.0 - (dot / (norm_a * norm_b))


class InMemoryAdapter:
    """Process-local storage implementing the frozen §7 contract."""

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        *,
        id_factory: Callable[[], str] | None = None,
        archive_callback: Callable[[MemoryEvent], None] | None = None,
    ) -> None:
        """
        Args:
            embedding_dim: enforced on every embedding written, so a dimension
                mismatch fails at the write instead of corrupting retrieval.
            id_factory: override for deterministic ids in seeded runs.
            archive_callback: invoked when a memory is archived — this is how the
                S3 offload reaches the core without the core importing boto3.
        """
        self.embedding_dim = embedding_dim
        self._new_id = id_factory or (lambda: str(uuid.uuid4()))
        self._archive_callback = archive_callback

        self._lock = threading.RLock()
        self._agents: dict[str, dict[str, str]] = {}
        self._memories: dict[str, MemoryEvent] = {}
        self._provenance: dict[str, ProvenanceLink] = {}  # mem_id -> link
        self._canonical: dict[str, CanonicalFact] = {}
        self._quarantine_log: list[QuarantineRecord] = []

    # ------------------------------------------------------------------ agents
    def register_agent(self, agent_id: str, role: str, token_hash: str) -> None:
        if not agent_id:
            raise ValueError("agent_id is required")
        with self._lock:
            self._agents[agent_id] = {
                "role": role,
                "token_hash": token_hash,
                "created_at": self._agents.get(agent_id, {}).get("created_at") or utcnow(),
            }

    def _require_agent(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise AgentNotRegistered(f"agent not registered: {agent_id!r}")

    def _require_memory(self, mem_id: str) -> MemoryEvent:
        try:
            return self._memories[mem_id]
        except KeyError:
            raise MemoryNotFound(f"no such memory: {mem_id!r}") from None

    # ------------------------------------------------------------------ writes
    def write_episode(self, agent_id: str, event: MemoryEvent) -> str:
        """Atomic in spirit: memory + embedding + provenance under one lock."""
        if event.agent_id != agent_id:
            raise ValueError(
                f"agent_id mismatch: called with {agent_id!r}, event carries {event.agent_id!r}"
            )
        if event.embedding is not None:
            self._check_dim(event.embedding)

        with self._lock:
            self._require_agent(agent_id)
            if event.parent_mem is not None and event.parent_mem not in self._memories:
                raise MemoryNotFound(f"parent memory does not exist: {event.parent_mem!r}")

            mem_id = event.mem_id or self._new_id()
            stored = replace(
                event,
                mem_id=mem_id,
                embedding=list(event.embedding) if event.embedding is not None else None,
            )
            self._memories[mem_id] = stored
            self._provenance[mem_id] = ProvenanceLink(
                link_id=self._new_id(),
                mem_id=mem_id,
                parent_mem=event.parent_mem,
                agent_id=agent_id,
                signature=event.signature or "",
                hop_count=event.hop_count,
                created_at=stored.created_at,
            )
            return mem_id

    def upsert_embedding(self, mem_id: str, vec: list[float], meta: dict) -> None:
        self._check_dim(vec)
        with self._lock:
            stored = self._require_memory(mem_id)
            stored.embedding = list(vec)
            if meta:
                stored.meta.update(meta)

    def _check_dim(self, vec: Sequence[float]) -> None:
        if len(vec) != self.embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch: got {len(vec)}, adapter expects "
                f"{self.embedding_dim} (see AletheiaConfig.embedding_dim)"
            )

    # ------------------------------------------------------------------- reads
    def read_recent(self, agent_id: str, k: int) -> list[MemoryEvent]:
        if k <= 0:
            return []
        with self._lock:
            self._require_agent(agent_id)
            own = [
                m
                for m in self._memories.values()
                if m.agent_id == agent_id and m.status not in _NON_RETRIEVABLE
            ]
        own.sort(key=lambda m: (m.created_at, m.mem_id or ""), reverse=True)
        return [self._copy(m) for m in own[:k]]

    def query_semantic(
        self, agent_id: str, vec: list[float], k: int, budget_tokens: int
    ) -> list[MemoryHit]:
        """Shared-memory semantic search, truncated to the retrieval budget.

        Results are ordered by ascending distance and accumulated until the next
        hit would exceed ``budget_tokens``; accumulation stops there rather than
        skipping ahead to smaller hits, so the truncation stays deterministic and
        identical across experimental arms (CLAUDE.md §8.7).
        """
        self._check_dim(vec)
        if k <= 0 or budget_tokens <= 0:
            return []

        with self._lock:
            self._require_agent(agent_id)
            candidates = [
                (cosine_distance(vec, m.embedding), m)
                for m in self._memories.values()
                if m.embedding is not None and m.status not in _NON_RETRIEVABLE
            ]

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

        # Retrieval is an access: it is what keeps a memory alive under the
        # forgetting budget, so it must be recorded.
        with self._lock:
            now = utcnow()
            for hit in hits:
                stored = self._memories.get(hit.mem_id)
                if stored is not None:
                    stored.access_count += 1
                    stored.last_accessed = now
        return hits

    def get_memory(self, mem_id: str) -> MemoryEvent:
        """Fetch by id regardless of status — the audit path. (Extension.)"""
        with self._lock:
            return self._copy(self._require_memory(mem_id))

    # ------------------------------------------------------ status transitions
    def supersede(self, old_mem_id: str, new_mem_id: str) -> None:
        if old_mem_id == new_mem_id:
            raise ValueError("a memory cannot supersede itself")
        with self._lock:
            old = self._require_memory(old_mem_id)
            self._require_memory(new_mem_id)
            old.status = MemoryStatus.SUPERSEDED
            old.superseded_by = new_mem_id

    def quarantine(self, mem_id: str, reason: str, detector: str, payload: dict) -> None:
        with self._lock:
            stored = self._require_memory(mem_id)
            stored.status = MemoryStatus.QUARANTINED
            self._quarantine_log.append(
                QuarantineRecord(
                    q_id=self._new_id(),
                    mem_id=mem_id,
                    reason=QuarantineReason(reason),
                    detector=detector,
                    payload=dict(payload or {}),
                )
            )

    def archive(self, mem_id: str) -> None:
        with self._lock:
            stored = self._require_memory(mem_id)
            stored.status = MemoryStatus.ARCHIVED
            snapshot = self._copy(stored)
        # Callback outside the lock: the offload is I/O and must not block writers.
        if self._archive_callback is not None:
            self._archive_callback(snapshot)

    # --------------------------------------------------------------- canonical
    def get_canonical(self, key: str) -> CanonicalFact | None:
        with self._lock:
            return self._canonical.get(key)

    def set_canonical(self, key: str, content: str, source_mem: str) -> None:
        with self._lock:
            if source_mem is not None:
                self._require_memory(source_mem)
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
        """Walk from the memory back to its root observation, nearest first."""
        chain: list[ProvenanceLink] = []
        seen: set[str] = set()
        with self._lock:
            self._require_memory(mem_id)
            cursor: str | None = mem_id
            while cursor is not None and cursor not in seen:
                seen.add(cursor)
                link = self._provenance.get(cursor)
                if link is None:
                    break
                chain.append(link)
                cursor = link.parent_mem
        return chain

    def quarantine_log(self) -> list[QuarantineRecord]:
        """Full rejection feed, newest last — powers the immune panel. (Extension.)"""
        with self._lock:
            return list(self._quarantine_log)

    # -------------------------------------------------------------------- misc
    def stats(self, agent_id: str | None = None) -> MemoryStats:
        with self._lock:
            scope = [
                m for m in self._memories.values() if agent_id is None or m.agent_id == agent_id
            ]
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
        """Hand out copies: callers must not mutate stored state by accident."""
        return replace(
            mem,
            embedding=list(mem.embedding) if mem.embedding is not None else None,
            meta=dict(mem.meta),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"InMemoryAdapter(agents={len(self._agents)}, memories={len(self._memories)}, "
            f"canonical={len(self._canonical)}, dim={self.embedding_dim})"
        )

    # Kept for symmetry with future adapters that hold connections.
    def __enter__(self) -> InMemoryAdapter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
