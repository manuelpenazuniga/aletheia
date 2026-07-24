"""The storage contract every backend must satisfy (the project plan §7). Frozen.

`InMemoryAdapter` (tests, fast iteration) and `CockroachDBAdapter` (production,
vector index, serializable transactions) both implement this protocol. The memory
logic in `core/` talks only to this interface, which is what keeps the core
portable: no boto3, no psycopg, no network calls above this line. Infrastructure
enters through an adapter or through injected callbacks — never through an import.

Semantics every implementation must honour
------------------------------------------
* **Atomicity.** ``write_episode`` persists the memory row, its embedding and its
  provenance link as one unit. A reader must never observe a memory without its
  provenance, nor a vector without its row.
* **Nothing is deleted.** ``supersede``, ``quarantine`` and ``archive`` are status
  transitions (``MemoryStatus``). The history is the audit trail.
* **Retrieval is budgeted.** ``query_semantic`` returns the best hits that fit in
  ``budget_tokens``, walking results in descending relevance and stopping before
  the budget is exceeded. The budget is identical across experimental arms.
* **Superseded, quarantined and archived memories are not retrievable** by
  ``query_semantic`` or ``read_recent``; they remain reachable by id for audit.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.models import (
    CanonicalFact,
    MemoryEvent,
    MemoryHit,
    MemoryKind,
    MemoryStats,
    MemoryStatus,
    ProvenanceLink,
)


class AdapterError(RuntimeError):
    """Base class for storage failures surfaced to the core."""


class MemoryNotFound(AdapterError):
    """Raised when an operation targets a mem_id that does not exist."""


class DuplicateMemory(AdapterError):
    """Raised when a write would collide with an existing mem_id.

    Mirrors the ``memories`` PRIMARY KEY: overwriting an existing memory would
    destroy it and its provenance, which "nothing is deleted" forbids.
    """


class AgentNotRegistered(AdapterError):
    """Raised when a write references an unknown agent.

    Mirrors the ``memories.agent_id -> agents.agent_id`` foreign key: an
    unattributable memory is not accepted by the memory layer, ever.
    """


@runtime_checkable
class StorageAdapter(Protocol):
    """Storage backend contract. The signatures below are frozen by the project plan §7."""

    # -- §7 contract ----------------------------------------------------------
    def write_episode(self, agent_id: str, event: MemoryEvent) -> str:
        """Persist one memory (+ embedding + provenance) atomically. Returns mem_id."""
        ...

    def read_recent(self, agent_id: str, k: int) -> list[MemoryEvent]:
        """The agent's ``k`` most recent active memories, newest first."""
        ...

    def upsert_embedding(self, mem_id: str, vec: list[float], meta: dict) -> None:
        """Attach or replace the embedding of an existing memory."""
        ...

    def query_semantic(
        self, agent_id: str, vec: list[float], k: int, budget_tokens: int
    ) -> list[MemoryHit]:
        """Semantic search over shared fleet memory, truncated to the token budget.

        ``agent_id`` is the *caller*, not a filter: institutional memory is shared.
        It is carried for auditing and for agent-relative signals (own history).
        """
        ...

    def supersede(self, old_mem_id: str, new_mem_id: str) -> None:
        """Knowledge-update: mark ``old`` superseded by ``new``, keeping the chain."""
        ...

    def get_canonical(self, key: str) -> CanonicalFact | None:
        """Current version of a canonical fact, or None."""
        ...

    def set_canonical(self, key: str, content: str, source_mem: str) -> None:
        """Create or bump a canonical fact, incrementing ``version``."""
        ...

    def provenance_chain(self, mem_id: str) -> list[ProvenanceLink]:
        """Provenance from the memory back to its root observation, nearest first."""
        ...

    def quarantine(self, mem_id: str, reason: str, detector: str, payload: dict) -> None:
        """Immune rejection: status -> quarantined, plus a ``quarantine_log`` entry."""
        ...

    def archive(self, mem_id: str) -> None:
        """Metabolic forgetting: status -> archived (content offloaded to S3 by a callback)."""
        ...

    def stats(self, agent_id: str | None) -> MemoryStats:
        """Footprint and counts; fleet-wide when ``agent_id`` is None."""
        ...

    # -- extensions beyond §7 -------------------------------------------------
    # Required by the `memories.agent_id -> agents.agent_id` foreign key in
    # infra/ddl.sql: a memory cannot exist without a registered author. Kept
    # explicit rather than auto-creating agents on write, because silent agent
    # creation would defeat the provenance guarantee the immune system rests on.
    def register_agent(self, agent_id: str, role: str, token_hash: str) -> None:
        """Register (or update) an agent authorised to write memory."""
        ...

    def close(self) -> None:
        """Release backend resources. No-op for in-memory storage."""
        ...

    # Required by the consolidation cycle (C2) and metabolic forgetting (C3):
    # both must enumerate the fleet's memories to compute the active footprint,
    # rank prune candidates, and bucket episodes by fact key. A single shared
    # extension serves both — do not add a second listing method.
    def list_memories(
        self,
        *,
        status: MemoryStatus | str | None = MemoryStatus.ACTIVE,
        kind: MemoryKind | str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryEvent]:
        """Enumerate stored memories, filtered conjunctively by status/kind/agent.

        Filters are conjunctive; ``None`` on an axis means "no filter" (any
        status, any kind, fleet-wide). ``status`` and ``kind`` accept the enum or
        its string value, coerced the way :class:`~core.models.MemoryEvent` does.
        Returns COPIES (never internal references), consistent with ``get_memory``
        and ``read_recent``, in deterministic ascending ``(created_at, mem_id)``
        order so consolidation and forgetting are reproducible.

        Extension beyond §7. Backed by ``idx_mem_agent_status`` on the production
        backend; the ``agent_id`` filter is there for future per-agent budgets.
        """
        ...
