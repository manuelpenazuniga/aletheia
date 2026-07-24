"""Core domain models (CLAUDE.md §7). Frozen contract.

These are the only types that cross the boundary between the memory logic and a
:class:`core.adapter.StorageAdapter`. They mirror the schema in infra/ddl.sql
one-to-one, so a change here is a schema migration and vice versa.

Portable core: no boto3, no psycopg, no network.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

# -----------------------------------------------------------------------------
# Enumerations — the string values are what lands in the database.
# -----------------------------------------------------------------------------


class MemoryKind(StrEnum):
    """`memories.kind`."""

    EPISODIC = "episodic"  # raw observation from an agent working an incident
    SEMANTIC = "semantic"  # generalisation produced by the consolidation cycle
    CANONICAL_REF = "canonical_ref"  # pointer to a promoted canonical fact


class MemoryStatus(StrEnum):
    """`memories.status`.

    Nothing is ever deleted: quarantine and supersede are status transitions, so
    the audit trail survives (CLAUDE.md §6 rule (a)).
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"  # knowledge-update: a newer memory replaced it
    QUARANTINED = "quarantined"  # immune system rejected it
    ARCHIVED = "archived"  # metabolic forgetting moved it to S3


class QuarantineReason(StrEnum):
    """`quarantine_log.reason`."""

    BAD_PROVENANCE = "bad_provenance"
    SEMANTIC_ANOMALY = "semantic_anomaly"
    INJECTION_PATTERN = "injection_pattern"


class AgentRole(StrEnum):
    """`agents.role`."""

    SRE = "sre"
    OPS = "ops"
    ADVERSARY = "adversary"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def utcnow() -> datetime:
    """Timezone-aware UTC now. Every timestamp in the system uses this."""
    return datetime.now(UTC)


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate used for retention cost and retrieval budget.

    An approximation (~4 characters per token) on purpose: it must be identical
    across every experimental arm and reproducible offline, so that a budget
    comparison measures design rather than tokenizer drift. Real provider token
    counts are recorded separately when available.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryEvent:
    """One unit of memory: what an agent learned, plus how it came to know it.

    Produced by an agent, validated by the immune gate, then written as a row in
    `memories` plus a link in `provenance` — in a single serializable
    transaction, so a vector never exists without its operational row.
    """

    agent_id: str
    content: str
    kind: MemoryKind = MemoryKind.EPISODIC
    embedding: Sequence[float] | None = None
    importance: float = 0.5
    cost_tokens: int | None = None  # None -> derived from content on write
    status: MemoryStatus = MemoryStatus.ACTIVE

    # Assigned by storage on write.
    mem_id: str | None = None
    superseded_by: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    last_accessed: datetime = field(default_factory=utcnow)
    access_count: int = 0

    # Provenance carried alongside the write (becomes a `provenance` row).
    parent_mem: str | None = None  # None = direct observation
    signature: str | None = None  # HMAC(agent token, content)
    hop_count: int = 0  # gossip hops away from the original observation

    # Non-persisted routing/debug metadata (incident id, scenario, arm...).
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("MemoryEvent.agent_id is required")
        if not self.content:
            raise ValueError("MemoryEvent.content is required")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("MemoryEvent.importance must be in [0.0, 1.0]")
        if self.hop_count < 0:
            raise ValueError("MemoryEvent.hop_count must be non-negative")
        if self.cost_tokens is None:
            self.cost_tokens = estimate_tokens(self.content)
        elif self.cost_tokens <= 0:
            # A non-positive retention cost is nonsensical and actively dangerous:
            # a negative cost reads as credit in the retrieval budget, letting a
            # single poisoned memory pull in unlimited others, and it corrupts the
            # forgetting footprint. Reject it at construction.
            raise ValueError("MemoryEvent.cost_tokens must be positive")
        # Enum coercion, so callers may pass plain strings from JSON payloads.
        self.kind = MemoryKind(self.kind)
        self.status = MemoryStatus(self.status)

    @property
    def is_direct_observation(self) -> bool:
        """A memory with no parent and no hops is first-hand evidence."""
        return self.parent_mem is None and self.hop_count == 0

    def with_id(self, mem_id: str) -> MemoryEvent:
        return replace(self, mem_id=mem_id)


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """A retrieval result: a memory plus why it was returned."""

    mem_id: str
    agent_id: str
    content: str
    kind: MemoryKind
    score: float  # similarity in [0, 1]; higher is closer
    distance: float  # raw vector distance from the backend
    cost_tokens: int
    importance: float = 0.5
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime | None = None
    hop_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CanonicalFact:
    """The fleet's shared source of truth for one key.

    `version` increments on every knowledge-update; the version history is what
    the demo app renders as the timeline of a belief.
    """

    fact_key: str  # e.g. 'runbook:db-latency:fix'
    content: str
    version: int = 1
    source_mem: str | None = None
    updated_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True, slots=True)
class ProvenanceLink:
    """One edge of the provenance graph: who claimed this, and from whom.

    `hop_count` is the degradation distance from direct observation — the signal
    the immune system and consolidation both lean on.
    """

    mem_id: str
    agent_id: str
    signature: str
    link_id: str | None = None
    parent_mem: str | None = None
    hop_count: int = 0
    created_at: datetime = field(default_factory=utcnow)

    @property
    def is_root(self) -> bool:
        return self.parent_mem is None


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    """An immune-system rejection, kept for audit (`quarantine_log`)."""

    reason: QuarantineReason
    detector: str
    mem_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    q_id: str | None = None
    created_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True, slots=True)
class MemoryStats:
    """Footprint of the memory layer — the observable behind cost control (C3).

    Scoped to one agent, or to the whole fleet when ``agent_id`` is None.
    """

    agent_id: str | None
    total_memories: int
    active: int
    superseded: int
    quarantined: int
    archived: int
    total_cost_tokens: int
    canonical_facts: int = 0
    oldest_created_at: datetime | None = None
    newest_created_at: datetime | None = None

    @property
    def footprint_mb(self) -> float:
        """Rough retained-content size in MB (4 bytes per estimated token)."""
        return round(self.total_cost_tokens * 4 / 1_048_576, 4)
