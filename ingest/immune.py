"""The immune-gate seam (C5) — the write-path adapter over the core detectors.

The ingest service inspects every write through an :class:`ImmuneGate` before it
is persisted, but *only* when ``config.enable_immune`` is set. The seam has two
implementations: :class:`PassthroughImmuneGate` (allows everything — kept for the
disabled/ablation path and tests) and :class:`CoreImmuneGate`, which delegates to
the portable :class:`core.immune.ImmuneSystem` and translates its verdict into the
HTTP-facing :class:`ImmuneVerdict`.

Kept out of ``core/`` because it is part of the write path, but it speaks only
core types (:class:`~core.models.MemoryEvent`) so the real detector logic stays
portable and unit-testable — all of it lives in :mod:`core.immune`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.adapter import MemoryNotFound, StorageAdapter
from core.embeddings import Embedder
from core.immune import ImmuneSystem
from core.models import MemoryEvent, MemoryStatus


@dataclass(frozen=True)
class ImmuneVerdict:
    """The gate's decision on one write.

    ``allowed=False`` carries the ``reason`` and ``detector`` that the endpoint
    surfaces (and that Phase 2 will persist to ``quarantine_log``).
    """

    allowed: bool
    reason: str | None = None
    detector: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ImmuneGate(Protocol):
    """Inspect a provisional memory and allow or reject it."""

    def inspect(self, event: MemoryEvent) -> ImmuneVerdict: ...


class PassthroughImmuneGate:
    """Allows every write. Used for the disabled path, ablations, and tests."""

    def inspect(self, event: MemoryEvent) -> ImmuneVerdict:
        return ImmuneVerdict(allowed=True)


class CoreImmuneGate:
    """Real gate: delegates to :class:`core.immune.ImmuneSystem`.

    Translates the core verdict (``accepted`` + a :class:`QuarantineReason` enum)
    into this module's HTTP-facing :class:`ImmuneVerdict` (``allowed`` + the enum
    *value* string that the endpoint persists to ``quarantine_log`` and logs). All
    detection logic lives in the portable core; this class only adapts types.
    """

    def __init__(self, immune: ImmuneSystem) -> None:
        self._immune = immune

    def inspect(self, event: MemoryEvent) -> ImmuneVerdict:
        verdict = self._immune.inspect(event)
        return ImmuneVerdict(
            allowed=verdict.accepted,
            reason=verdict.reason.value if verdict.reason is not None else None,
            detector=verdict.detector or None,
            payload=dict(verdict.payload),
        )


def persist_quarantine(
    adapter: StorageAdapter,
    embedder: Embedder,
    event: MemoryEvent,
    *,
    reason: str,
    detector: str,
    payload: dict[str, Any],
) -> str:
    """Write a rejected memory already-QUARANTINED, then log the rejection.

    The single audit path shared by every write surface (the ingest HTTP endpoint
    and the offline fleet writer) so a caught attack is recorded IDENTICALLY
    wherever it enters: auditability over deletion (the project plan §6a). The
    content is retained for the immune panel but is permanently non-retrievable
    (QUARANTINED is excluded from retrieval from the instant the row exists).

    The embedding is recomputed here by the trusted embedder, never taken from the
    client. A claimed parent that does not exist is stripped from the persisted
    copy (else ``write_episode`` re-raises MemoryNotFound on the FK and the
    rejection could not be audited), with the claim preserved in the payload — for
    EVERY rejection, not just a bad-parent one, so an attack that trips another
    detector while also carrying a fake parent is still audited.

    Returns the quarantined memory's id.
    """
    payload = dict(payload)
    parent_mem = event.parent_mem
    if parent_mem is not None and not _memory_exists(adapter, parent_mem):
        payload.setdefault("claimed_parent", parent_mem)
        parent_mem = None

    quarantined = MemoryEvent(
        agent_id=event.agent_id,
        content=event.content,
        kind=event.kind,
        embedding=embedder.embed(event.content),
        importance=event.importance,
        status=MemoryStatus.QUARANTINED,
        parent_mem=parent_mem,
        hop_count=event.hop_count,
        signature=event.signature,
        meta=dict(event.meta),
    )
    mem_id = adapter.write_episode(event.agent_id, quarantined)
    adapter.quarantine(mem_id, reason, detector, payload)
    return mem_id


def _memory_exists(adapter: StorageAdapter, mem_id: str) -> bool:
    """True if a memory with this id exists (any status), via the audit read."""
    try:
        adapter.get_memory(mem_id)
    except MemoryNotFound:
        return False
    return True
