"""The immune-gate seam (C5) — a passthrough in Phase 1.

The ingest service inspects every write through an :class:`ImmuneGate` before it
is persisted, but *only* when ``config.enable_immune`` is set. Phase 1 ships
:class:`PassthroughImmuneGate`, which allows everything: the seam exists so
Phase 2 can drop in the real detector (provenance validation, semantic-anomaly
distance, injection patterns) without touching the endpoint.

Kept out of ``core/`` because it is part of the write path, but it speaks only
core types (:class:`~core.models.MemoryEvent`) so the future real gate is
portable and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.models import MemoryEvent


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
    """Phase-1 gate: allows every write. The seam, without the detector yet."""

    def inspect(self, event: MemoryEvent) -> ImmuneVerdict:
        return ImmuneVerdict(allowed=True)
