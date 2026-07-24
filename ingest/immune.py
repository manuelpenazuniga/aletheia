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

from core.immune import ImmuneSystem
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
