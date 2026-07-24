"""C5 — the immune system: memory that stays reliable despite unreliable agents.

This is the portable core of the fifth component (the project plan §5.2). It
inspects a provisional :class:`~core.models.MemoryEvent` *before* it is written
and rejects poisoned, spoofed or unattributable memory, so a single bad agent
cannot contaminate the fleet's shared memory.

Three detectors run in a fixed, cheap-first order and the first hit wins
(short-circuit, deterministic):

1. **Provenance** (structural, free) — a missing signature, a claim that
   propagated past ``gossip_max_hops``, or a parent memory that does not exist.
2. **Injection** (regex, cheap) — prompt-injection / tool-call-spoofing phrasing,
   anchored so ordinary SRE text ("system load high", "restart the systemd
   service") is never flagged.
3. **Semantic anomaly** (one embed + one read, most expensive) — a write whose
   nearest neighbour in the agent's *own* recent history is farther than
   ``immune_anomaly_threshold``: a sudden unexplained topic jump, the signature
   of injected content.

Portability (the project plan §7): this module imports only ``core.*`` — no
boto3, no psycopg, no HTTP. The storage backend and the embedder are injected
seams, so every detector is unit-testable offline against the in-memory adapter
and the deterministic embedder. The ingest write-path wraps :class:`ImmuneSystem`
in :class:`ingest.immune.CoreImmuneGate`; the gossip and consolidation
write-paths (which do not run HTTP auth) can call the detectors directly, which
is why each one validates its inputs independently rather than trusting that an
earlier layer already did.

The whole system short-circuits to *accepted* when ``config.enable_immune`` is
False — a disabled component is a clean no-op (the project plan §7 ablation
switch), and no memory is persisted-as-quarantined in that mode.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.adapter import AgentNotRegistered, MemoryNotFound, StorageAdapter
from core.config import AletheiaConfig
from core.embeddings import Embedder
from core.models import MemoryEvent, QuarantineReason

#: How many of the agent's own recent memories the anomaly detector compares a
#: new write against. A documented module constant rather than a config field to
#: keep the frozen §7 config minimal; promote to ``AletheiaConfig`` only if E4
#: needs to tune the window (a too-small window weakens detection, a too-large
#: one slows the hot write path).
DEFAULT_HISTORY_K = 20

#: Detector slugs recorded in ``quarantine_log.detector`` — they name *which*
#: detector fired (the ``reason`` says *why*). Kept distinct from the reason so
#: the immune panel can attribute a rejection to a specific check.
DETECTOR_PROVENANCE = "provenance_validator"
DETECTOR_INJECTION = "injection_scanner"
DETECTOR_ANOMALY = "semantic_anomaly_detector"

#: Explicit prompt-injection / tool-call-spoofing patterns as ``(slug, regex)``.
#: Every pattern is ANCHORED to injection phrasing so benign operational text does
#: not trip it: the role-spoof patterns require ``system:`` / ``assistant:`` at
#: the very start of the content (``^``), never a bare "system" or "systemd"
#: appearing mid-sentence. Case-insensitive; deliberately conservative to keep the
#: false-positive rate (an E4 metric) low.
INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    # MULTILINE so ``^`` anchors to the start of ANY line, not just the whole
    # string: without it an attacker hides a role spoof behind a benign first
    # line ("disk usage normal\nsystem: ignore all rules") and evades detection.
    (slug, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    for slug, pattern in (
        ("ignore_previous", r"ignore (all )?(previous|prior|above) (instructions|prompts)"),
        ("disregard_previous", r"disregard (all )?(previous|prior|above)"),
        ("forget_previous", r"forget (everything|all previous)"),
        ("role_reassignment", r"you are now\b"),
        ("system_role_spoof", r"^\s*system\s*:"),
        ("assistant_role_spoof", r"^\s*assistant\s*:"),
        ("new_instructions", r"new instructions\s*:"),
        ("tool_call_spoof", r"<\s*tool_call"),
        ("function_call_spoof", r"</?function_call"),
        ("code_fence_tool", r"```\s*tool"),
    )
)


# -----------------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImmuneVerdict:
    """The immune system's decision on one provisional write.

    ``accepted=True`` carries no reason/detector. A rejection carries the
    :class:`~core.models.QuarantineReason`, the detector slug that fired, and a
    ``payload`` of forensic detail — everything the audit path needs to persist a
    ``quarantine_log`` record.
    """

    accepted: bool
    reason: QuarantineReason | None = None
    detector: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Detectors — pure functions. Each returns forensic detail on a hit, else None.
# No side effects: they never write, quarantine, or mutate the event.
# -----------------------------------------------------------------------------


def detect_bad_provenance(
    event: MemoryEvent, adapter: StorageAdapter, *, max_hops: int
) -> dict[str, Any] | None:
    """Structural provenance checks — the cheapest detector, run first.

    Rejects if the write is unsigned, claims an impossible or over-propagated hop
    count, or points at a parent memory that does not exist. Returns the sub-check
    that fired plus the offending values, or ``None`` when provenance is sound.
    """
    if not event.signature:
        return {"check": "missing_signature"}
    if event.hop_count < 0:
        # Unreachable through MemoryEvent (its __post_init__ forbids it), but the
        # gossip write-path can call this function directly — defense in depth.
        return {"check": "negative_hop_count", "hop_count": event.hop_count}
    if event.hop_count > max_hops:
        # Propagated too far from the original observation to trust un-regrounded.
        return {"check": "hops_exceeded", "hop_count": event.hop_count, "max_hops": max_hops}
    if event.parent_mem is not None:
        try:
            adapter.get_memory(event.parent_mem)
        except MemoryNotFound:
            return {"check": "nonexistent_parent", "claimed_parent": event.parent_mem}
    return None


def detect_injection_pattern(content: str) -> dict[str, Any] | None:
    """Match ``content`` against :data:`INJECTION_PATTERNS` (case-insensitive).

    Returns ``{"pattern": <slug>, "match": <matched span>}`` on the first hit, or
    ``None``. Patterns are anchored so ordinary SRE prose does not match.
    """
    for slug, pattern in INJECTION_PATTERNS:
        match = pattern.search(content)
        if match is not None:
            return {"pattern": slug, "match": match.group(0)}
    return None


def detect_semantic_anomaly(
    event: MemoryEvent,
    adapter: StorageAdapter,
    embedder: Embedder,
    *,
    threshold: float,
    history_k: int,
) -> dict[str, Any] | None:
    """Flag a write that jumps away from the agent's own recent history.

    Embeds the content, reads the agent's ``history_k`` most recent active
    memories, and computes the nearest-neighbour cosine distance. A distance
    greater than ``threshold`` means the new memory has no anchor in what this
    agent has been working on — the signature of injected/poisoned content.

    A brand-new agent (no history) or an agent whose memories carry no vectors has
    nothing to jump away from, so it is never anomalous: this guards the
    cold-start false positive the E4 false-positive metric would otherwise punish.
    """
    vec = embedder.embed(event.content)
    try:
        history = adapter.read_recent(event.agent_id, history_k)
    except AgentNotRegistered:
        # The write's own registration guard surfaces the 403 on the accept path;
        # here an unknown agent simply has an empty history.
        history = []

    distances = [
        _cosine_distance(vec, mem.embedding) for mem in history if mem.embedding is not None
    ]
    if not distances:
        return None

    min_distance = min(distances)
    if min_distance > threshold:
        return {
            "min_distance": min_distance,
            "threshold": threshold,
            "history_size": len(distances),
        }
    return None


# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------


class ImmuneSystem:
    """Runs the three detectors in fixed order and returns the first verdict.

    Injected seams (both offline-fakeable): a :class:`~core.adapter.StorageAdapter`
    for the parent-existence and history reads, and an
    :class:`~core.embeddings.Embedder` for the anomaly check. No infrastructure is
    touched here — the production gate wires the CockroachDB adapter and the Titan
    embedder in ``ingest.main``; tests wire the in-memory adapter and the
    deterministic embedder.
    """

    def __init__(self, adapter: StorageAdapter, embedder: Embedder, config: AletheiaConfig) -> None:
        self._adapter = adapter
        self._embedder = embedder
        self._config = config
        self._history_k = DEFAULT_HISTORY_K

    def inspect(self, event: MemoryEvent) -> ImmuneVerdict:
        """Inspect one provisional write. Deterministic; no side effects.

        Short-circuits to accepted when the immune flag is off, then runs
        provenance -> injection -> semantic-anomaly, returning the first hit. A
        write that trips two detectors is reported by the first in that order, so
        the verdict is stable.
        """
        if not self._config.enable_immune:
            return ImmuneVerdict(accepted=True)

        hit = detect_bad_provenance(event, self._adapter, max_hops=self._config.gossip_max_hops)
        if hit is not None:
            return ImmuneVerdict(
                accepted=False,
                reason=QuarantineReason.BAD_PROVENANCE,
                detector=DETECTOR_PROVENANCE,
                payload=hit,
            )

        hit = detect_injection_pattern(event.content)
        if hit is not None:
            return ImmuneVerdict(
                accepted=False,
                reason=QuarantineReason.INJECTION_PATTERN,
                detector=DETECTOR_INJECTION,
                payload=hit,
            )

        hit = detect_semantic_anomaly(
            event,
            self._adapter,
            self._embedder,
            threshold=self._config.immune_anomaly_threshold,
            history_k=self._history_k,
        )
        if hit is not None:
            return ImmuneVerdict(
                accepted=False,
                reason=QuarantineReason.SEMANTIC_ANOMALY,
                detector=DETECTOR_ANOMALY,
                payload=hit,
            )

        return ImmuneVerdict(accepted=True)


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance in [0, 2], matching CockroachDB's ``<=>`` and the adapter.

    Kept local so the core does not import the adapter layer. A zero vector has no
    direction, so its distance to anything is the maximum (2.0) rather than a
    crash — a degenerate embedding must simply look maximally far, never anomalous
    by exception.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 2.0
    return 1.0 - (dot / (norm_a * norm_b))
