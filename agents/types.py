"""Value types crossing the agent loop and the fleet runner.

These are the small, portable dataclasses the SRE fleet speaks in: an
:class:`Incident` handed to an agent, the :class:`Diagnosis` a model produces, the
:class:`AgentStep` one turn of the loop returns, and the :class:`AuditRecord` /
:class:`FleetRunResult` the runner accumulates into a queryable trail.

Deliberately distinct from :class:`scenarios.loader.Incident`: that type is the
committed *corpus* record (a runbook plus its revisions, the knowledge-update
signal E3 keys on). This :class:`Incident` is the *runtime* task an agent is asked
to work — title, free-text description and observed signals. A caller that wants
to drive the fleet over the committed corpus maps one to the other at the call
site (in ``scenarios``/``experiments``); ``agents`` never imports ``scenarios``,
so the generation seam can reuse ``agents.model_client`` without an import cycle.

Portable: speaks only stdlib + :mod:`core.models`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from core.models import MemoryKind

#: The three audit operations one incident produces: exactly one ``recall`` per
#: turn, followed by either a ``remember`` (a memory was persisted) or a
#: ``reject`` (the immune gate refused the write before anything was persisted).
AuditOp = Literal["recall", "remember", "reject"]


@dataclass(frozen=True, slots=True)
class Incident:
    """One runtime task for an SRE agent to diagnose.

    ``signals`` are the raw observations (log lines, alerts) the agent perceives;
    ``description`` is the free-text summary used when no signals are present.
    ``meta`` carries routing labels (scenario id, source corpus id) that flow onto
    the resulting memory but never change the loop's behaviour.
    """

    incident_id: str
    title: str
    description: str = ""
    signals: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """What the model concluded about one incident.

    ``learned_fact`` is the durable institutional lesson written to memory; it is
    always non-empty (the loop guarantees a fallback), because
    :class:`~core.models.MemoryEvent` forbids empty content.
    """

    summary: str
    action: str
    learned_fact: str
    importance: float = 0.5
    kind: MemoryKind = MemoryKind.EPISODIC


@dataclass(frozen=True, slots=True)
class AgentStep:
    """The outcome of one agent handling one incident.

    On a successful write ``mem_id`` is set and ``rejected`` is False. When the
    immune gate refuses the write, ``rejected`` is True, ``mem_id`` is None (the
    reject happens before any persist), and ``reason`` names the detector verdict.
    """

    agent_id: str
    role: str
    incident_id: str
    recalled_ids: list[str]
    diagnosis: Diagnosis | None
    mem_id: str | None
    rejected: bool = False
    reason: str | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One monotonically-sequenced entry in the fleet's audit trail.

    ``seq`` increases by one per record across the whole run, so the trail can be
    ordered and diffed deterministically. ``latency_ms`` and ``ts`` are
    observational (wall-clock) and are the only non-reproducible fields.
    """

    seq: int
    agent_id: str
    role: str
    incident_id: str
    op: AuditOp
    mem_id: str | None
    recalled_ids: list[str]
    reason: str | None
    latency_ms: float
    ts: datetime


@dataclass(frozen=True, slots=True)
class FleetRunResult:
    """Everything one :func:`agents.fleet.run_fleet` call produced.

    ``writes``/``rejects``/``recalls`` are tallies over ``steps``; ``audit`` is the
    full, ordered trail (two records per incident: a recall, then a remember or a
    reject).
    """

    steps: list[AgentStep]
    audit: list[AuditRecord]
    writes: int
    rejects: int
    recalls: int
