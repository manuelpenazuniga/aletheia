"""The fleet runner + an audit trail.

:func:`run_fleet` drives a set of agents over a set of incidents **sequentially
and deterministically**, accumulating a monotonically-sequenced audit trail. It is
intentionally single-threaded: real concurrency is experiment E1's concern, and
threading here would make the trail non-reproducible. Because the agents share one
backing store, a later agent recalls what an earlier agent wrote — institutional
memory, demonstrated end to end.

Two convenience builders (:func:`make_local_sre_agent`, :func:`make_local_adversary`)
wire a fully-offline agent over a shared :class:`~core.memory.MemoryService`: a
per-agent :class:`~agents.signing.HmacSigner`, a
:class:`~agents.write_client.LocalMemoryWriter` (with the optional immune gate) and
a :class:`~agents.read_client.LocalMemoryReader`. Production wiring swaps in the
Bedrock model, the MCP reader and the HTTP writer without touching the loop.

Portable: :mod:`core` + sibling ``agents`` modules only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from agents.adversary import DEFAULT_TARGET_FACT_KEY, AdversaryAgent
from agents.agent import SREAgent
from agents.model_client import ModelClient
from agents.read_client import LocalMemoryReader
from agents.signing import HmacSigner
from agents.types import AuditRecord, FleetRunResult
from agents.write_client import ImmuneGate, LocalMemoryWriter
from core.config import AletheiaConfig
from core.memory import MemoryService
from core.models import AgentRole, utcnow

Assignment = Literal["round_robin", "broadcast"]


def run_fleet(
    agents: Sequence[SREAgent],
    incidents: Sequence[object],
    *,
    assignment: Assignment = "round_robin",
) -> FleetRunResult:
    """Run ``agents`` over ``incidents`` and return the steps + audit trail.

    ``round_robin`` assigns incident ``i`` to ``agents[i % len(agents)]``;
    ``broadcast`` has every agent handle every incident. Every agent is registered
    before any write. Two records are appended per incident: a ``recall``, then a
    ``remember`` (on a persisted write) or a ``reject`` (on a caught poison).
    """
    if not agents:
        raise ValueError("run_fleet requires at least one agent")

    for agent in agents:
        agent.ensure_registered()

    steps = []
    audit: list[AuditRecord] = []
    seq = 0
    writes = rejects = recalls = 0

    for agent, incident in _schedule(agents, incidents, assignment):
        step = agent.handle_incident(incident)  # type: ignore[arg-type]
        steps.append(step)
        recalls += 1

        seq += 1
        audit.append(
            AuditRecord(
                seq=seq,
                agent_id=agent.agent_id,
                role=agent.role,
                incident_id=step.incident_id,
                op="recall",
                mem_id=None,
                recalled_ids=list(step.recalled_ids),
                reason=None,
                latency_ms=step.latency_ms,
                ts=utcnow(),
            )
        )

        seq += 1
        if step.rejected:
            rejects += 1
            audit.append(
                AuditRecord(
                    seq=seq,
                    agent_id=agent.agent_id,
                    role=agent.role,
                    incident_id=step.incident_id,
                    op="reject",
                    mem_id=None,
                    recalled_ids=list(step.recalled_ids),
                    reason=step.reason,
                    latency_ms=step.latency_ms,
                    ts=utcnow(),
                )
            )
        else:
            writes += 1
            audit.append(
                AuditRecord(
                    seq=seq,
                    agent_id=agent.agent_id,
                    role=agent.role,
                    incident_id=step.incident_id,
                    op="remember",
                    mem_id=step.mem_id,
                    recalled_ids=list(step.recalled_ids),
                    reason=None,
                    latency_ms=step.latency_ms,
                    ts=utcnow(),
                )
            )

    return FleetRunResult(steps=steps, audit=audit, writes=writes, rejects=rejects, recalls=recalls)


def _schedule(
    agents: Sequence[SREAgent], incidents: Sequence[object], assignment: Assignment
) -> list[tuple[SREAgent, object]]:
    if assignment == "round_robin":
        return [(agents[i % len(agents)], inc) for i, inc in enumerate(incidents)]
    if assignment == "broadcast":
        return [(agent, inc) for inc in incidents for agent in agents]
    raise ValueError(f"unknown assignment {assignment!r}; use 'round_robin' or 'broadcast'")


# -----------------------------------------------------------------------------
# Offline builders — one shared MemoryService, one gate, per-agent signers.
# -----------------------------------------------------------------------------


def make_local_sre_agent(
    agent_id: str,
    memory_service: MemoryService,
    model: ModelClient,
    config: AletheiaConfig,
    *,
    secret: str,
    role: str = AgentRole.SRE.value,
    gate: ImmuneGate | None = None,
) -> SREAgent:
    """Build a fully-offline SRE agent over the shared ``memory_service``."""
    signer = HmacSigner.for_agent(secret, agent_id)
    writer = LocalMemoryWriter(memory_service, signer, config, immune_gate=gate)
    reader = LocalMemoryReader(memory_service)
    return SREAgent(agent_id, role, reader, writer, model, config)


def make_local_adversary(
    agent_id: str,
    memory_service: MemoryService,
    model: ModelClient,
    config: AletheiaConfig,
    *,
    secret: str,
    attack: str,
    gate: ImmuneGate | None = None,
    target_fact_key: str = DEFAULT_TARGET_FACT_KEY,
) -> AdversaryAgent:
    """Build a fully-offline adversary agent over the shared ``memory_service``."""
    signer = HmacSigner.for_agent(secret, agent_id)
    writer = LocalMemoryWriter(memory_service, signer, config, immune_gate=gate)
    reader = LocalMemoryReader(memory_service)
    return AdversaryAgent(
        agent_id, reader, writer, model, config, attack=attack, target_fact_key=target_fact_key
    )
