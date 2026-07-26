"""The demo world — a populated memory layer the dashboard reads from.

Built entirely offline (an :class:`~adapters.memory_inmem.InMemoryAdapter` + the
seeded :class:`~core.embeddings.DeterministicEmbedder`), from the SAME committed
scenarios and the SAME core cycles the real system runs. It is a real memory
layer with real content — a consolidated runbook whose obsolete version was
``supersede``d and promoted to a canonical fact, a poisoned write the immune
system ``quarantine``d, and a spread of fleet memories — just without Bedrock or a
cluster. In production the app is handed a CockroachDBAdapter + a Bedrock embedder
instead (see :func:`demoapp.app.create_app`); nothing else changes.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from adapters.memory_inmem import InMemoryAdapter
from core.config import AletheiaConfig
from core.consolidation import consolidate
from core.embeddings import DeterministicEmbedder, Embedder
from core.immune import ImmuneSystem
from core.models import MemoryEvent, MemoryKind
from ingest.immune import persist_quarantine
from scenarios.loader import incident_to_events, load_incidents, load_poison_cases

DEMO_DIM = 128  # enough lexical structure for the search view; offline + fast


@dataclass(frozen=True, slots=True)
class DemoWorld:
    adapter: InMemoryAdapter
    embedder: Embedder
    config: AletheiaConfig


def build_demo_world(*, incident_count: int = 8, poison_count: int = 4) -> DemoWorld:
    """Populate an in-memory world: fleet memories, a knowledge-update, a quarantine."""
    config = AletheiaConfig(embedding_dim=DEMO_DIM)
    embedder = DeterministicEmbedder(dim=DEMO_DIM, seed=0)
    adapter = InMemoryAdapter(embedding_dim=DEMO_DIM)

    incidents = load_incidents()[:incident_count]

    # 1. Write each incident's runbook v1 and its revisions as episodic memories
    #    (same author, increasing timestamps). meta['fact_key'] ties the versions
    #    together so the consolidation cycle can supersede the obsolete ones.
    for inc in incidents:
        for event in incident_to_events(inc, embedder):
            _ensure_agent(adapter, event.agent_id, "sre")
            adapter.write_episode(event.agent_id, event)

    # 2. Run the REAL consolidation cycle: obsolete runbook versions are
    #    superseded (a knowledge-update), the current one promoted to a canonical
    #    fact with a version history — the "git of beliefs" the memory view shows.
    consolidate(adapter, config)

    # 3. A few benign fleet observations from distinct SRE agents, so the fleet
    #    view has more than one active author.
    fleet = ["sre-01", "sre-02", "sre-03"]
    observations = [
        ("sre-01", "p99 read latency on shard 4 climbed to 480ms after the 14:10 deploy"),
        ("sre-02", "checkout error rate back to baseline after rolling back to the green build"),
        ("sre-03", "disk pressure on node 7 cleared once log rotation was re-enabled"),
        ("sre-01", "pgbouncer pool saturated at 200 connections; raised the ceiling to 400"),
        ("sre-02", "cert on the api gateway expired; rotated via ACM and redeployed the listener"),
    ]
    for aid in fleet:
        _ensure_agent(adapter, aid, "sre")
    for aid, content in observations:
        adapter.write_episode(
            aid,
            MemoryEvent(
                agent_id=aid,
                content=content,
                kind=MemoryKind.EPISODIC,
                embedding=embedder.embed(content),
            ),
        )

    # 4. The immune system in action: run committed poison cases through the real
    #    detectors; a rejection is persisted QUARANTINED (non-retrievable) and
    #    logged — exactly what the ingest write path does, so the immune feed is
    #    real evidence, not a mock.
    immune = ImmuneSystem(adapter, embedder, config)
    for case in load_poison_cases(category="injection_pattern")[:poison_count]:
        _ensure_agent(adapter, case.agent_id, "adversary")
        event = MemoryEvent(
            agent_id=case.agent_id,
            content=case.content,
            kind=MemoryKind.EPISODIC,
            signature=case.provenance.signature or "unsigned",
        )
        verdict = immune.inspect(event)
        if not verdict.accepted:
            persist_quarantine(
                adapter,
                embedder,
                event,
                reason=(verdict.reason.value if verdict.reason else ""),
                detector=verdict.detector or "",
                payload=dict(verdict.payload),
            )

    return DemoWorld(adapter=adapter, embedder=embedder, config=config)


def _ensure_agent(adapter: InMemoryAdapter, agent_id: str, role: str) -> None:
    # already registered → register_agent raises; idempotent for our purposes
    with contextlib.suppress(Exception):
        adapter.register_agent(agent_id, role, "demo-hash")
