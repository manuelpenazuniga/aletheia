"""C4 — gossip propagation: findings spread across the fleet, degrading with distance.

One ``gossip_tick`` models a single round of an epidemic spread of findings: each
ACTIVE episodic memory (a "source") is re-shared to peer agents that have not yet
received it, and the second-hand copy is *degraded* — words drop out and its
importance decays — the further it travels from the original observation.

This degradation is the **adverse phenomenon** the consolidation cycle (C2) later
repairs; gossip motivates the problem, consolidation is the solution (the project
plan §5.2). The degradation here is a **controlled, deterministic simulation** —
pure stdlib, no LLM, no network — not a real model call and not a claim about how
the product distorts information: it is the honest way to reproduce "information
degrades as it propagates" in a reproducible experiment.

Design notes worth keeping in view:

* **One hop per tick.** The fleet is snapshotted once at tick start, so children
  written this tick cannot cascade within the same tick. That guarantees
  termination and determinism.
* **Idempotent.** An edge ``(source, recipient)`` that already exists in the
  provenance graph is never re-sent, so repeated ticks converge and stop instead
  of duplicating forever.
* **Hop cap.** A source at ``config.gossip_max_hops`` is never propagated again;
  a child's ``hop_count`` is therefore always ``<= gossip_max_hops``.
* **No fact_key.** A gossip child deliberately omits ``fact_key`` from its meta so
  C2's newest-wins grouping cannot mistake a degraded copy for canonical truth.

Portable core: no boto3, no psycopg, no httpx. The degraded embedding is computed
on the trusted side (here) and handed to ``adapter.write_episode`` — an agent
never supplies the vector. Gated by ``AletheiaConfig.enable_gossip``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from core.adapter import StorageAdapter
from core.config import AletheiaConfig
from core.embeddings import Embedder
from core.models import MemoryEvent, MemoryKind, MemoryStatus

#: Fraction of content lost per hop of distance from the original observation.
GOSSIP_LOSS_PER_HOP: float = 0.15
#: Ceiling on degradation, so a claim never fully vanishes however far it travels.
GOSSIP_MAX_LOSS: float = 0.6
#: How many fresh peers a single source reaches per tick. One keeps the spread
#: legible and the per-tick cost bounded; the experiment runner may raise it.
DEFAULT_GOSSIP_FANOUT: int = 1


@dataclass(frozen=True, slots=True)
class GossipSummary:
    """Counts from one gossip tick."""

    candidates: int  # ACTIVE episodic sources eligible to propagate (hop_count < max)
    propagations: int  # new gossip memories written this tick
    capped: int  # ACTIVE episodic sources skipped because the hop cap was reached
    peers: int  # distinct agents in the fleet this tick (memory authors)
    skipped: bool = False  # True iff disabled by config.enable_gossip


def degradation_loss(child_hop: int, max_hops: int) -> float:
    """Fraction of content lost at ``child_hop`` hops from the observation.

    Monotonically non-decreasing in ``child_hop`` and capped at
    :data:`GOSSIP_MAX_LOSS`. ``max_hops`` is accepted for interface symmetry with
    the tick (which enforces the cap separately) but does not alter the curve.
    """
    return min(GOSSIP_MAX_LOSS, max(0, child_hop) * GOSSIP_LOSS_PER_HOP)


def degrade_content(content: str, child_hop: int, seed: int) -> str:
    """Return a degraded copy of ``content`` for a memory ``child_hop`` hops out.

    Each word is dropped independently with probability equal to the hop loss, so
    the text both shortens and perturbs as it travels. Deterministic: the same
    ``(content, child_hop, seed)`` always yields the same result, in any process.
    Never returns an empty string — :class:`~core.models.MemoryEvent` requires
    non-empty content, so at least the first word is always kept.
    """
    words = content.split()
    if not words:
        return content
    loss = degradation_loss(child_hop, 0)
    rng = random.Random(f"{seed}:{child_hop}:{content}")
    kept = [w for w in words if rng.random() >= loss]
    if not kept:
        kept = [words[0]]
    return " ".join(kept)


def gossip_tick(
    adapter: StorageAdapter,
    embedder: Embedder,
    config: AletheiaConfig,
    *,
    fanout: int = DEFAULT_GOSSIP_FANOUT,
) -> GossipSummary:
    """Run one gossip round; return the counts. See the module docstring."""
    if not config.enable_gossip:
        # No adapter mutation whatsoever when disabled (the enable_gossip
        # passthrough rule; mirrors consolidation's no-op-when-disabled contract).
        return GossipSummary(candidates=0, propagations=0, capped=0, peers=0, skipped=True)

    if embedder.dim != config.embedding_dim:
        raise ValueError(
            f"embedder dimension {embedder.dim} != config.embedding_dim "
            f"{config.embedding_dim}; they must agree (single source of truth)"
        )

    # Snapshot the fleet ONCE, so children written this tick do not cascade within
    # the same tick (one hop per tick => termination + determinism).
    all_mems = adapter.list_memories(status=None)
    candidates = adapter.list_memories(status=MemoryStatus.ACTIVE, kind=MemoryKind.EPISODIC)

    # The "fleet" this tick = distinct authors of ACTIVE memory. Deriving peers
    # from authors needs no new adapter method and guarantees every recipient is
    # already registered (write_episode's foreign-key check will pass).
    peers_all = sorted({m.agent_id for m in all_mems if m.status == MemoryStatus.ACTIVE})

    # Already-delivered edges, from the snapshot: what makes the tick idempotent.
    # Superseded/quarantined copies count too — do not re-gossip something already
    # delivered and then rejected.
    existing_pairs = {(m.parent_mem, m.agent_id) for m in all_mems if m.parent_mem is not None}

    propagations = 0
    capped = 0
    for source in candidates:
        # Hop cap: child_hop = source.hop_count + 1 would exceed the cap, so this
        # source can travel no further. gossip_max_hops == 0 => nothing propagates.
        if source.hop_count >= config.gossip_max_hops:
            capped += 1
            continue

        peers = [a for a in peers_all if a != source.agent_id]
        if not peers:
            continue  # single-agent fleet, or the source is the only author

        # Seeded per-source order, stable across ticks so the spread fills peers in
        # a consistent sequence as successive ticks widen the reach.
        rng = random.Random(f"{config.seed}:{source.mem_id}")
        order = sorted(peers)
        rng.shuffle(order)

        already = {b for (p, b) in existing_pairs if p == source.mem_id}
        targets = [b for b in order if b not in already][:fanout]

        for recipient in targets:
            child_hop = source.hop_count + 1  # 1 <= child_hop <= gossip_max_hops
            loss = degradation_loss(child_hop, config.gossip_max_hops)
            degraded = degrade_content(source.content, child_hop, config.seed)
            importance = round(max(0.0, source.importance * (1.0 - loss)), 6)
            vec = embedder.embed(degraded)  # re-embed the DEGRADED text: drift is honest
            event = MemoryEvent(
                agent_id=recipient,
                content=degraded,
                kind=MemoryKind.EPISODIC,
                embedding=vec,
                importance=importance,
                parent_mem=source.mem_id,
                hop_count=child_hop,
                signature=None,  # a gossip copy is second-hand: no first-hand HMAC
                meta={
                    "gossip": True,
                    "simulated_degradation": True,
                    "gossip_source_mem": source.mem_id,
                    "gossip_source_agent": source.agent_id,
                    "gossip_hop": child_hop,
                    "gossip_loss": round(loss, 4),
                },
            )
            adapter.write_episode(recipient, event)
            # Update the dedup set as we go, so a larger fanout can't pick the same
            # peer twice within this source.
            existing_pairs.add((source.mem_id, recipient))
            propagations += 1

    return GossipSummary(
        candidates=len(candidates) - capped,
        propagations=propagations,
        capped=capped,
        peers=len(peers_all),
    )
