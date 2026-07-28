"""The demo world — a populated memory layer the dashboard reads from.

Built entirely offline (an :class:`~adapters.memory_inmem.InMemoryAdapter` + the
seeded :class:`~core.embeddings.DeterministicEmbedder`), from the SAME committed
scenarios and the SAME core cycles the real system runs. It is a real memory
layer with real content — a consolidated runbook whose obsolete version was
``supersede``d and promoted to a canonical fact, a poisoned write the immune
system ``quarantine``d, and a spread of fleet memories — just without Bedrock or a
cluster. In production the app is handed a CockroachDBAdapter + a Bedrock embedder
instead (see :func:`demoapp.app.create_app`); nothing else changes.

The world is **flag-aware**: it is populated by passing an
:class:`~core.config.AletheiaConfig` through the real, flag-gated core cycles
(``consolidate`` skips when ``enable_consolidation`` is off; ``ImmuneSystem.inspect``
accepts everything when ``enable_immune`` is off; ``run_forgetting`` is a no-op when
``enable_forgetting`` is off). Flipping a flag therefore produces a genuinely
different world — which is what the kill-switch and the ablation wall (the project
plan §9.1.3, §9.1.6) show: each component's necessity, recomputed for real, not
mocked.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from adapters.memory_inmem import InMemoryAdapter
from core.adapter import MemoryNotFound
from core.config import AletheiaConfig
from core.consolidation import consolidate
from core.embeddings import DeterministicEmbedder, Embedder
from core.forgetting import run_forgetting
from core.gossip import gossip_tick
from core.immune import ImmuneSystem
from core.models import MemoryEvent, MemoryKind, MemoryStatus
from ingest.immune import persist_quarantine
from scenarios.loader import (
    incident_to_events,
    load_incidents,
    load_poison_cases,
    poison_to_event,
)

DEMO_DIM = 128  # enough lexical structure for the search view; offline + fast
# The active footprint (~291 tokens) sits above this budget so that, WITH
# forgetting on, the cycle visibly prunes the lowest-value memory (transient fleet
# chatter, below) to fit, while the higher-importance runbook fixes survive — and
# the no-forgetting ablation prunes nothing. A documented demo tuning, not a result.
DEMO_MEMORY_BUDGET = 250
FLEET_OBS_IMPORTANCE = 0.2  # transient chatter: genuinely lower value than a runbook fix
# Incident runbook versions are stamped with a deterministic created_at that strictly
# increases with the version number, so consolidation (which picks the newest by
# (created_at, mem_id)) ALWAYS crowns the true latest version. Without this, the
# versions are constructed microseconds apart and can tie on created_at, letting the
# random mem_id crown a stale version — a nondeterministic, wrong canonical fact.
_INCIDENT_ANCHOR = datetime(2026, 1, 1, tzinfo=UTC)

_POISON_META_KEY = "poison_id"  # poison_to_event tags every attack with this; countable

# The four panels of the ablation wall (§9.1.6): full, and each single component
# removed. Each names the target metric that panel is meant to move and a plain
# sentence of the degradation, so the wall reads without a legend.
ABLATION_PANELS: tuple[tuple[str, dict[str, bool], str, str], ...] = (
    (
        "full",
        {},
        "none",
        "All components on: no stale knowledge, no poison in the fleet, footprint under budget.",
    ),
    (
        "no-consolidation",
        {"enable_consolidation": False},
        "canonical_facts",
        "No knowledge-update: no canonical source-of-truth facts are promoted (0) and no obsolete "
        "runbook version is superseded. (Forgetting still runs, so some obsolete versions may be "
        "archived for cost rather than left retrievable — the lost source of truth is the point.)",
    ),
    (
        "no-immune",
        {"enable_immune": False},
        "poison_active",
        "No immune gate: the seeded poison writes are accepted and sit in the fleet's active memory "
        "instead of being quarantined.",
    ),
    (
        "no-forgetting",
        {"enable_forgetting": False},
        "active_cost_tokens",
        "No metabolic forgetting: nothing is pruned, so the full active footprint is retained "
        "(on this scenario, the measured delta shown above vs the full run).",
    ),
)


@dataclass(frozen=True, slots=True)
class DemoWorld:
    adapter: InMemoryAdapter
    embedder: Embedder
    config: AletheiaConfig


def demo_config(base: AletheiaConfig | None = None) -> AletheiaConfig:
    """A config safe to build a demo world from: the four feature flags are taken
    from ``base`` (default all-on), but ``embedding_dim`` and ``memory_budget_tokens``
    are pinned to the demo values — the dim MUST match the embedder, and the budget
    is the fixed tuning that makes forgetting observable."""
    base = base or AletheiaConfig()
    return replace(base, embedding_dim=DEMO_DIM, memory_budget_tokens=DEMO_MEMORY_BUDGET)


def build_demo_world(
    *,
    incident_count: int = 8,
    poison_count: int = 4,
    config: AletheiaConfig | None = None,
) -> DemoWorld:
    """Populate an in-memory world through the real, flag-gated core cycles.

    With the default (all-on) config: fleet memories, a knowledge-update
    (consolidation supersedes obsolete runbook versions and promotes canonical
    facts), quarantined poison, and a forgetting pass that prunes to budget. Flip
    a flag in ``config`` and the corresponding cycle changes the world for real.
    """
    config = demo_config(config)
    embedder = DeterministicEmbedder(dim=DEMO_DIM, seed=0)
    adapter = InMemoryAdapter(embedding_dim=DEMO_DIM)

    incidents = load_incidents()[:incident_count]

    # 1. Write each incident's runbook v1 and its revisions as episodic memories.
    #    meta['fact_key'] ties the versions together so consolidation can supersede
    #    the obsolete ones. created_at is stamped deterministically, strictly
    #    increasing with runbook_version (see _INCIDENT_ANCHOR), so the latest
    #    version always wins consolidation — reproducibly, never by a random tie.
    seq = 0
    for inc in incidents:
        for event in incident_to_events(inc, embedder):
            _ensure_agent(adapter, event.agent_id, "sre")
            version = int(event.meta.get("runbook_version") or 0)
            created = _INCIDENT_ANCHOR + timedelta(seconds=version * 1000, microseconds=seq)
            adapter.write_episode(event.agent_id, replace(event, created_at=created))
            seq += 1

    # 2. Consolidation cycle (flag-gated inside consolidate): obsolete runbook
    #    versions are superseded (a knowledge-update), the current one promoted to
    #    a canonical fact with a version history — the "git of beliefs". With
    #    enable_consolidation=False this is a no-op and the obsolete versions stay
    #    ACTIVE, which is exactly the no-consolidation ablation.
    consolidate(adapter, config)

    # 3. A few benign fleet observations from distinct SRE agents, so the fleet
    #    view has more than one active author. Low, STRICTLY DECREASING importance:
    #    transient chatter is worth less than a runbook fix (so forgetting prunes it
    #    first), and distinct values give forgetting a total order with no ties — so
    #    which memories get pruned at the budget boundary is fully deterministic
    #    (equal values would tie-break on the random mem_id and vary run to run).
    observations = [
        ("sre-01", "p99 read latency on shard 4 climbed to 480ms after the 14:10 deploy"),
        ("sre-02", "checkout error rate back to baseline after rolling back to the green build"),
        ("sre-03", "disk pressure on node 7 cleared once log rotation was re-enabled"),
        ("sre-01", "pgbouncer pool saturated at 200 connections; raised the ceiling to 400"),
        ("sre-02", "cert on the api gateway expired; rotated via ACM and redeployed the listener"),
    ]
    for i, (aid, content) in enumerate(observations):
        _ensure_agent(adapter, aid, "sre")
        adapter.write_episode(
            aid,
            MemoryEvent(
                agent_id=aid,
                content=content,
                kind=MemoryKind.EPISODIC,
                importance=round(FLEET_OBS_IMPORTANCE - i * 0.02, 3),
                embedding=embedder.embed(content),
            ),
        )

    # 4. The immune system in action: run committed poison cases through the real,
    #    flag-gated gate. A rejection is persisted QUARANTINED (non-retrievable) and
    #    logged — exactly what the ingest write path does. With enable_immune=False,
    #    inspect() accepts everything, so the poison is written as an ACTIVE memory
    #    and contaminates the fleet — exactly the no-immune ablation.
    immune = ImmuneSystem(adapter, embedder, config)
    for case in load_poison_cases(category="injection_pattern")[:poison_count]:
        _ingest_poison_case(adapter, embedder, immune, case)

    # 5. Metabolic forgetting (flag-gated inside run_forgetting): prune the fewest
    #    lowest-value active memories to fit the budget. With enable_forgetting=False
    #    this is a true no-op and the footprint stays large — the no-forgetting ablation.
    run_forgetting(adapter, config)

    return DemoWorld(adapter=adapter, embedder=embedder, config=config)


def _ingest_poison_case(adapter, embedder, immune, case) -> tuple[Any, str | None]:
    """Run one poison case through the real immune gate and persist the outcome.

    The single write path shared by the seed loop and the live "launch attack"
    button: build the attack event WITH its full attack data intact (bad provenance,
    over-propagated hops, missing signature — via ``poison_to_event``), inspect →
    if rejected, persist QUARANTINED (non-retrievable, logged), else write ACTIVE
    (only when the immune flag is off, or the attack slipped past). Returns the
    verdict and the quarantined memory id (or None). One function means a launched
    attack is handled BYTE-IDENTICALLY to a seeded one — the feed is real.
    """
    _ensure_agent(adapter, case.agent_id, "adversary")
    event = poison_to_event(case, embedder)
    verdict = immune.inspect(event)
    if not verdict.accepted:
        quar_id = persist_quarantine(
            adapter,
            embedder,
            event,
            reason=(verdict.reason.value if verdict.reason else ""),
            detector=verdict.detector or "",
            payload=dict(verdict.payload),
        )
        return verdict, quar_id
    # Accepted (immune off, or the attack was not caught): write it ACTIVE so it
    # contaminates the fleet — the point of the no-immune ablation. Strip a claimed
    # parent that does not exist, else a bad-provenance payload's dangling FK would
    # crash the write instead of demonstrating contamination.
    if event.parent_mem is not None and not _memory_exists(adapter, event.parent_mem):
        event = replace(event, parent_mem=None)
    adapter.write_episode(event.agent_id, event)
    return verdict, None


def _memory_exists(adapter: InMemoryAdapter, mem_id: str) -> bool:
    try:
        adapter.get_memory(mem_id)
    except MemoryNotFound:
        return False
    return True


def launch_attack(
    adapter: InMemoryAdapter,
    embedder: Embedder,
    config: AletheiaConfig,
    *,
    category: str = "injection_pattern",
    index: int = 0,
) -> dict[str, Any]:
    """Release the adversary: run one committed poison case through the live immune
    gate against ``adapter`` and report what happened (§9.1.5).

    Real, not staged: it calls the same :class:`ImmuneSystem` and quarantine path
    the fleet uses, so a caught attack lands in the quarantine feed exactly as a
    production interception would. ``index`` cycles through the category's cases so
    repeated clicks show different attacks.
    """
    cases = load_poison_cases(category=category)
    if not cases:
        raise ValueError(f"no poison cases for category {category!r}")
    case = cases[index % len(cases)]
    immune = ImmuneSystem(adapter, embedder, config)
    verdict, quar_id = _ingest_poison_case(adapter, embedder, immune, case)
    return {
        "attack": {
            "case_id": case.id,
            "category": category,
            "agent_id": case.agent_id,
            "content": case.content,
        },
        "detected": not verdict.accepted,
        "reason": verdict.reason.value if verdict.reason else None,
        "detector": verdict.detector,
        "quarantined_mem_id": quar_id,
    }


_CONTAGION_FLEET = ("sre-01", "sre-02", "sre-03", "ops-01")


def build_contagion(*, enable_immune: bool, ticks: int = 6) -> dict[str, Any]:
    """Trace how a single poisoned fact spreads through the fleet — or doesn't.

    Real mechanics, not a mock: a benign fleet of SRE agents each holds one active
    memory (so they are gossip peers), then the adversary introduces ONE poisoned
    fact through the REAL immune gate. With ``enable_immune=False`` the poison
    survives, becomes a gossip candidate, and the real ``gossip_tick`` cycle
    propagates it hop by hop (degrading the text as it travels); we reconstruct the
    exact contagion tree from provenance (``parent_mem`` / ``gossip_source_agent``).
    With ``enable_immune=True`` the same write is quarantined at the gate, is never
    an active gossip candidate, and the fleet stays clean.

    Returns the fleet graph the dashboard renders: nodes (agents, contaminated or
    not), the propagation edges of the poison lineage, and the mutating content at
    each hop. Deterministic — agent-level structure does not depend on random ids.
    """
    config = demo_config(AletheiaConfig(enable_immune=enable_immune, enable_gossip=True))
    embedder = DeterministicEmbedder(dim=DEMO_DIM, seed=0)
    adapter = InMemoryAdapter(embedding_dim=DEMO_DIM)

    # 1. A benign fleet: each agent holds one active episodic memory, which is what
    #    makes it a gossip peer (peers = distinct authors of ACTIVE memory).
    for aid in _CONTAGION_FLEET:
        _ensure_agent(adapter, aid, "sre")
        content = f"{aid}: nominal — no anomalies on my shard this cycle"
        adapter.write_episode(
            aid,
            MemoryEvent(
                agent_id=aid,
                content=content,
                kind=MemoryKind.EPISODIC,
                importance=0.5,
                embedding=embedder.embed(content),
            ),
        )

    # 2. The adversary introduces ONE poisoned fact through the real immune gate.
    case = load_poison_cases(category="injection_pattern")[0]
    _ensure_agent(adapter, case.agent_id, "adversary")
    event = poison_to_event(case, embedder)
    verdict = ImmuneSystem(adapter, embedder, config).inspect(event)
    if verdict.accepted:  # immune off (or slipped) → written active, will propagate
        poison_id = adapter.write_episode(
            event.agent_id, replace(event, embedding=embedder.embed(event.content))
        )
    else:  # immune on → quarantined at the gate, never a gossip candidate
        poison_id = persist_quarantine(
            adapter,
            embedder,
            event,
            reason=(verdict.reason.value if verdict.reason else ""),
            detector=verdict.detector or "",
            payload=dict(verdict.payload),
        )

    # 3. Run the REAL gossip cycle. Each tick moves the poison (and its children,
    #    within the hop cap) one hop further across the fleet.
    for _ in range(ticks):
        gossip_tick(adapter, embedder, config)

    # 4. Reconstruct the poison lineage from provenance: every memory whose ancestry
    #    roots at the poison id.
    mems = adapter.list_memories(status=None)
    by_id = {m.mem_id: m for m in mems if m.mem_id}

    def roots_at_poison(m: MemoryEvent) -> bool:
        seen: set[str] = set()
        cur: MemoryEvent | None = m
        while cur is not None and cur.mem_id and cur.mem_id not in seen:
            seen.add(cur.mem_id)
            if cur.mem_id == poison_id:
                return True
            cur = by_id.get(cur.parent_mem) if cur.parent_mem else None
        return False

    origin = case.agent_id
    lineage = [m for m in mems if roots_at_poison(m)]
    # Victims = fleet agents (not the origin) now holding an active poison-derived
    # memory. The adversary is the source, never counted as a victim.
    contaminated = {
        m.agent_id
        for m in lineage
        if m.status == MemoryStatus.ACTIVE and m.parent_mem is not None and m.agent_id != origin
    }

    # First-infection tree: ONE edge per victim — how it was first contaminated
    # (minimum hop, deterministic tie-break by source). The full gossip mesh has
    # redundant re-shares whose exact shape depends on random ids; the first-
    # infection tree is the clean, reproducible contagion path.
    first: dict[str, tuple[int, str, str]] = {}  # victim -> (hop, source, content)
    degraded: dict[int, str] = {}  # hop -> a representative content, for the mutation view
    for m in lineage:
        src = m.meta.get("gossip_source_agent") if isinstance(m.meta, dict) else None
        if not (src and m.status == MemoryStatus.ACTIVE and m.agent_id != origin):
            continue
        cand = (m.hop_count, src, m.content)
        if m.agent_id not in first or cand < first[m.agent_id]:
            first[m.agent_id] = cand
        degraded.setdefault(m.hop_count, m.content)
    edges = [
        {"from": src, "to": victim, "hop": hop, "content": content}
        for victim, (hop, src, content) in sorted(first.items())
    ]
    edges.sort(key=lambda e: (e["hop"], e["from"], e["to"]))

    # The poison text mutating as it travels — "information degrades as it
    # propagates" (C4), made visible. Hop 0 is the original write.
    degradation = [{"hop": 0, "content": case.content}] + [
        {"hop": h, "content": degraded[h]} for h in sorted(degraded)
    ]

    agents = [*_CONTAGION_FLEET, origin]
    nodes = [
        {
            "agent_id": aid,
            "role": "adversary" if aid == case.agent_id else "sre",
            "origin": aid == case.agent_id,
            "contaminated": aid in contaminated,
        }
        for aid in agents
    ]
    return {
        "enable_immune": enable_immune,
        "detected": not verdict.accepted,
        "quarantined": not verdict.accepted,
        "poison_content": case.content,
        "detector": verdict.detector,
        "nodes": nodes,
        "edges": edges,
        "degradation": degradation,
        "contaminated_count": sum(1 for n in nodes if n["contaminated"]),
        "fleet_size": len(_CONTAGION_FLEET),
    }


def poison_categories() -> list[str]:
    """Attack families the launch-attack button offers, in a stable order.

    Only families the gate reliably catches OFFLINE are exposed. ``semantic_anomaly``
    is deliberately excluded: that detector measures a write's distance from the
    author's OWN history, and the offline adversary has no benign history to deviate
    from — so it fires only ~2/8 offline. Exposing it would misrepresent the immune
    system as weak; it is exercised properly in the E4 experiment, not this button."""
    return ["injection_pattern", "bad_provenance"]


def world_metrics(adapter: InMemoryAdapter) -> dict[str, int]:
    """Derive the ablation-target metrics from an adapter's current state.

    Pure over the adapter; the same function scores every panel so the numbers are
    comparable. ``stale_active`` and ``poison_active`` are the degradations the
    kill-switch makes visible — obsolete knowledge left retrievable, poison left
    in the fleet — while ``active_cost_tokens`` is what forgetting bounds.
    """
    mems = adapter.list_memories(status=None)
    active = [m for m in mems if m.status == MemoryStatus.ACTIVE]

    # Highest runbook version seen per fact_key (across all statuses).
    max_version: dict[str, int] = {}
    for m in mems:
        key = m.meta.get("fact_key") if isinstance(m.meta, dict) else None
        if isinstance(key, str) and key:
            max_version[key] = max(max_version.get(key, 0), int(m.meta.get("runbook_version") or 0))

    stale_active = 0
    poison_active = 0
    for m in active:
        meta = m.meta if isinstance(m.meta, dict) else {}
        key = meta.get("fact_key")
        if (
            isinstance(key, str)
            and key
            and int(meta.get("runbook_version") or 0) < max_version[key]
        ):
            stale_active += 1
        if meta.get(_POISON_META_KEY):
            poison_active += 1

    stats = adapter.stats(None)
    return {
        "active": len(active),
        "superseded": stats.superseded,
        "quarantined": stats.quarantined,
        "archived": stats.archived,
        "canonical_facts": stats.canonical_facts,
        "stale_active": stale_active,
        "poison_active": poison_active,
        "active_cost_tokens": sum(m.cost_tokens or 0 for m in active),
    }


def build_ablation_panels() -> list[dict]:
    """The four ablation-wall panels, each a real world scored by world_metrics."""
    panels = []
    for label, overrides, target, effect in ABLATION_PANELS:
        cfg = demo_config(AletheiaConfig(**overrides))
        world = build_demo_world(config=cfg)
        panels.append(
            {
                "label": label,
                "flags": flags_of(cfg),
                "target_metric": target,
                "effect": effect,
                "metrics": world_metrics(world.adapter),
            }
        )
    return panels


def flags_of(config: AletheiaConfig) -> dict[str, bool]:
    """The four feature flags of a config, as a plain dict for the API."""
    return {
        "enable_consolidation": config.enable_consolidation,
        "enable_forgetting": config.enable_forgetting,
        "enable_gossip": config.enable_gossip,
        "enable_immune": config.enable_immune,
    }


def _ensure_agent(adapter: InMemoryAdapter, agent_id: str, role: str) -> None:
    # already registered → register_agent raises; idempotent for our purposes
    with contextlib.suppress(Exception):
        adapter.register_agent(agent_id, role, "demo-hash")
