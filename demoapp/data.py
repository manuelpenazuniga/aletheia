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

from adapters.memory_inmem import InMemoryAdapter
from core.config import AletheiaConfig
from core.consolidation import consolidate
from core.embeddings import DeterministicEmbedder, Embedder
from core.forgetting import run_forgetting
from core.immune import ImmuneSystem
from core.models import MemoryEvent, MemoryKind, MemoryStatus
from ingest.immune import persist_quarantine
from scenarios.loader import incident_to_events, load_incidents, load_poison_cases

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

_POISON_META_KEY = "poison_case"  # tags a seeded attack so contamination is countable

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
        _ensure_agent(adapter, case.agent_id, "adversary")
        event = MemoryEvent(
            agent_id=case.agent_id,
            content=case.content,
            kind=MemoryKind.EPISODIC,
            signature=case.provenance.signature or "unsigned",
            meta={_POISON_META_KEY: case.id},
        )
        verdict = immune.inspect(event)
        if verdict.accepted:
            adapter.write_episode(
                event.agent_id,
                replace(event, embedding=embedder.embed(event.content)),
            )
        else:
            persist_quarantine(
                adapter,
                embedder,
                event,
                reason=(verdict.reason.value if verdict.reason else ""),
                detector=verdict.detector or "",
                payload=dict(verdict.payload),
            )

    # 5. Metabolic forgetting (flag-gated inside run_forgetting): prune the fewest
    #    lowest-value active memories to fit the budget. With enable_forgetting=False
    #    this is a true no-op and the footprint stays large — the no-forgetting ablation.
    run_forgetting(adapter, config)

    return DemoWorld(adapter=adapter, embedder=embedder, config=config)


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
