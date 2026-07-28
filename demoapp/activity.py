"""A real fleet-activity run for the live sparklines on the Fleet view.

Performs REAL memory operations — episodic writes and semantic retrievals — against
a fresh in-memory world, timing each, and returns a per-tick timeline of throughput,
p95 retrieval latency, and footprint. Nothing is synthesised: the latency is
measured with ``perf_counter`` around real calls, and it visibly climbs as the
memory grows (retrieval scans more) — which is exactly what metabolic forgetting is
there to bound.
"""

from __future__ import annotations

import random
from time import perf_counter
from typing import Any

from adapters.memory_inmem import InMemoryAdapter
from core.embeddings import DeterministicEmbedder
from core.models import MemoryEvent, MemoryKind

ACT_DIM = 64
_CORPUS = (
    "disk pressure on node 7 climbing",
    "p99 read latency spike on shard 4",
    "certificate on the api gateway expired",
    "pgbouncer connection pool saturated",
    "checkout error rate above baseline",
    "deploy rolled back to the green build",
    "replica lag exceeded the alert threshold",
    "oom killer terminated the worker",
    "tls handshake failures on the edge",
    "runbook: drain the zone before failover",
)


def run_fleet_activity(*, ticks: int = 40, ops_per_tick: int = 24, seed: int = 0) -> dict[str, Any]:
    """Run a real write+retrieve workload and return a per-tick activity timeline."""
    rng = random.Random(seed)
    embedder = DeterministicEmbedder(dim=ACT_DIM, seed=0)
    adapter = InMemoryAdapter(embedding_dim=ACT_DIM)
    agents = [f"sre-{i:02d}" for i in range(1, 6)]
    for a in agents:
        adapter.register_agent(a, "sre", f"hash-{a}")

    timeline: list[dict[str, Any]] = []
    total_writes = 0
    seq = 0
    for t in range(ticks):
        latencies: list[float] = []
        wall0 = perf_counter()
        for _ in range(ops_per_tick):
            agent = rng.choice(agents)
            if rng.random() < 0.6:  # a write
                seq += 1
                content = f"{rng.choice(_CORPUS)} #{seq}"
                s = perf_counter()
                adapter.write_episode(
                    agent,
                    MemoryEvent(
                        agent_id=agent,
                        content=content,
                        kind=MemoryKind.EPISODIC,
                        embedding=embedder.embed(content),
                    ),
                )
                latencies.append((perf_counter() - s) * 1000.0)
                total_writes += 1
            else:  # a semantic retrieval
                q = rng.choice(_CORPUS)
                s = perf_counter()
                adapter.query_semantic(agent, embedder.embed(q), 5, 4000)
                latencies.append((perf_counter() - s) * 1000.0)
        wall = perf_counter() - wall0
        latencies.sort()
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
        stats = adapter.stats(None)
        timeline.append(
            {
                "tick": t,
                "ops_per_sec": round(ops_per_tick / wall) if wall > 0 else 0,
                "p95_ms": round(p95, 3),
                "footprint_kb": round(stats.footprint_mb * 1024.0, 1),
                "active": stats.active,
            }
        )
    return {
        "ticks": ticks,
        "ops_per_tick": ops_per_tick,
        "total_ops": ticks * ops_per_tick,
        "total_writes": total_writes,
        "timeline": timeline,
    }
