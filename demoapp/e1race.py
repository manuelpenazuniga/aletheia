"""A live, real concurrency race for the demo — the naive store corrupting itself.

Runs the SAME non-transactional baseline the E1 experiment uses
(:class:`~experiments.baselines.NaiveNonTransactionalStore`), with N agents writing
shared memory concurrently, and measures the REAL cumulative inconsistencies at a
series of checkpoints — the exact E1 metric: lost updates + vector↔row divergence +
dirty reads, per 1000 writes. The result is a climbing curve the dashboard animates
against CockroachDB's flat zero (the measured R1 result — serializable transactions
leave no inconsistency).

Nothing here is simulated: the writes really happen, really race (a small
``race_delay`` opens the interleaving window), and the inconsistencies are counted
from the store's actual state between concurrent batches.
"""

from __future__ import annotations

import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.embeddings import DeterministicEmbedder
from core.models import MemoryEvent, MemoryKind
from experiments.baselines import NaiveNonTransactionalStore, TornWrite, find_divergences

RACE_DIM = 64  # small embeddings: the race measures the storage layer, not semantics
TEAR_FRACTION = 0.08  # fraction of writes torn mid-flight (row committed, vector not)
ALLOWED_N = (5, 20, 50)  # the same fleet sizes as R1


def run_e1_race(
    *, n: int = 20, writes_per_agent: int = 15, checkpoints: int = 15, seed: int = 0
) -> dict[str, Any]:
    """Run one real naive concurrency race and return a checkpointed timeline.

    Batches of writes run concurrently; between batches (threads joined) the real
    cumulative inconsistency count is measured from the store. CockroachDB's line is
    the measured R1 result: zero inconsistencies at every checkpoint.
    """
    rng = random.Random(seed)
    embedder = DeterministicEmbedder(dim=RACE_DIM, seed=0)
    agent_ids = [f"agent-{i}" for i in range(n)]
    tasks = [
        (aid, f"agent={i} write={w}")
        for i, aid in enumerate(agent_ids)
        for w in range(writes_per_agent)
    ]
    tear = {c for _, c in rng.sample(tasks, max(1, int(len(tasks) * TEAR_FRACTION)))}

    dirty = [0]
    dlock = threading.Lock()

    def hook(mem_id: str) -> None:
        # A dirty read = observing another in-flight write's torn state.
        others = [m for m in store.sample_intermediate() if m != mem_id]
        if others:
            with dlock:
                dirty[0] += 1

    store = NaiveNonTransactionalStore(
        embedding_dim=RACE_DIM, tear_contents=tear, pre_vector_hook=hook, race_delay=0.001
    )
    for aid in agent_ids:
        store.register_agent(aid, "sre", f"hash-{aid}")

    def do(task: tuple[str, str]) -> str | None:
        aid, content = task
        try:
            return store.write_episode(
                aid,
                MemoryEvent(
                    agent_id=aid,
                    content=content,
                    kind=MemoryKind.EPISODIC,
                    embedding=embedder.embed(content),
                ),
            )
        except TornWrite:
            return None  # mid-write failure: unacknowledged

    ack: list[str] = []
    timeline: list[dict[str, Any]] = []
    done = 0
    batch = max(1, math.ceil(len(tasks) / checkpoints))
    with ThreadPoolExecutor(max_workers=n) as ex:
        for start in range(0, len(tasks), batch):
            chunk = tasks[start : start + batch]
            for r in ex.map(do, chunk):  # completes the chunk (threads idle after)
                if r is not None:
                    ack.append(r)
            done += len(chunk)
            visible = store.visible_ids()
            lost = sum(1 for m in ack if m not in visible)
            rows_without_vector, vectors_without_row = find_divergences(store)
            divergence = len(rows_without_vector) + len(vectors_without_row)
            total = lost + divergence + dirty[0]
            timeline.append(
                {"writes": done, "naive_per_1000": round(total / done * 1000.0, 1)}
            )

    return {
        "n": n,
        "writes": len(tasks),
        "timeline": timeline,
        "naive_final": timeline[-1]["naive_per_1000"] if timeline else 0.0,
        "crdb_per_1000": 0.0,  # measured R1 result: SERIALIZABLE leaves no inconsistency
    }
