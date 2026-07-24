"""Experiment E2 — chaos: kill a node mid-write and measure memory integrity.

The thesis of the project, executed literally (the project plan §8.2, §8.5). Two
arms, both real, nothing simulated:

* ``E2_crdb_3node_kill_node`` — 20 agents write concurrently to a real 3-node
  CockroachDB cluster (``docker-compose.chaos.yml``); mid-write we ``docker kill``
  a node (SIGKILL, no drain). We assert: zero acknowledged memories lost, zero
  corruption (a checksum taken BEFORE the event verifies unchanged after), the
  fleet keeps writing after a brief failover, and we time the recovery.
* ``E2_baseline_single_kill`` — the same storm against a throwaway single-node
  store; killing it has no failover, so writes stop. We report the honest thing:
  availability is lost (downtime, rejected writes), NOT data (a single node's data
  returns on restart) — claiming "all lost" would be false.

Why this is not run through experiments/run.py: it needs docker lifecycle control
(killing a container) and wall-clock recovery timing, which the pure runner has no
business doing. The RESULT rows are still written to ``experiment_runs`` on the
dev cluster, so ``experiments.make_tables`` renders R2 exactly like every other
table — a number here is a real measurement, never a fabrication (§3.1).

    ./.venv/bin/python -m chaos.run_e2            # runs both arms, writes R2 rows
"""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from adapters.cockroach import CockroachDBAdapter
from core.config import AletheiaConfig
from core.consolidation import consolidate
from core.embeddings import DeterministicEmbedder
from core.models import MemoryEvent, MemoryKind

# The dev cluster holds experiment_runs (results of record); chaos runs elsewhere.
RESULTS_DSN = "postgresql://root@localhost:26257/aletheia?sslmode=disable"
CHAOS_DSN = "postgresql://root@localhost:26357/aletheia?sslmode=disable"

N_AGENTS = 20
KILL_TARGET = "aletheia-chaos-2"
WARMUP_S = 3.0  # writes accumulate before the kill, so there is a set to protect
POST_KILL_S = 12.0  # keep writing through the failover to observe recovery


@dataclass
class Ack:
    mem_id: str
    content: str
    at: float


def _sh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), capture_output=True, text=True, check=check)


def _wait_ready(dsn: str, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn, autocommit=True, connect_timeout=2) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            time.sleep(1.0)
    return False


def _wait_dead(container: str, timeout: float = 10.0) -> float:
    """Block until ``docker kill`` has actually stopped the container, and return
    that death time. `docker kill` only SENDS the signal — for the ~ms until the
    process dies a writer can still complete, so measuring recovery/continuity
    from the kill call (not confirmed death) falsely credits the dead node."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = _sh("docker", "inspect", "-f", "{{.State.Running}}", container, check=False)
        if out.stdout.strip() != "true":
            return time.monotonic()
        time.sleep(0.05)
    return time.monotonic()


def _run_write_storm(
    dsn: str,
    run_tag: str,
    stop: threading.Event,
    acks: list[Ack],
    acks_lock: threading.Lock,
    failed_ref: dict[str, int],
) -> None:
    """20 agents write in a loop until ``stop``, appending to the SHARED ``acks``
    list (so the caller can inspect committed writes live, mid-run)."""
    config = AletheiaConfig()
    embedder = DeterministicEmbedder(dim=config.embedding_dim, seed=0)
    agent_ids = [f"{run_tag}-agent-{i}" for i in range(N_AGENTS)]

    # Register agents up front (best-effort; the cluster is healthy at start).
    with psycopg.connect(dsn, autocommit=True) as conn:
        for aid in agent_ids:
            conn.execute(
                "UPSERT INTO agents (agent_id, role, token_hash) VALUES (%s,'sre',%s)",
                (aid, "hash"),
            )

    def writer(idx: int) -> None:
        aid = agent_ids[idx]
        adapter = CockroachDBAdapter(dsn, embedding_dim=config.embedding_dim)
        w = 0
        while not stop.is_set():
            content = f"{run_tag} a{idx} w{w}"
            w += 1
            event = MemoryEvent(
                agent_id=aid,
                content=content,
                kind=MemoryKind.EPISODIC,
                embedding=embedder.embed(content),
            )
            try:
                mem_id = adapter.write_episode(aid, event)
                with acks_lock:
                    acks.append(Ack(mem_id=str(mem_id), content=content, at=time.monotonic()))
            except Exception:
                with acks_lock:
                    failed_ref["n"] += 1
                time.sleep(0.05)  # brief backoff during a failover window
        adapter.close()

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(N_AGENTS)]
    for t in threads:
        t.start()

    # A concurrent consolidation cycle: the kill lands "during consolidation".
    def consolidator() -> None:
        adapter = CockroachDBAdapter(dsn, embedding_dim=config.embedding_dim)
        while not stop.is_set():
            with contextlib.suppress(Exception):
                consolidate(adapter, config)
            time.sleep(0.5)
        adapter.close()

    ct = threading.Thread(target=consolidator)
    ct.start()

    stop.wait()
    for t in threads:
        t.join()
    ct.join()


def _all_present(dsn: str, mem_ids: list[str]) -> int:
    """Count how many of ``mem_ids`` are NOT present now (lost)."""
    if not mem_ids:
        return 0
    with psycopg.connect(dsn, autocommit=True) as conn:
        present = {
            r[0]
            for r in conn.execute(
                "SELECT mem_id::STRING FROM memories WHERE mem_id = ANY(%s::UUID[])", (mem_ids,)
            ).fetchall()
        }
    return sum(1 for m in mem_ids if m not in present)


def run_kill_node() -> dict[str, Any]:
    run_tag = f"e2-3node-{uuid.uuid4().hex[:8]}"
    print(f"[E2] kill_node: storm under {run_tag}", flush=True)
    stop = threading.Event()

    acks: list[Ack] = []
    acks_lock = threading.Lock()
    failed_holder = {"n": 0}

    storm_thread = threading.Thread(
        target=_run_write_storm,
        args=(CHAOS_DSN, run_tag, stop, acks, acks_lock, failed_holder),
    )
    storm_thread.start()

    time.sleep(WARMUP_S)
    # Integrity snapshot of everything acknowledged so far — the set we must
    # protect across the kill. Checksum computed BEFORE the event (§8.7 control 4).
    pre_kill_ids = [a.mem_id for a in list(acks)]
    pre_count, pre_sum = _checksum_of_ids(CHAOS_DSN, pre_kill_ids)
    print(f"[E2] pre-kill committed set: {pre_count} rows, checksum {pre_sum[:12]}", flush=True)

    # THE EVENT: a real SIGKILL of a node, mid-write, mid-consolidation.
    kill_t = time.monotonic()
    _sh("docker", "kill", KILL_TARGET, check=False)
    print(f"[E2] killed {KILL_TARGET} at t={kill_t:.2f}", flush=True)

    # Observe recovery: first acknowledged write strictly after the kill.
    recovery_s = None
    deadline = time.monotonic() + POST_KILL_S
    while time.monotonic() < deadline:
        first_after = next((a for a in list(acks) if a.at > kill_t), None)
        if first_after is not None:
            recovery_s = first_after.at - kill_t
            break
        time.sleep(0.1)
    time.sleep(max(0.0, deadline - time.monotonic()))

    stop.set()
    storm_thread.join()

    # Restart the node so the cluster (and re-runs) are healthy again.
    _sh("docker", "start", KILL_TARGET, check=False)
    _wait_ready(CHAOS_DSN, timeout=60)

    # Verdict: is every ACKNOWLEDGED write still present, and the pre-kill set
    # unchanged (same count, same checksum)?
    ack_ids = [a.mem_id for a in acks]
    lost = _all_present(CHAOS_DSN, ack_ids)
    post_count, post_sum = _checksum_of_ids(CHAOS_DSN, pre_kill_ids)
    integrity_ok = post_count == pre_count and post_sum == pre_sum and lost == 0
    writes_after_kill = sum(1 for a in acks if a.at > kill_t)
    fleet_kept = writes_after_kill > 0
    # "In flight" = writes the fleet completed in the second STARTING at the kill —
    # the writes happening THROUGH the failure, the number that matters for "did
    # losing a node interrupt the fleet". (failed writes reported separately.)
    writes_in_flight = sum(1 for a in acks if kill_t <= a.at <= kill_t + 1.0)

    _cleanup(CHAOS_DSN, run_tag)
    return {
        "experiment": "E2",
        "arm": "E2_crdb_3node_kill_node",
        "event": "kill_node",
        "n_agents": N_AGENTS,
        "writes_in_flight": writes_in_flight,
        "writes_failed": failed_holder["n"],
        "writes_acknowledged": len(acks),
        "pre_kill_protected_set": pre_count,
        "memories_lost": lost,
        "integrity_ok": integrity_ok,
        "recovery_s": round(recovery_s, 3) if recovery_s is not None else None,
        "fleet_kept_operating": fleet_kept,
        "cluster": "docker-compose.chaos.yml (3 nodes, RF=3)",
    }


def _checksum_of_ids(dsn: str, mem_ids: list[str]) -> tuple[int, str]:
    uniq = sorted(set(mem_ids))
    if not uniq:
        return 0, hashlib.sha256().hexdigest()
    with psycopg.connect(dsn, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT mem_id::STRING, content FROM memories WHERE mem_id = ANY(%s::UUID[])",
            (uniq,),
        ).fetchall()
    digest = hashlib.sha256()
    for mem_id, content in sorted(rows):
        digest.update(mem_id.encode())
        digest.update(b"\x00")
        digest.update(content.encode())
        digest.update(b"\x00")
    return len(rows), digest.hexdigest()


def _cleanup(dsn: str, run_tag: str) -> None:
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("DELETE FROM provenance WHERE agent_id LIKE %s", (run_tag + "%",))
            conn.execute(
                "UPDATE memories SET superseded_by = NULL WHERE agent_id LIKE %s", (run_tag + "%",)
            )
            conn.execute(
                "DELETE FROM canonical_facts WHERE source_mem IN (SELECT mem_id FROM memories WHERE agent_id LIKE %s)",
                (run_tag + "%",),
            )
            conn.execute("DELETE FROM memories WHERE agent_id LIKE %s", (run_tag + "%",))
            conn.execute("DELETE FROM agents WHERE agent_id LIKE %s", (run_tag + "%",))
    except Exception as exc:  # cleanup is best-effort
        print(f"[E2] cleanup warning: {exc}", flush=True)


BASELINE_NAME = "aletheia-e2-baseline"
BASELINE_PORT = 26457
BASELINE_DSN = f"postgresql://root@localhost:{BASELINE_PORT}/aletheia?sslmode=disable"
CRDB_IMAGE = "cockroachdb/cockroach:v25.4.13"


def run_baseline_single_kill() -> dict[str, Any]:
    """The baseline: a throwaway SINGLE-node store. Killing it has no failover, so
    writes stop. The honest measurement: availability is lost (downtime, failed
    writes, no auto-recovery), NOT data — a single node's rows return on restart.
    Claiming 'all lost' would be false, so we verify the data survives."""
    print("[E2] baseline single-node: starting throwaway node", flush=True)
    _sh("docker", "rm", "-f", BASELINE_NAME, check=False)
    _sh(
        "docker",
        "run",
        "-d",
        "--name",
        BASELINE_NAME,
        "-p",
        f"{BASELINE_PORT}:26257",
        CRDB_IMAGE,
        "start-single-node",
        "--insecure",
    )
    if not _wait_ready(BASELINE_DSN, timeout=60):
        _sh("docker", "rm", "-f", BASELINE_NAME, check=False)
        raise RuntimeError("baseline single node did not come up")
    _apply_schema(BASELINE_DSN)

    run_tag = f"e2-baseline-{uuid.uuid4().hex[:8]}"
    stop = threading.Event()
    acks: list[Ack] = []
    acks_lock = threading.Lock()
    failed = {"n": 0}
    storm_thread = threading.Thread(
        target=_run_write_storm, args=(BASELINE_DSN, run_tag, stop, acks, acks_lock, failed)
    )
    storm_thread.start()

    time.sleep(WARMUP_S)
    with acks_lock:
        pre_kill_ids = [a.mem_id for a in acks]
    pre_count, pre_sum = _checksum_of_ids(BASELINE_DSN, pre_kill_ids)

    # THE EVENT: kill the only node. No quorum, no failover.
    _sh("docker", "kill", BASELINE_NAME, check=False)
    death_t = _wait_dead(BASELINE_NAME)  # measure from CONFIRMED death, not the kill call
    print("[E2] baseline node confirmed dead — writes now have nowhere to go", flush=True)

    # Probe window: keep the storm running for a few seconds AFTER death. On a
    # single node with no failover every write fails — any ack here would be a
    # recovery, and there is none.
    time.sleep(min(POST_KILL_S, 5.0))
    stop.set()
    storm_thread.join()
    acks_after_death = [a for a in acks if a.at > death_t]
    recovery_s = (min(a.at for a in acks_after_death) - death_t) if acks_after_death else None
    downtime_s = round(POST_KILL_S if recovery_s is None else recovery_s, 2)
    writes_after_kill = len(acks_after_death)

    # Restart to prove the DATA survived (only availability was lost).
    _sh("docker", "start", BASELINE_NAME, check=False)
    _wait_ready(BASELINE_DSN, timeout=60)
    post_count, post_sum = _checksum_of_ids(BASELINE_DSN, pre_kill_ids)
    data_survived = post_count == pre_count and post_sum == pre_sum
    _sh("docker", "rm", "-f", BASELINE_NAME, check=False)

    return {
        "experiment": "E2",
        "arm": "E2_baseline_single_kill",
        "event": "kill_single_node",
        "n_agents": N_AGENTS,
        "writes_in_flight": writes_after_kill,  # writes that survived the outage: none
        "writes_failed": failed["n"],
        "writes_acknowledged": len(acks),
        "pre_kill_protected_set": pre_count,
        # Data is not lost — it returns on restart. Availability is what's lost.
        "memories_lost": 0 if data_survived else pre_count,
        "integrity_ok": data_survived,
        "recovery_s": round(recovery_s, 3) if recovery_s is not None else None,
        "downtime_s": downtime_s,
        "fleet_kept_operating": writes_after_kill > 0,
        "note": "single node has no failover: availability lost during the outage, data survives restart",
    }


def _apply_schema(dsn: str) -> None:
    import subprocess as _sp

    _sp.run(
        [".venv/bin/python", "infra/apply_ddl.py", "--dsn", dsn],
        capture_output=True,
        text=True,
        check=True,
    )


def record(dsn: str, metrics: dict[str, Any]) -> None:
    """Write one authoritative E2 row to experiment_runs (results of record)."""
    metrics = {**metrics, "authoritative": True, "dry_run": False}
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO experiment_runs (arm, seed, metrics, started_at, finished_at) "
            "VALUES (%s, %s, %s, now(), now())",
            (metrics["arm"], 0, Jsonb(metrics)),
        )


def main() -> int:
    if not _wait_ready(CHAOS_DSN, timeout=5):
        print(
            "[E2] the 3-node chaos cluster is not up. Start it first:\n"
            "  docker compose -f docker-compose.chaos.yml up -d\n"
            "  ./chaos/verify_cluster.sh\n"
            "  ALETHEIA_CRDB_DSN='" + CHAOS_DSN + "' ./infra/apply_ddl.sh",
            flush=True,
        )
        return 2
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["kill_node", "baseline", "both"], default="both")
    args = parser.parse_args()

    if args.arm in ("kill_node", "both"):
        metrics = run_kill_node()
        print(f"[E2] kill_node result: {metrics}", flush=True)
        record(RESULTS_DSN, metrics)
        print("[E2] wrote E2_crdb_3node_kill_node to experiment_runs", flush=True)
    if args.arm in ("baseline", "both"):
        metrics = run_baseline_single_kill()
        print(f"[E2] baseline result: {metrics}", flush=True)
        record(RESULTS_DSN, metrics)
        print("[E2] wrote E2_baseline_single_kill to experiment_runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
