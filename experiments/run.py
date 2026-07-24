"""The experiment runner CLI (the project plan §8.3).

One process runs one ``(experiment, arm, seed)`` and records **one row** in the
``experiment_runs`` table — nothing is shared between processes except the
database, so the human's orchestrator can fan a whole grid out in parallel
(``python -m experiments.run --exp E3 --arm A0_full --seed 2``).

Two seams keep this honest and offline-testable:

* **Dependencies are injected.** A run needs a storage backend, an embedder, a
  model and the scenario data; :class:`Deps` bundles factories for all of them.
  :func:`build_dry_run_deps` wires the fully-offline stack (in-memory adapter +
  the deterministic embedder + a scripted model); :func:`build_live_deps` wires
  the real one (CockroachDB + Bedrock) and **fails loudly** when the
  infrastructure it needs is absent. ``--dry-run`` and every test take the first
  path, so no test ever touches the network.
* **The recorder is a seam too.** :class:`InMemoryRecorder` captures the row in a
  list (what the tests assert against); :class:`CockroachRecorder` persists it.

Integrity (the project plan §3.1, §8.6): a dry run **never emits an authoritative
number**. Its :class:`Deps` carry ``authoritative=False``, that flag is stamped
into every metrics dict, and the row is captured in memory rather than written to
the results table. The Bedrock-dependent result cells (E3 knowledge-update, E5
cost, E4 fleet-contamination, the E4 semantic-anomaly slice) are the literal
string ``"pendiente"`` until a real run against real infrastructure produces
them — this runner fabricates nothing.

This module lives under ``experiments/`` and may import ``adapters`` (and, lazily,
``psycopg``/``boto3`` on the live path only). ``core/`` stays clean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from adapters.memory_inmem import InMemoryAdapter
from agents.model_client import ModelClient, ScriptedModel
from core.adapter import StorageAdapter
from core.config import AletheiaConfig
from core.embeddings import DeterministicEmbedder, Embedder
from core.immune import ImmuneSystem
from core.models import MemoryEvent, MemoryKind, utcnow
from experiments.arms import EXPERIMENTS, Arm, get_arm
from experiments.baselines import NaiveNonTransactionalStore, build_naive_backend
from experiments.scoring import (
    StoreState,
    WriteObservation,
    count_inconsistencies,
    detection_rate,
    false_positive_rate,
    p95,
)
from scenarios.loader import (
    PoisonSuite,
    load_incidents,
    load_poison_suite,
    poison_to_event,
)

__all__ = [
    "Deps",
    "InMemoryRecorder",
    "Recorder",
    "RunRecord",
    "RunnerError",
    "build_dry_run_deps",
    "build_live_deps",
    "main",
    "run_once",
]

#: How many episodes each simulated agent writes in an E1 run. Small on purpose:
#: E1 measures *contention* (many writers on shared cells), not throughput.
E1_WRITES_PER_AGENT = 5

#: The two E4 attack categories a purely-offline run can score against the
#: committed labels: provenance is structural and injection is a regex — both
#: deterministic. ``semantic_anomaly`` needs real embeddings and a populated
#: agent history, so that slice stays ``pendiente`` until a live run (§8.4).
_E4_OFFLINE_CATEGORIES = frozenset({"bad_provenance", "injection_pattern"})

#: The honest placeholder for a cell that only a real run can fill (§3.1).
PENDIENTE = "pendiente"


class RunnerError(RuntimeError):
    """A run could not proceed — bad arguments, or missing infrastructure.

    Raised (rather than silently degrading) so a live run without a cluster fails
    loudly instead of emitting a number nobody can trust.
    """


# -----------------------------------------------------------------------------
# The run record — one row of experiment_runs (the project plan §6)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One ``experiment_runs`` row: ``(run_id, arm, seed, metrics, started, finished)``.

    ``metrics`` is the JSONB payload. It always carries a ``dry_run`` and an
    ``authoritative`` flag so a reader can never mistake a smoke run for a run of
    record.
    """

    run_id: str
    arm: str
    seed: int
    metrics: dict[str, Any]
    started_at: Any
    finished_at: Any

    def as_json(self) -> dict[str, Any]:
        """A JSON-ready snapshot mirroring the table's columns (timestamps -> ISO)."""
        return {
            "run_id": self.run_id,
            "arm": self.arm,
            "seed": self.seed,
            "metrics": self.metrics,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


@runtime_checkable
class Recorder(Protocol):
    """Persists a :class:`RunRecord`. Two impls: in-memory (tests) and CockroachDB."""

    def record(self, run: RunRecord) -> None: ...


class InMemoryRecorder:
    """Captures rows in a list — the offline seam the tests assert against.

    Writes nothing anywhere, so a dry run leaves no trace and cannot pollute the
    results table with non-authoritative numbers.
    """

    def __init__(self) -> None:
        self.records: list[RunRecord] = []

    def record(self, run: RunRecord) -> None:
        self.records.append(run)


class CockroachRecorder:
    """Persists a row to ``experiment_runs`` (the live path).

    ``psycopg`` is imported lazily inside :meth:`record` so importing this module
    pulls no database driver; the offline default stays network-free.
    """

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise RunnerError("CockroachRecorder requires a DSN")
        self._dsn = dsn

    def record(self, run: RunRecord) -> None:  # pragma: no cover - requires a cluster
        import psycopg
        from psycopg.types.json import Jsonb

        with psycopg.connect(self._dsn, autocommit=True) as conn:
            conn.execute(
                "INSERT INTO experiment_runs "
                "(run_id, arm, seed, metrics, started_at, finished_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    run.run_id,
                    run.arm,
                    run.seed,
                    Jsonb(run.metrics),
                    run.started_at,
                    run.finished_at,
                ),
            )


# -----------------------------------------------------------------------------
# Injected dependencies
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Deps:
    """Everything an experiment runner needs, as injected factories.

    ``authoritative`` is the single source of truth for "may this run produce a
    published number?". A dry run sets it ``False``; the flag is copied into every
    metrics dict and gates the Bedrock-dependent cells to ``pendiente``.
    ``parallel`` selects real concurrency for E1 (the live path) versus a
    deterministic single-threaded sweep (dry runs / tests).
    """

    embedder: Embedder
    model: ModelClient
    make_adapter: Callable[[AletheiaConfig], StorageAdapter]
    make_naive_store: Callable[[AletheiaConfig], NaiveNonTransactionalStore]
    load_incidents: Callable[[], Sequence[Any]]
    load_poison_suite: Callable[[], PoisonSuite]
    authoritative: bool = False
    parallel: bool = False
    base_config: AletheiaConfig = field(default_factory=AletheiaConfig)


def build_dry_run_deps(base_config: AletheiaConfig | None = None) -> Deps:
    """Wire the fully-offline stack: in-memory adapter, deterministic embedder,
    scripted model, the committed scenario loaders. ``authoritative=False``."""
    config = base_config or AletheiaConfig()
    embedder = DeterministicEmbedder(dim=config.embedding_dim, seed=config.seed)
    return Deps(
        embedder=embedder,
        model=ScriptedModel(default="OFFLINE"),
        make_adapter=lambda cfg: InMemoryAdapter(embedding_dim=cfg.embedding_dim),
        make_naive_store=lambda cfg: build_naive_backend(embedding_dim=cfg.embedding_dim),
        load_incidents=load_incidents,
        load_poison_suite=load_poison_suite,
        authoritative=False,
        parallel=False,
        base_config=config,
    )


def build_live_deps(dsn: str | None, base_config: AletheiaConfig | None = None) -> Deps:
    """Wire the real stack (CockroachDB + Bedrock). Fails loudly without infra.

    A missing DSN, or a missing cloud SDK, raises :class:`RunnerError` rather than
    quietly falling back to an offline fake — a run of record must run against the
    real cluster or not at all.
    """
    if not dsn:
        raise RunnerError(
            "a live run needs a DSN — pass --dsn or set ALETHEIA_DSN/DATABASE_URL, "
            "or use --dry-run for an offline smoke run"
        )
    config = base_config or AletheiaConfig()
    try:  # pragma: no cover - exercised only where boto3/psycopg are installed
        from adapters.bedrock_embedder import BedrockEmbedder
        from adapters.cockroach import CockroachDBAdapter
        from agents.model_client import BedrockModelClient
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RunnerError(
            f"a live run needs the cloud SDKs (boto3/psycopg); import failed: {exc}"
        ) from exc

    embedder = BedrockEmbedder(config)  # pragma: no cover - needs AWS
    model = BedrockModelClient(config)  # pragma: no cover - needs AWS
    return Deps(  # pragma: no cover - needs a cluster
        embedder=embedder,
        model=model,
        make_adapter=lambda cfg: CockroachDBAdapter(dsn, embedding_dim=cfg.embedding_dim),
        make_naive_store=lambda cfg: build_naive_backend(
            embedding_dim=cfg.embedding_dim, race_delay=0.001
        ),
        load_incidents=load_incidents,
        load_poison_suite=load_poison_suite,
        authoritative=True,
        parallel=True,
        base_config=config,
    )


# -----------------------------------------------------------------------------
# Per-experiment runners — each returns the metrics JSONB payload for one row.
# -----------------------------------------------------------------------------


def _close(store: Any) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        close()


def _run_e1(
    arm: Arm, seed: int, config: AletheiaConfig, deps: Deps, dry_run: bool
) -> dict[str, Any]:
    """E1 concurrency: N agents write shared memory; count inconsistencies + p95.

    Runs against the naive non-transactional store (``naive_baseline``) or the
    serializable-like backend (``crdb_serializable`` -> the in-memory oracle offline,
    CockroachDB live). The metric is real for whatever backend is wired; a dry run
    is single-threaded and therefore deterministic (no simulated race), which is
    exactly why it is a *smoke* run and never authoritative.
    """
    n = arm.fleet_size or 0
    if n <= 0:
        raise RunnerError(f"E1 arm {arm.name!r} has no fleet_size")

    is_naive = arm.backend == "naive_baseline"
    store: Any = deps.make_naive_store(config) if is_naive else deps.make_adapter(config)
    try:
        agent_ids = [f"e1-agent-{i}" for i in range(n)]
        for aid in agent_ids:
            store.register_agent(aid, "sre", f"hash-{aid}")

        # Build every write payload up front (deterministic, seeded) so the only
        # thing that races is the storage op itself.
        tasks: list[tuple[str, str]] = []
        for i, aid in enumerate(agent_ids):
            for w in range(E1_WRITES_PER_AGENT):
                tasks.append((aid, f"E1 seed={seed} agent={i} write={w}"))

        def _do_write(task: tuple[str, str]) -> tuple[str, str, float]:
            aid, content = task
            event = MemoryEvent(
                agent_id=aid,
                content=content,
                kind=MemoryKind.EPISODIC,
                embedding=deps.embedder.embed(content),
            )
            start = perf_counter()
            mem_id = store.write_episode(aid, event)
            return mem_id, content, (perf_counter() - start) * 1000.0

        if deps.parallel and n > 1:
            with ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(_do_write, tasks))
        else:
            results = [_do_write(task) for task in tasks]

        observed = [
            WriteObservation(key=mem_id, value=content, has_vector=True)
            for mem_id, content, _ in results
        ]
        write_ms = [ms for _, _, ms in results]

        events = store.list_memories(status=None)
        state = StoreState(
            rows={e.mem_id: e.content for e in events},
            vector_keys=[e.mem_id for e in events if e.embedding],
        )
        report = count_inconsistencies(observed, state)

        read_ms: list[float] = []
        for aid in agent_ids:
            query = deps.embedder.embed(f"E1 query for {aid}")
            start = perf_counter()
            store.query_semantic(aid, query, k=5, budget_tokens=config.retrieval_budget_tokens)
            read_ms.append((perf_counter() - start) * 1000.0)

        return {
            "experiment": "E1",
            "arm": arm.name,
            "seed": seed,
            "backend": arm.backend,
            "n_agents": n,
            "writes": len(observed),
            "inconsistencies_per_1000": report.per_1000,
            "lost_updates": report.lost_updates,
            "divergence": report.divergence,
            "dirty_reads": report.dirty_reads,
            "p95_write_ms": p95(write_ms),
            "p95_read_ms": p95(read_ms),
            "parallel": deps.parallel,
            "dry_run": dry_run,
            "authoritative": deps.authoritative,
        }
    finally:
        _close(store)


def _run_e4(
    arm: Arm, seed: int, config: AletheiaConfig, deps: Deps, dry_run: bool
) -> dict[str, Any]:
    """E4 poisoning: run the poison suite through the immune gate; score detection.

    The provenance and injection detectors are deterministic, so detection rate
    and false-positive rate over those categories are scored offline against the
    committed labels. With ``A4_no_immune`` the gate is disabled and detection
    honestly falls to ``0.0``. The semantic-anomaly slice and fleet-contamination
    need real embeddings / a real fleet and stay ``pendiente``.
    """
    suite = deps.load_poison_suite()
    adapter = deps.make_adapter(config)
    try:
        immune = ImmuneSystem(adapter, deps.embedder, config)

        labels: dict[str, str] = {}
        legit_ids: list[str] = []
        verdicts: dict[str, bool] = {}
        for case in suite.all():
            # Only the offline-deterministic attack categories are scored here;
            # every legit case is scored (any quarantine of one is a false positive).
            if case.is_attack and case.category not in _E4_OFFLINE_CATEGORIES:
                continue
            verdict = immune.inspect(poison_to_event(case, deps.embedder))
            verdicts[case.id] = not verdict.accepted  # True == quarantined/detected
            if case.is_attack:
                labels[case.id] = "attack"
            else:
                legit_ids.append(case.id)

        det = detection_rate(labels, verdicts) if labels else PENDIENTE
        fpr = false_positive_rate(legit_ids, verdicts) if legit_ids else PENDIENTE

        return {
            "experiment": "E4",
            "arm": arm.name,
            "seed": seed,
            "immune_enabled": config.enable_immune,
            "detection_rate": det,
            "false_positive_rate": fpr,
            "attacks_scored": len(labels),
            "legit_scored": len(legit_ids),
            "scored_categories": sorted(_E4_OFFLINE_CATEGORIES),
            # Both need real embeddings / a real fleet — never fabricated here.
            "semantic_anomaly_detection": PENDIENTE,
            "fleet_contamination": PENDIENTE,
            "dry_run": dry_run,
            "authoritative": deps.authoritative,
        }
    finally:
        _close(adapter)


def _pending_runner(
    experiment: str, result_cells: tuple[str, ...], reason: str
) -> Callable[[Arm, int, AletheiaConfig, Deps, bool], dict[str, Any]]:
    """Build a runner whose result cells are ``pendiente`` until real infra runs it.

    E2 (chaos) needs the docker cluster and ``ccloud``; E3/E5/E6 need real Bedrock
    inference. Rather than fabricate a number, the runner records an honest
    ``pendiente`` row carrying the arm/seed/config context (the project plan §3.1).
    """

    def _run(
        arm: Arm, seed: int, config: AletheiaConfig, deps: Deps, dry_run: bool
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "experiment": experiment,
            "arm": arm.name,
            "seed": seed,
            "status": reason,
            "dry_run": dry_run,
            "authoritative": False,
        }
        for cell in result_cells:
            metrics[cell] = PENDIENTE
        return metrics

    return _run


_PENDING_BEDROCK = "pending_real_inference"
_PENDING_CHAOS = "pending_chaos_harness"

#: Experiment id -> runner. E1/E4 are offline-capable now; the rest record an
#: honest ``pendiente`` row until their infrastructure (docker chaos, Bedrock) runs.
_DISPATCH: dict[str, Callable[[Arm, int, AletheiaConfig, Deps, bool], dict[str, Any]]] = {
    "E1": _run_e1,
    "E2": _pending_runner("E2", ("memories_lost", "recovery_s", "integrity_ok"), _PENDING_CHAOS),
    "E3": _pending_runner("E3", ("ku_accuracy", "stale_fact_rate"), _PENDING_BEDROCK),
    "E4": _run_e4,
    "E5": _pending_runner(
        "E5", ("tokens_per_query", "footprint_mb", "task_success"), _PENDING_BEDROCK
    ),
    "E6": _pending_runner("E6", ("task_success",), _PENDING_BEDROCK),
}


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def run_once(
    exp: str,
    arm_name: str,
    seed: int,
    *,
    deps: Deps,
    recorder: Recorder,
    dry_run: bool,
) -> RunRecord:
    """Run one ``(exp, arm, seed)``, record it, and return the row.

    Validates that ``arm`` is registered for ``exp`` (the arm table is the source
    of truth), materialises the arm's :class:`AletheiaConfig` on top of
    ``deps.base_config``, dispatches to the experiment runner, then hands the row
    to ``recorder``. Unknown experiment/arm, or an arm/experiment mismatch, raises
    :class:`RunnerError`.
    """
    if exp not in EXPERIMENTS:
        raise RunnerError(f"unknown experiment {exp!r}; known: {list(EXPERIMENTS)}")
    try:
        arm = get_arm(arm_name)
    except ValueError as exc:
        raise RunnerError(str(exc)) from None
    if exp not in arm.experiments:
        raise RunnerError(
            f"arm {arm_name!r} is not registered for {exp}; it belongs to {list(arm.experiments)}"
        )

    config = arm.config(deps.base_config)
    runner = _DISPATCH[exp]

    started = utcnow()
    metrics = runner(arm, seed, config, deps, dry_run)
    finished = utcnow()

    record = RunRecord(
        run_id=str(uuid.uuid4()),
        arm=arm.name,
        seed=seed,
        metrics=metrics,
        started_at=started,
        finished_at=finished,
    )
    recorder.record(record)
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.run",
        description="Run one (experiment, arm, seed) and record it in experiment_runs.",
    )
    parser.add_argument("--exp", required=True, choices=EXPERIMENTS, help="experiment id")
    parser.add_argument("--arm", required=True, help="arm name (see experiments/arms.py)")
    parser.add_argument("--seed", type=int, default=0, help="run seed (default 0)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="offline smoke run: in-memory backend + scripted model, nothing persisted, "
        "no authoritative numbers",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="CockroachDB DSN for a live run (falls back to ALETHEIA_DSN / DATABASE_URL)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 ok, 2 on a runner error)."""
    args = _build_parser().parse_args(argv)
    try:
        if args.dry_run:
            deps = build_dry_run_deps()
            recorder: Recorder = InMemoryRecorder()
        else:
            dsn = args.dsn or os.environ.get("ALETHEIA_DSN") or os.environ.get("DATABASE_URL")
            deps = build_live_deps(dsn)
            recorder = CockroachRecorder(dsn or "")

        record = run_once(
            args.exp,
            args.arm,
            args.seed,
            deps=deps,
            recorder=recorder,
            dry_run=args.dry_run,
        )
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(record.as_json(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
