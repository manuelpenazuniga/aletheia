"""Render the results tables (R1/R2/R3/R3b) from ``experiment_runs`` (the project plan §8.5).

This is the machine half of ``docs/results.md``. That document is the
**pre-registration**, hand-written and carrying the ▲/▼ predictions declared
before any run. This module generates a *separate* file — never
``docs/results.md`` — whose every cell is filled **only** from a real row in the
``experiment_runs`` table. Until a cell has a run it says ``pendiente``; it never
inherits the pre-registered number or the prediction markers (the project plan §3.1,
§8.6). The generated file therefore carries **no** ``▲``/``▼`` and no hardcoded
values: the CRDB "inconsistencies" cells read ``pendiente`` until a real E1 run,
never the pre-registered ``0``.

Purity boundary
---------------
Everything here is a pure function of a list of :class:`Run` values except
:func:`fetch_runs`, the single impure edge that runs one ``SELECT`` against the
table (``experiments/`` may import ``psycopg``; ``core/`` may not). The tests
never call :func:`fetch_runs`: they build :class:`Run` lists in memory and drive
:func:`aggregate` / the renderers / :func:`build_document` directly, so the whole
suite stays offline.

Row identity
------------
A row is selected by ``(experiment, arm)``. The experiment id disambiguates the
arms shared across experiments — ``A0_full`` runs in E3, E4 **and** E5 and fills
different R3 columns in each — so the runner writes a reserved ``experiment`` key
into the metrics JSONB (see ``experiments/run.py``); ``fetch_runs`` reads it back
(accepting the alias ``exp``) and ``n_agents`` for the E1 grid. A run whose
``(experiment, arm)`` matches no table cell is not rendered but is listed in an
audit footnote, so a stray or mistyped row is visible rather than silently dropped
— and never invented into a table.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.scoring import mean_std

__all__ = [
    "Agg",
    "Run",
    "aggregate",
    "build_document",
    "fetch_runs",
    "fmt_bool",
    "fmt_cell",
    "main",
    "render_r1",
    "render_r2",
    "render_r3",
    "render_r3b",
]

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The honest placeholder for a cell no real run has filled (the project plan §3.1).
PENDIENTE = "pendiente"
#: Rendered in a cell that is structurally not-applicable (the "—" of §8.5), kept
#: distinct from ``pendiente`` (run-but-pending vs. never-meaningful).
EM_DASH = "—"


# -----------------------------------------------------------------------------
# The run record read back from experiment_runs
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run:
    """One ``experiment_runs`` row, as the table generator consumes it.

    ``exp`` and ``n_agents`` are lifted out of the ``metrics`` JSONB (reserved
    keys written by the runner); ``metrics`` still carries them so a renderer can
    read any metric by name.
    """

    exp: str | None
    arm: str
    seed: int
    n_agents: int | None
    metrics: Mapping[str, Any]
    started_at: Any = None
    finished_at: Any = None


# -----------------------------------------------------------------------------
# Cell specifications — hardcoded to mirror the project plan §8.5 / docs/results.md
# -----------------------------------------------------------------------------
#
# A spec is a small tuple ``(kind, ...)``:
#   ("na",)                       -> structurally not applicable, renders "—"
#   ("num", exp, arm, metric_key) -> numeric cell: mean ± sigma over matching runs
#   ("bool", exp, arm, metric_key)     -> boolean consensus: yes / no / mixed
#   ("bool_inv", exp, arm, metric_key) -> boolean consensus of the negation
# A cell with no matching run renders ``pendiente`` — never a fabricated value.

NA: tuple[str, ...] = ("na",)


def _num(exp: str, arm: str, key: str) -> tuple[str, str, str, str]:
    return ("num", exp, arm, key)


def _pct(exp: str, arm: str, key: str) -> tuple[str, str, str, str]:
    """A rate stored as a fraction in [0,1] but rendered in a %-labelled column,
    so 0.8 shows as 80, not 0.8 (external review: unit mismatch)."""
    return ("pct", exp, arm, key)


def _bool(exp: str, arm: str, key: str) -> tuple[str, str, str, str]:
    return ("bool", exp, arm, key)


def _bool_inv(exp: str, arm: str, key: str) -> tuple[str, str, str, str]:
    return ("bool_inv", exp, arm, key)


# R1 — Concurrency (E1). Rows are the (backend, N) grid; each maps to one E1 arm.
_R1_HEADER = ("Backend", "N agents", "Inconsist./1000 wr", "p95 write ms", "p95 read ms")
_R1_ROWS: tuple[tuple[str, str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "Naive (non-transactional)",
        "5",
        (
            _num("E1", "E1_naive_n5", "inconsistencies_per_1000"),
            _num("E1", "E1_naive_n5", "p95_write_ms"),
            _num("E1", "E1_naive_n5", "p95_read_ms"),
        ),
    ),
    (
        "Naive",
        "20",
        (
            _num("E1", "E1_naive_n20", "inconsistencies_per_1000"),
            _num("E1", "E1_naive_n20", "p95_write_ms"),
            _num("E1", "E1_naive_n20", "p95_read_ms"),
        ),
    ),
    (
        "Naive",
        "50",
        (
            _num("E1", "E1_naive_n50", "inconsistencies_per_1000"),
            _num("E1", "E1_naive_n50", "p95_write_ms"),
            _num("E1", "E1_naive_n50", "p95_read_ms"),
        ),
    ),
    (
        "CockroachDB SERIALIZABLE",
        "5",
        (
            _num("E1", "E1_crdb_serializable_n5", "inconsistencies_per_1000"),
            _num("E1", "E1_crdb_serializable_n5", "p95_write_ms"),
            _num("E1", "E1_crdb_serializable_n5", "p95_read_ms"),
        ),
    ),
    (
        "CockroachDB SERIALIZABLE",
        "20",
        (
            _num("E1", "E1_crdb_serializable_n20", "inconsistencies_per_1000"),
            _num("E1", "E1_crdb_serializable_n20", "p95_write_ms"),
            _num("E1", "E1_crdb_serializable_n20", "p95_read_ms"),
        ),
    ),
    (
        "CockroachDB SERIALIZABLE",
        "50",
        (
            _num("E1", "E1_crdb_serializable_n50", "inconsistencies_per_1000"),
            _num("E1", "E1_crdb_serializable_n50", "p95_write_ms"),
            _num("E1", "E1_crdb_serializable_n50", "p95_read_ms"),
        ),
    ),
)

# R2 — Chaos (E2). Only kill_node has a registered arm; the kill-region and
# single-node-baseline rows have no arm and stay pendiente until such a run
# exists. Predictions from docs/results.md are deliberately NOT reproduced here.
_R2_HEADER = (
    "Event",
    "Writes in flight",
    "Memories lost",
    "Corruption",
    "Recovery s",
    "Fleet kept operating",
)
_R2_ROWS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "kill node",
        (
            _num("E2", "E2_crdb_3node_kill_node", "writes_in_flight"),
            _num("E2", "E2_crdb_3node_kill_node", "memories_lost"),
            _bool_inv("E2", "E2_crdb_3node_kill_node", "integrity_ok"),
            _num("E2", "E2_crdb_3node_kill_node", "recovery_s"),
            _bool("E2", "E2_crdb_3node_kill_node", "fleet_kept_operating"),
        ),
    ),
    (
        "kill region",
        (
            _num("E2", "E2_crdb_3node_kill_region", "writes_in_flight"),
            _num("E2", "E2_crdb_3node_kill_region", "memories_lost"),
            _bool_inv("E2", "E2_crdb_3node_kill_region", "integrity_ok"),
            _num("E2", "E2_crdb_3node_kill_region", "recovery_s"),
            _bool("E2", "E2_crdb_3node_kill_region", "fleet_kept_operating"),
        ),
    ),
    (
        "baseline single-node, kill",
        (
            _num("E2", "E2_baseline_single_kill", "writes_in_flight"),
            _num("E2", "E2_baseline_single_kill", "memories_lost"),
            _bool_inv("E2", "E2_baseline_single_kill", "integrity_ok"),
            _num("E2", "E2_baseline_single_kill", "recovery_s"),
            _bool("E2", "E2_baseline_single_kill", "fleet_kept_operating"),
        ),
    ),
)

# R3 — knowledge-update (E3), poisoning (E4), cost (E5). A0_full is shared across
# the three; the exp key on each spec is what routes a run to the right column.
_R3_HEADER = (
    "Config",
    "KU accuracy",
    "% stale fact",
    "Poison det. %",
    "False pos. %",
    "Tokens/query",
    "Task success",
)
_R3_ROWS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "A0 full",
        (
            _num("E3", "A0_full", "ku_accuracy"),
            _pct("E3", "A0_full", "stale_fact_rate"),
            _pct("E4", "A0_full", "detection_rate"),
            _pct("E4", "A0_full", "false_positive_rate"),
            _num("E5", "A0_full", "tokens_per_query"),
            _num("E5", "A0_full", "task_success"),
        ),
    ),
    (
        "A1 -consolidation",
        (
            _num("E3", "A1_no_consolidation", "ku_accuracy"),
            _pct("E3", "A1_no_consolidation", "stale_fact_rate"),
            NA,
            NA,
            NA,
            _num("E3", "A1_no_consolidation", "task_success"),
        ),
    ),
    (
        "A4 -immune",
        (
            NA,
            NA,
            _pct("E4", "A4_no_immune", "detection_rate"),
            NA,
            NA,
            _num("E4", "A4_no_immune", "task_success"),
        ),
    ),
    (
        "A2 -forgetting",
        (
            NA,
            NA,
            NA,
            NA,
            _num("E5", "A2_no_forgetting", "tokens_per_query"),
            _num("E5", "A2_no_forgetting", "task_success"),
        ),
    ),
    (
        "BL plain RAG",
        (
            _num("E3", "BL_rag", "ku_accuracy"),
            _pct("E3", "BL_rag", "stale_fact_rate"),
            NA,
            NA,
            _num("E3", "BL_rag", "tokens_per_query"),
            _num("E3", "BL_rag", "task_success"),
        ),
    ),
    (
        "BL single-agent (E6)",
        (
            NA,
            NA,
            NA,
            NA,
            NA,
            _num("E6", "E6_bl_single", "task_success"),
        ),
    ),
)

# R3b — external validation on LongMemEval (E3b). Filled by experiments/scoring_lme.
_R3B_HEADER = (
    "Config",
    "LME knowledge-update acc.",
    "LME temporal-reasoning acc.",
    "Replicates E3 direction?",
)
_R3B_ROWS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "A0 full",
        (
            _num("E3b", "A0_full", "lme_ku_acc"),
            _num("E3b", "A0_full", "lme_temporal_acc"),
            _bool("E3b", "A0_full", "replicates_e3_direction"),
        ),
    ),
    (
        "A1 -consolidation",
        (
            _num("E3b", "A1_no_consolidation", "lme_ku_acc"),
            _num("E3b", "A1_no_consolidation", "lme_temporal_acc"),
            _bool("E3b", "A1_no_consolidation", "replicates_e3_direction"),
        ),
    ),
    (
        "BL plain RAG",
        (
            _num("E3b", "BL_rag", "lme_ku_acc"),
            _num("E3b", "BL_rag", "lme_temporal_acc"),
            _bool("E3b", "BL_rag", "replicates_e3_direction"),
        ),
    ),
)


def _iter_specs() -> Iterator[tuple[str, ...]]:
    """Every cell spec across all four tables (for building the known-cell set)."""
    for _, _, specs in _R1_ROWS:
        yield from specs
    for _, specs in _R2_ROWS:
        yield from specs
    for _, specs in _R3_ROWS:
        yield from specs
    for _, specs in _R3B_ROWS:
        yield from specs


#: Every ``(experiment, arm)`` any table cell can draw from. A run outside this
#: set is excluded from the tables and surfaced in the audit footnote.
KNOWN_CELLS: frozenset[tuple[str, str]] = frozenset(
    (spec[1], spec[2]) for spec in _iter_specs() if spec[0] != "na"
)


# -----------------------------------------------------------------------------
# Aggregation (pure)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Agg:
    """Runs grouped for rendering.

    ``by_cell`` maps a ``(experiment, arm)`` key to the list of metrics dicts that
    matched it (one per rep/seed). ``unknown`` holds runs whose key is in no table,
    preserved for the audit footnote.
    """

    by_cell: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]
    unknown: Sequence[Run] = field(default_factory=tuple)


def _is_authoritative(metrics: Mapping[str, Any]) -> bool:
    """A row may enter the results tables ONLY if it is an authoritative, non-dry
    run of record. A smoke/dry row, or one lacking the flag, is excluded — no
    fake or provisional number can be rendered as a result (external review:
    make_tables must not trust an arbitrary row). Defense-in-depth: the runner
    already refuses to persist non-authoritative rows via the live recorder."""
    return metrics.get("authoritative") is True and metrics.get("dry_run") is not True


def aggregate(runs: Iterable[Run]) -> Agg:
    """Group runs by ``(experiment, arm)``, splitting off any unknown cell.

    Pure: no I/O, no fabrication. Two guards protect the tables:
    - Only authoritative, non-dry runs are aggregated (see :func:`_is_authoritative`).
    - Runs are de-duplicated by ``(exp, arm, seed)``, keeping the latest by
      ``finished_at``: a re-run of one seed must weight it ONCE, not inflate the
      ``n`` inside ``mean ± sigma``.
    A run whose ``(exp, arm)`` matches no known cell is treated as unknown.
    """
    # De-dup by (exp, arm, seed): later finished_at wins. None sorts first so a
    # timestamped rerun supersedes an untimed earlier row.
    latest: dict[tuple[str, str, int], Run] = {}
    for run in runs:
        if not _is_authoritative(run.metrics):
            continue
        key = (run.exp or "", run.arm, run.seed)
        prev = latest.get(key)
        if prev is None or _finished_key(run) >= _finished_key(prev):
            latest[key] = run

    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    unknown: list[Run] = []
    for run in latest.values():
        key = (run.exp or "", run.arm)
        if key in KNOWN_CELLS:
            by_cell[key].append(run.metrics)
        else:
            unknown.append(run)
    return Agg(by_cell=dict(by_cell), unknown=tuple(unknown))


def _finished_key(run: Run) -> tuple[int, str]:
    """Sortable finished_at: (has_timestamp, iso-string). Untimed rows sort first."""
    ts = run.finished_at
    return (1, ts.isoformat()) if ts is not None else (0, "")


# -----------------------------------------------------------------------------
# Cell formatting (pure)
# -----------------------------------------------------------------------------


def fmt_cell(values: Sequence[float]) -> str:
    """Format a numeric cell.

    Empty -> ``pendiente`` (never a silent zero). A single value ->
    ``"v (n=1)"`` (sigma is undefined for one sample and must not print as 0). Two
    or more -> ``"mean ± sigma"`` via :func:`experiments.scoring.mean_std` (the one
    definition used everywhere, so the tables agree with the runner). ``.3g``
    keeps a genuinely small nonzero as itself (e.g. ``4e-07``) rather than
    collapsing a real inconsistency count to ``0``.
    """
    if not values:
        return PENDIENTE
    if len(values) == 1:
        return f"{values[0]:.3g} (n=1)"
    mean, sigma = mean_std(values)
    return f"{mean:.3g} ± {sigma:.3g}"


def fmt_bool(values: Sequence[bool]) -> str:
    """Format a boolean-consensus cell: all True -> ``yes``, all False -> ``no``,
    disagreement -> ``mixed``, no data -> ``pendiente``."""
    if not values:
        return PENDIENTE
    if all(values):
        return "yes"
    if not any(values):
        return "no"
    return "mixed"


def _numeric_values(dicts: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    """Real numbers stored under ``key`` across reps.

    Booleans (a subclass of ``int``) and the string ``pendiente`` are excluded, so
    a not-yet-run rep contributes nothing rather than a fake ``0``.
    """
    out: list[float] = []
    for d in dicts:
        v = d.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _bool_values(dicts: Sequence[Mapping[str, Any]], key: str) -> list[bool]:
    return [d[key] for d in dicts if isinstance(d.get(key), bool)]


def _resolve_cell(agg: Agg, spec: tuple[str, ...]) -> str:
    kind = spec[0]
    if kind == "na":
        return EM_DASH
    _, exp, arm, key = spec
    dicts = agg.by_cell.get((exp, arm), ())
    if kind == "num":
        return fmt_cell(_numeric_values(dicts, key))
    if kind == "pct":
        return fmt_cell([v * 100.0 for v in _numeric_values(dicts, key)])
    if kind == "bool":
        return fmt_bool(_bool_values(dicts, key))
    if kind == "bool_inv":
        return fmt_bool([not v for v in _bool_values(dicts, key)])
    raise ValueError(f"unknown cell kind {kind!r}")  # pragma: no cover - guarded by specs


# -----------------------------------------------------------------------------
# Table + document rendering (pure)
# -----------------------------------------------------------------------------


def _table(header: Sequence[str], body_rows: Sequence[Sequence[str]]) -> str:
    """A GitHub-flavoured markdown table; first (label) column left, rest right."""
    aligns = ["---"] + ["---:"] * (len(header) - 1)
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    for row in body_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_r1(agg: Agg) -> str:
    rows = [[backend, n, *[_resolve_cell(agg, s) for s in specs]] for backend, n, specs in _R1_ROWS]
    return "## R1 — Concurrency (E1)\n\nCells: mean ± sigma over repetitions.\n\n" + _table(
        _R1_HEADER, rows
    )


def render_r2(agg: Agg) -> str:
    rows = [[event, *[_resolve_cell(agg, s) for s in specs]] for event, specs in _R2_ROWS]
    return (
        "## R2 — Chaos (E2)\n\nIntegrity checksums computed before the event, verified after.\n\n"
        + _table(_R2_HEADER, rows)
    )


def render_r3(agg: Agg) -> str:
    rows = [[config, *[_resolve_cell(agg, s) for s in specs]] for config, specs in _R3_ROWS]
    return (
        "## R3 — Knowledge-update (E3), poisoning (E4), cost (E5)\n\nCells: mean ± sigma over seeds.\n\n"
        + _table(_R3_HEADER, rows)
    )


def render_r3b(agg: Agg) -> str:
    rows = [[config, *[_resolve_cell(agg, s) for s in specs]] for config, specs in _R3B_ROWS]
    return (
        "## R3b — External validation on LongMemEval (E3b)\n\nCells: mean ± sigma over seeds, scored by the benchmark's own criteria.\n\n"
        + _table(_R3B_HEADER, rows)
    )


def _banner(source_note: str) -> str:
    return (
        "<!-- GENERATED FILE — do not edit by hand.\n"
        "Regenerate with: python -m experiments.make_tables\n"
        f"Source: {source_note}\n"
        "Reproducible by re-running against the same experiment_runs table state\n"
        "(no generation timestamp is embedded, so identical DB state -> identical file).\n"
        "This file carries NO predictions and no hardcoded numbers: every cell is\n"
        "'pendiente' until a real run fills it (the project plan §3.1, §8.6). -->"
    )


def _audit_footnote(agg: Agg) -> str:
    if not agg.unknown:
        return "> Audit: every run in experiment_runs mapped to a known results cell."
    lines = [
        f"> Audit: {len(agg.unknown)} run(s) had an (experiment, arm) matching no results "
        "cell and were excluded from the tables (listed, never dropped silently):",
    ]
    for run in agg.unknown:
        lines.append(f"> - experiment={run.exp!r} arm={run.arm!r} seed={run.seed}")
    return "\n".join(lines)


def build_document(runs: Iterable[Run], *, source_note: str = "experiment_runs") -> str:
    """Compose the whole generated results document. Pure — no file I/O."""
    agg = aggregate(runs)
    parts = [
        _banner(source_note),
        "# Results (generated)",
        (
            "Generated rendering of the `experiment_runs` table. The hand-written "
            "pre-registration with its up/down predictions lives in `docs/results.md`; "
            "this file reproduces none of them — a cell is `pendiente` until a real "
            "run fills it."
        ),
        render_r1(agg),
        render_r2(agg),
        render_r3(agg),
        render_r3b(agg),
        _audit_footnote(agg),
    ]
    return "\n\n".join(parts) + "\n"


# -----------------------------------------------------------------------------
# The single impure edge: read experiment_runs (never imported by tests)
# -----------------------------------------------------------------------------


def fetch_runs(dsn: str | None) -> list[Run]:  # pragma: no cover - requires a cluster
    """Read every ``experiment_runs`` row into :class:`Run` values.

    The only function in this module that touches the network. ``psycopg`` is
    imported lazily so importing the module (as the tests do) pulls in no driver.
    ``exp`` and ``n_agents`` are lifted from the metrics JSONB reserved keys
    written by the runner (``experiment`` preferred, ``exp`` accepted as an alias).
    """
    if not dsn:
        raise ValueError("fetch_runs needs a DSN — pass --dsn or set ALETHEIA_DSN / DATABASE_URL")
    import psycopg

    runs: list[Run] = []
    with psycopg.connect(dsn) as conn:
        cur = conn.execute(
            "SELECT arm, seed, metrics, started_at, finished_at FROM experiment_runs"
        )
        for arm, seed, metrics, started_at, finished_at in cur:
            m: Mapping[str, Any] = metrics or {}
            exp = m.get("experiment") or m.get("exp")
            n_agents = m.get("n_agents")
            runs.append(
                Run(
                    exp=exp,
                    arm=arm,
                    seed=seed,
                    n_agents=n_agents,
                    metrics=m,
                    started_at=started_at,
                    finished_at=finished_at,
                )
            )
    return runs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

_DEFAULT_OUT = REPO_ROOT / "docs" / "results_generated.md"
_RESULTS_MD = REPO_ROOT / "docs" / "results.md"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.make_tables",
        description="Render R1/R2/R3/R3b from experiment_runs into a generated file.",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="CockroachDB DSN (falls back to ALETHEIA_DSN / DATABASE_URL)",
    )
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="output path (default docs/results_generated.md); may not be docs/results.md",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, fetch: Any = None) -> int:
    """CLI entry point. Returns 0 on success, 2 on a guarded refusal / error.

    ``fetch`` is an injection seam used only by tests (a fake returning a
    :class:`Run` list); in normal use it defaults to :func:`fetch_runs`, the impure
    reader. The guard against overwriting ``docs/results.md`` runs **before** any
    fetch, so the refusal never needs a database.
    """
    args = _build_parser().parse_args(argv)
    out = Path(args.out).resolve()
    if out == _RESULTS_MD.resolve():
        print(
            "error: refusing to write docs/results.md — that file is the hand-written "
            "pre-registration; generated tables go to a separate file",
            file=sys.stderr,
        )
        return 2

    fetch_fn = fetch if fetch is not None else fetch_runs
    dsn = args.dsn or os.environ.get("ALETHEIA_DSN") or os.environ.get("DATABASE_URL")
    try:
        runs = list(fetch_fn(dsn))
    except Exception as exc:  # surfaced as a clean CLI error, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_document(runs, source_note="experiment_runs"), encoding="utf-8")
    print(f"wrote {out} ({len(runs)} run(s))")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
