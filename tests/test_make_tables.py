"""Offline tests for experiments/make_tables.py (the results-table generator).

These never call ``fetch_runs`` (the one impure edge): they build :class:`Run`
lists in memory and drive the pure aggregate/render/build_document path. The
guard tests assert that ``main`` refuses to touch ``docs/results.md`` and that a
run to a temp path leaves it byte-for-byte unchanged. No network, no DB, and no
fabricated result ever reaches ``experiment_runs`` or the generated file.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from experiments.make_tables import (
    REPO_ROOT,
    Run,
    aggregate,
    build_document,
    fmt_bool,
    fmt_cell,
    main,
    render_r1,
    render_r2,
    render_r3,
    render_r3b,
)


def _run(exp: str, arm: str, seed: int, metrics: Mapping[str, Any]) -> Run:
    full = {"experiment": exp, "arm": arm, "seed": seed, **metrics}
    return Run(exp=exp, arm=arm, seed=seed, n_agents=metrics.get("n_agents"), metrics=full)


def _row_cells(table_md: str, label: str) -> list[str]:
    """Return the trimmed cells of the first table row whose first cell == label."""
    for line in table_md.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == label:
            return cells
    raise AssertionError(f"row {label!r} not found")


# -----------------------------------------------------------------------------
# Empty DB — the current, correct state of the repo
# -----------------------------------------------------------------------------


def test_empty_runs_render_all_pendiente_r1_r2_r3_r3b():
    doc = build_document([])
    # Every table is present.
    assert "## R1 — Concurrency (E1)" in doc
    assert "## R2 — Chaos (E2)" in doc
    assert "## R3 — Knowledge-update (E3), poisoning (E4), cost (E5)" in doc
    assert "## R3b — External validation on LongMemEval (E3b)" in doc
    # No data cell holds a number; every filled cell is pendiente or a dash.
    assert "pendiente" in doc
    assert "0 ±" not in doc  # no fabricated zero-inconsistency claim


def test_r1_skeleton_matches_section_8_5_rows_and_columns():
    table = render_r1(aggregate([]))
    header = _row_cells(table, "Backend")
    assert header == ["Backend", "N agents", "Inconsist./1000 wr", "p95 write ms", "p95 read ms"]
    # Rows in §8.5 order.
    assert _row_cells(table, "Naive (non-transactional)")[1] == "5"
    crdb = _row_cells(table, "CockroachDB SERIALIZABLE")
    assert crdb[1] == "5"


def test_generated_doc_contains_no_prediction_markers():
    doc = build_document([])
    assert "▲" not in doc  # ▲
    assert "▼" not in doc  # ▼


def test_crdb_inconsistency_cell_is_pendiente_not_zero_when_unrun():
    table = render_r1(aggregate([]))
    # The pre-registration predicts 0 here; the GENERATED doc must not inherit it.
    crdb = _row_cells(table, "CockroachDB SERIALIZABLE")
    inconsistency_cell = crdb[2]
    assert inconsistency_cell == "pendiente"
    assert inconsistency_cell != "0"


# -----------------------------------------------------------------------------
# Synthetic rows — exact formatting
# -----------------------------------------------------------------------------


def test_synthetic_rows_produce_exact_mean_plus_minus_sigma():
    # inconsistencies [2, 4, 6] -> mean 4, population sigma sqrt(8/3)=1.6329931 -> 1.63.
    runs = [
        _run("E1", "E1_naive_n20", s, {"inconsistencies_per_1000": v})
        for s, v in ((0, 2.0), (1, 4.0), (2, 6.0))
    ]
    table = render_r1(aggregate(runs))
    row = _row_cells(table, "Naive")  # first "Naive" row is n=20
    assert row[1] == "20"
    assert row[2] == "4 ± 1.63"
    # Untouched columns stay pendiente — never backfilled.
    assert row[3] == "pendiente"


def test_single_rep_formats_with_n_equals_1():
    runs = [_run("E1", "E1_naive_n5", 0, {"p95_write_ms": 12.5})]
    table = render_r1(aggregate(runs))
    row = _row_cells(table, "Naive (non-transactional)")
    assert row[3] == "12.5 (n=1)"


def test_not_applicable_cells_render_em_dash():
    table = render_r3(aggregate([]))
    a1 = _row_cells(table, "A1 -consolidation")
    # KU accuracy + % stale fact are pendiente; poison/falsepos/tokens are N/A.
    assert a1[1] == "pendiente"
    assert a1[3] == "—"  # Poison det. % — structurally N/A
    assert a1[4] == "—"


def test_e2_boolean_consensus_and_mixed():
    # fleet_kept_operating all True -> "yes"; integrity_ok all True -> corruption "no".
    yes_runs = [
        _run(
            "E2", "E2_crdb_3node_kill_node", s, {"fleet_kept_operating": True, "integrity_ok": True}
        )
        for s in range(3)
    ]
    table = render_r2(aggregate(yes_runs))
    row = _row_cells(table, "kill node")
    assert row[3] == "no"  # Corruption = not integrity_ok
    assert row[5] == "yes"  # Fleet kept operating

    mixed_runs = [
        _run("E2", "E2_crdb_3node_kill_node", 0, {"fleet_kept_operating": True}),
        _run("E2", "E2_crdb_3node_kill_node", 1, {"fleet_kept_operating": False}),
    ]
    row = _row_cells(render_r2(aggregate(mixed_runs)), "kill node")
    assert row[5] == "mixed"


def test_unknown_arm_row_excluded_and_listed_in_footnote():
    runs = [_run("E1", "totally_bogus_arm", 0, {"inconsistencies_per_1000": 999.0})]
    doc = build_document(runs)
    assert "totally_bogus_arm" in doc  # surfaced in the audit footnote
    assert "999" not in doc  # never rendered into a table
    assert "Audit:" in doc


def test_missing_metric_key_yields_pendiente_cell():
    # A row exists for (E3, A0_full) but carries no ku_accuracy -> pendiente, not 0.
    runs = [_run("E3", "A0_full", 0, {"stale_fact_rate": 0.1})]
    table = render_r3(aggregate(runs))
    row = _row_cells(table, "A0 full")
    assert row[1] == "pendiente"  # KU accuracy: missing key -> pendiente
    assert row[2] == "0.1 (n=1)"  # stale fact rate: present


def test_shared_arm_disambiguated_by_experiment():
    # A0_full runs in E3, E4 and E5; the exp key routes each to its own column.
    runs = [
        _run("E3", "A0_full", 0, {"ku_accuracy": 0.9}),
        _run("E4", "A0_full", 0, {"detection_rate": 0.8}),
        _run("E5", "A0_full", 0, {"tokens_per_query": 1200.0}),
    ]
    row = _row_cells(render_r3(aggregate(runs)), "A0 full")
    assert row[1] == "0.9 (n=1)"  # KU accuracy (E3)
    assert row[3] == "0.8 (n=1)"  # Poison det. % (E4)
    assert row[5] == "1.2e+03 (n=1)"  # Tokens/query (E5)


def test_r3b_renders_lme_columns():
    runs = [_run("E3b", "A0_full", 0, {"lme_ku_acc": 0.75, "lme_temporal_acc": 0.5})]
    table = render_r3b(aggregate(runs))
    row = _row_cells(table, "A0 full")
    assert row[1] == "0.75 (n=1)"
    assert row[2] == "0.5 (n=1)"


# -----------------------------------------------------------------------------
# Pure formatting units
# -----------------------------------------------------------------------------


def test_fmt_cell_edge_cases():
    assert fmt_cell([]) == "pendiente"
    assert fmt_cell([5.0]) == "5 (n=1)"
    assert fmt_cell([1.0, 3.0]) == "2 ± 1"


def test_fmt_bool_edge_cases():
    assert fmt_bool([]) == "pendiente"
    assert fmt_bool([True, True]) == "yes"
    assert fmt_bool([False, False]) == "no"
    assert fmt_bool([True, False]) == "mixed"


def test_build_document_is_pure_and_deterministic(tmp_path):
    # Same input -> identical output, and building writes no file.
    before = sorted(tmp_path.iterdir())
    doc1 = build_document([])
    doc2 = build_document([])
    assert doc1 == doc2
    assert isinstance(doc1, str)
    assert sorted(tmp_path.iterdir()) == before


# -----------------------------------------------------------------------------
# CLI guards — docs/results.md is sacred
# -----------------------------------------------------------------------------


def test_main_refuses_to_write_docs_results_md():
    results_md = REPO_ROOT / "docs" / "results.md"
    rc = main(["--out", str(results_md)], fetch=lambda dsn: [])
    assert rc == 2


def test_main_to_tmp_leaves_docs_results_md_unchanged(tmp_path):
    results_md = REPO_ROOT / "docs" / "results.md"
    before = hashlib.sha256(results_md.read_bytes()).hexdigest()

    out = tmp_path / "results_generated.md"
    rc = main(["--out", str(out)], fetch=lambda dsn: [])
    assert rc == 0

    after = hashlib.sha256(results_md.read_bytes()).hexdigest()
    assert before == after
    written = out.read_text(encoding="utf-8")
    assert "pendiente" in written
    assert "▲" not in written and "▼" not in written


def test_main_reports_a_fetch_error_without_crashing(tmp_path):
    def _boom(_dsn):
        raise RuntimeError("no cluster")

    out = tmp_path / "results_generated.md"
    rc = main(["--out", str(out)], fetch=_boom)
    assert rc == 2
    assert not out.exists()
