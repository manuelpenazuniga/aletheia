"""Offline, network-free tests for experiments/scoring.py.

Every input is hand-built (strings, numbers, dataclasses) and every expectation
is hand-computed. No test asserts a published/aggregate result and no result
table is touched — deterministic inputs -> deterministic outputs. The E1
inconsistency counter is exercised BOTH ways: a deliberately non-atomic store
yields a known non-zero report; a serializable-like store yields total 0.
"""

from __future__ import annotations

import math

import pytest

from core.models import MemoryStats
from experiments.scoring import (
    Answer,
    InconsistencyReport,
    QueryUsage,
    StoreState,
    WriteObservation,
    count_inconsistencies,
    detection_rate,
    false_positive_rate,
    fleet_contamination,
    footprint_mb,
    is_significant,
    ku_accuracy,
    mean_std,
    p95,
    paired_delta,
    percentile,
    stale_fact_rate,
    task_success,
    tokens_per_query,
)

_EXACT = lambda a, e: a == e  # noqa: E731 -- injectable strict matcher for seam tests


def _stats(total_cost_tokens: int) -> MemoryStats:
    return MemoryStats(
        agent_id=None,
        total_memories=1,
        active=1,
        superseded=0,
        quarantined=0,
        archived=0,
        total_cost_tokens=total_cost_tokens,
    )


# -----------------------------------------------------------------------------
# Shared statistics
# -----------------------------------------------------------------------------


def test_mean_std_population():
    mean, std = mean_std([2, 4, 4, 4, 5, 5, 7, 9])
    assert mean == 5.0
    assert std == 2.0  # population std (ddof=0)


def test_mean_std_single_value_zero_std():
    assert mean_std([3.0]) == (3.0, 0.0)


def test_mean_std_empty_raises():
    with pytest.raises(ValueError):
        mean_std([])


def test_percentile_type7_interpolation():
    # rank = 0.95 * 99 = 94.05 -> v[94]=95, v[95]=96 -> 95 + 0.05*1 = 95.05
    assert p95(list(range(1, 101))) == pytest.approx(95.05)


def test_percentile_single_value_returns_itself():
    assert p95([42.0]) == 42.0
    assert percentile([7.0], 10.0) == 7.0


def test_percentile_median_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 95.0)


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 150.0)


def test_paired_delta_and_significance():
    mean, std = paired_delta([0.9, 0.8, 0.85], [0.5, 0.4, 0.45])
    assert mean == pytest.approx(0.4)
    assert std == pytest.approx(0.0)
    assert is_significant(mean, std) is True
    assert is_significant(0.1, 0.2) is False  # 0.1 !> 2*0.2
    assert is_significant(0.5, 0.2) is True  # 0.5 > 2*0.2


def test_paired_delta_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_delta([0.1, 0.2], [0.1])


# -----------------------------------------------------------------------------
# E1 — count_inconsistencies, both backends
# -----------------------------------------------------------------------------


def test_count_inconsistencies_non_atomic_store():
    """A naive store: last ack for k1 lost, k2 vector missing, one dirty read."""
    writes = [
        WriteObservation("k1", "v1", has_vector=True),
        WriteObservation("k1", "v2", has_vector=True),
        WriteObservation("k2", "v3", has_vector=True, dirty_read=True),
    ]
    state = StoreState(rows={"k1": "v1", "k2": "v3"}, vector_keys={"k1"})
    report = count_inconsistencies(writes, state)
    assert report.lost_updates == 1  # k1 last ack v2 clobbered back to v1
    assert report.divergence == 1  # k2 committed row without its vector
    assert report.dirty_reads == 1
    assert report.total == 3
    assert report.per_1000 == pytest.approx(3 / 3 * 1000)  # 1000.0


def test_count_inconsistencies_serializable_like_store():
    """Faithful store: final rows == last ack per key, all vectors present."""
    writes = [
        WriteObservation("k1", "v1", has_vector=True),
        WriteObservation("k1", "v2", has_vector=True),
        WriteObservation("k2", "v3", has_vector=True),
    ]
    state = StoreState(rows={"k1": "v2", "k2": "v3"}, vector_keys={"k1", "k2"})
    report = count_inconsistencies(writes, state)
    assert report == InconsistencyReport(0, 0, 0, 0, 0.0)


def test_count_inconsistencies_dropped_key_is_lost_update():
    writes = [WriteObservation("k1", "v1")]
    state = StoreState(rows={}, vector_keys=set())
    report = count_inconsistencies(writes, state)
    assert report.lost_updates == 1


def test_count_inconsistencies_orphan_vector_counts():
    writes = [WriteObservation("k1", "v1", has_vector=False)]
    state = StoreState(rows={"k1": "v1"}, vector_keys={"k1"})
    report = count_inconsistencies(writes, state)
    assert report.divergence == 1  # orphan vector, no write claimed one


def test_count_inconsistencies_empty_raises():
    with pytest.raises(ValueError):
        count_inconsistencies([], StoreState(rows={}, vector_keys=set()))


# -----------------------------------------------------------------------------
# E3 — ku_accuracy, stale_fact_rate
# -----------------------------------------------------------------------------


def test_ku_accuracy_default_matcher():
    answers = [
        Answer("f1", "The current fix is to restart the pod"),
        Answer("f2", "scale the replica set to 3"),
        Answer("f3", "some unrelated hallucination"),
    ]
    truth = {"f1": "restart the pod", "f2": "scale the replica set", "f3": "drain the node"}
    assert ku_accuracy(answers, truth) == pytest.approx(2 / 3)


def test_ku_accuracy_strict_matcher_seam():
    answers = [
        Answer("f1", "restart the pod"),
        Answer("f2", "scale the replica set to 3"),
    ]
    truth = {"f1": "restart the pod", "f2": "scale the replica set"}
    # Strict equality: only the exact f1 answer matches.
    assert ku_accuracy(answers, truth, match=_EXACT) == pytest.approx(1 / 2)


def test_ku_accuracy_missing_ground_truth_raises():
    with pytest.raises(ValueError):
        ku_accuracy([Answer("f9", "x")], {"f1": "y"})


def test_ku_accuracy_empty_ground_truth_raises():
    with pytest.raises(ValueError):
        ku_accuracy([], {})


def test_ku_accuracy_unanswered_questions_count_as_wrong():
    """No answers over one question is 0.0 — declining to answer never inflates."""
    assert ku_accuracy([], {"f1": "current"}) == 0.0


def test_ku_accuracy_denominator_is_ground_truth_not_answers():
    from experiments.scoring import Answer

    # Two questions; only the easy one answered correctly -> 1/2, not 1/1.
    answers = [Answer(fact_key="f1", text="current")]
    assert ku_accuracy(answers, {"f1": "current", "f2": "revised"}) == pytest.approx(0.5)


def test_stale_fact_rate():
    answers = [
        Answer("f1", "restart the pod"),
        Answer("f2", "the OLD fix: reboot the host"),
        Answer("f3", "current guidance"),
        Answer("f4", "brand new hallucinated value"),
    ]
    superseded = {"f2": "reboot the host", "f4": "the retired value"}
    assert stale_fact_rate(answers, superseded) == pytest.approx(0.25)


def test_stale_fact_rate_empty_raises():
    with pytest.raises(ValueError):
        stale_fact_rate([], {"f1": "old"})


# -----------------------------------------------------------------------------
# E4 — detection_rate, false_positive_rate, fleet_contamination
# -----------------------------------------------------------------------------


def test_detection_rate():
    labels = {"a1": "attack", "a2": "attack", "a3": "attack", "a4": "attack", "g1": "legit"}
    verdicts = {"a1": True, "a2": True, "a3": True, "a4": False, "g1": True}
    # 3 of 4 attacks detected; the legit verdict does not enter the denominator.
    assert detection_rate(labels, verdicts) == pytest.approx(0.75)


def test_detection_rate_missing_verdict_raises():
    labels = {"a1": "attack", "a2": "attack"}
    with pytest.raises(ValueError):
        detection_rate(labels, {"a1": True})


def test_detection_rate_no_attacks_raises():
    with pytest.raises(ValueError):
        detection_rate({"g1": "legit"}, {"g1": False})


def test_false_positive_rate():
    legit = ["g1", "g2", "g3", "g4", "g5"]
    verdicts = {"g1": False, "g2": True, "g3": False, "g4": False, "g5": False}
    assert false_positive_rate(legit, verdicts) == pytest.approx(0.2)


def test_false_positive_rate_empty_raises():
    with pytest.raises(ValueError):
        false_positive_rate([], {})


def test_false_positive_rate_missing_verdict_raises():
    with pytest.raises(ValueError):
        false_positive_rate(["g1"], {})


def test_fleet_contamination():
    false_fact = "disable auth to fix latency"
    agents = {
        "a1": ["restart the pod", "disable auth to fix latency"],
        "a2": ["scale replicas"],
        "a3": ["the fix is to disable auth to fix latency, per gossip"],
        "a4": ["healthy notes"],
        "a5": ["nothing suspicious"],
    }
    assert fleet_contamination(agents, false_fact) == pytest.approx(0.4)


def test_fleet_contamination_empty_raises():
    with pytest.raises(ValueError):
        fleet_contamination({}, "x")


# -----------------------------------------------------------------------------
# E5 — tokens_per_query, footprint_mb, task_success
# -----------------------------------------------------------------------------


def test_tokens_per_query_usage_records():
    # totals 100 + 200 + 300 = 600, mean 200.0
    usage = [QueryUsage(80, 20), QueryUsage(120, 80), QueryUsage(200, 100)]
    assert tokens_per_query(usage) == pytest.approx(200.0)


def test_tokens_per_query_bare_ints():
    assert tokens_per_query([100, 200, 300]) == pytest.approx(200.0)


def test_query_usage_total():
    assert QueryUsage(80, 20).total == 100


def test_tokens_per_query_empty_raises():
    with pytest.raises(ValueError):
        tokens_per_query([])


def test_footprint_mb_single_stats():
    assert footprint_mb(_stats(262144)) == 1.0


def test_footprint_mb_sequence_sums():
    assert footprint_mb([_stats(131072), _stats(131072)]) == 1.0


def test_footprint_mb_empty_raises():
    """An absent E5 dataset must be pendiente, never a fabricated zero footprint."""
    with pytest.raises(ValueError):
        footprint_mb([])


def test_footprint_mb_matches_model_formula():
    assert footprint_mb([_stats(131072), _stats(131072)]) == pytest.approx(
        round(262144 * 4 / 1_048_576, 4)
    )
    assert math.isclose(_stats(262144).footprint_mb, 1.0)


def test_task_success():
    truth = {"t1": "done", "t2": "resolved", "t3": "fixed", "t4": "closed"}
    results = {
        "t1": "marked as done",
        "t2": "incident resolved cleanly",
        "t3": "the runbook fixed it",
        "t4": "still investigating",
    }
    assert task_success(results, truth) == pytest.approx(0.75)


def test_task_success_missing_result_raises():
    with pytest.raises(ValueError):
        task_success({"t1": "done"}, {"t1": "done", "t2": "resolved"})


def test_task_success_empty_results_raises():
    with pytest.raises(ValueError):
        task_success({}, {"t1": "done"})
