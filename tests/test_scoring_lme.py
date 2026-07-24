"""Offline tests for experiments/scoring_lme.py (the E3b LongMemEval adapter).

The module is importable and testable without the benchmark data (its content is
not committed — licensing). These tests cover the pure pieces: parsing the pinned
subset (the real placeholder file yields an empty subset, inventing no IDs; a
filled fixture parses exactly), the clean failure when the data is absent, and the
per-category scorer. No LongMemEval accuracy number is ever emitted into
``experiment_runs`` here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.scoring_lme import (
    MEMORY_MAPPING_VERSION,
    Answer,
    LmeDataUnavailable,
    LmeItem,
    Subset,
    load_benchmark_items,
    load_subset,
    score,
    to_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "lme_subset_sample.md"
REAL_SUBSET = REPO_ROOT / "scenarios" / "longmemeval" / "SUBSET.md"


def test_lme_load_subset_placeholder_returns_empty():
    subset = load_subset(REAL_SUBSET)
    # The committed file is still a placeholder: no IDs may be invented.
    assert subset.is_empty
    assert list(subset.refs) == []
    assert subset.release is None
    assert subset.seed is None


def test_lme_load_subset_parses_pinned_ids_from_fixture():
    subset = load_subset(FIXTURE)
    qids = [r.qid for r in subset.refs]
    assert qids == ["ku_0001", "ku_0002", "tr_0001"]
    cats = [r.category for r in subset.refs]
    # knowledge_update (underscore variant) canonicalises to knowledge-update.
    assert cats == ["knowledge-update", "knowledge-update", "temporal-reasoning"]
    assert subset.release == "v1.0-abc1234"
    assert subset.seed == 7


def test_lme_load_benchmark_items_raises_when_data_absent(tmp_path):
    subset = load_subset(FIXTURE)
    # No data_dir at all.
    with pytest.raises(LmeDataUnavailable):
        load_benchmark_items(None, subset)
    # An empty directory (nothing downloaded) is just as unavailable.
    with pytest.raises(LmeDataUnavailable):
        load_benchmark_items(tmp_path, subset)


def test_lme_score_per_category_accuracy_with_a_wrong_answer():
    items = [
        LmeItem("ku_1", "knowledge-update", "q1?", "restart the pod"),
        LmeItem("ku_2", "knowledge-update", "q2?", "increase the timeout"),
        LmeItem("tr_1", "temporal-reasoning", "q3?", "before the outage"),
    ]
    answers = [
        Answer("ku_1", "you should restart the pod now"),  # correct (substring)
        Answer("ku_2", "just reboot the server"),  # wrong
        Answer("tr_1", "that happened before the outage"),  # correct
    ]
    scored = score(answers, items)
    assert scored["lme_ku_acc"] == 0.5  # 1 of 2
    assert scored["lme_temporal_acc"] == 1.0  # 1 of 1
    assert scored["n_ku"] == 2
    assert scored["n_temporal"] == 1


def test_lme_score_raises_on_missing_answer():
    items = [LmeItem("ku_1", "knowledge-update", "q?", "gold")]
    with pytest.raises(ValueError):
        score([], items)


def test_lme_score_empty_category_is_none_not_zero():
    items = [LmeItem("ku_1", "knowledge-update", "q?", "gold")]
    scored = score([Answer("ku_1", "gold")], items)
    assert scored["lme_ku_acc"] == 1.0
    assert scored["lme_temporal_acc"] is None  # no temporal items -> None, never a fake 0
    assert scored["n_temporal"] == 0


def test_lme_to_metrics_tags_exp_e3b_and_mapping_version():
    scored = {"lme_ku_acc": 0.9, "lme_temporal_acc": 0.7, "n_ku": 3, "n_temporal": 2}
    metrics = to_metrics(scored, arm="A0_full", seed=2, release="v1.0-abc1234")
    assert metrics["experiment"] == "E3b"
    assert metrics["arm"] == "A0_full"
    assert metrics["seed"] == 2
    assert metrics["session_provenance_mapping"] == MEMORY_MAPPING_VERSION
    assert metrics["benchmark_release"] == "v1.0-abc1234"
    assert metrics["lme_ku_acc"] == 0.9


def test_lme_score_rejects_out_of_scope_category():
    items = [LmeItem("x_1", "single-session-user", "q?", "gold")]
    with pytest.raises(ValueError):
        score([Answer("x_1", "gold")], items)


def test_subset_is_empty_property():
    assert Subset(refs=()).is_empty
    assert not Subset(refs=(_ref(),)).is_empty


def _ref():
    from experiments.scoring_lme import QuestionRef

    return QuestionRef(qid="ku_1", category="knowledge-update")
