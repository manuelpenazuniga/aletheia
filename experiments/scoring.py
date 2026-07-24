"""Pure, offline metric functions for the experiment harness (the project plan §8).

One group of functions per experiment, plus the shared statistics and text-match
helpers they lean on. Everything here is a pure function of plain
mappings/sequences and :mod:`core.models` types — it opens no connection, reads
no file, and imports nothing beyond the standard library and ``core.models``.
That keeps scoring trivially unit-testable and decoupled from the richer records
(``PoisonCase``, ``ImmuneVerdict``, ``AuditRecord``, scenario/agent objects): the
*runner* is responsible for adapting those down to the plain shapes below, so a
mapping bug lives in one place and cannot reach into scoring.

**Scoring computes no aggregate results.** It turns the per-run observations the
runner captured into the numbers that — only after real runs — populate R1/R3.
E1 (concurrency) is fully offline-testable now. E3/E5 accuracy over real LLM
answers needs Amazon Bedrock, so those RESULT cells stay ``pendiente``; the
functions here are exercised on synthetic inputs, never on fabricated results.

Two deliberate, documented choices so published figures are reproducible:

* **Population standard deviation** (``ddof=0``). Well-defined and ``0.0`` for a
  single value, deterministic — so the ``mean ± sigma`` in the tables has one meaning.
* **Percentile via linear interpolation between closest ranks** (numpy
  ``type-7`` / "inclusive"), so R1 latency figures match common tooling.

The default text matcher is a lenient normalised substring test. Real E3/E5
accuracy over LLM answers will want a stricter (possibly LLM-judge) matcher; every
text metric takes an injectable ``match`` callable for exactly that, and the
offline tests cover only deterministic matching.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.models import MemoryStats

__all__ = [
    "Answer",
    "InconsistencyReport",
    "QueryUsage",
    "StoreState",
    "WriteObservation",
    "count_inconsistencies",
    "detection_rate",
    "false_positive_rate",
    "fleet_contamination",
    "footprint_mb",
    "is_significant",
    "ku_accuracy",
    "mean_std",
    "p95",
    "paired_delta",
    "percentile",
    "stale_fact_rate",
    "task_success",
    "tokens_per_query",
]

# A sentinel distinct from any legitimate stored value (including ``None``), so a
# key that is entirely absent from the final rows is detected as a dropped write
# rather than mistaken for one that legitimately stored ``None``.
_MISSING: Any = object()

Matcher = Callable[[str, str], bool]


# -----------------------------------------------------------------------------
# Shared statistics
# -----------------------------------------------------------------------------


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Return ``(mean, population_std)`` of ``values``.

    Population std (``ddof=0``): variance is the mean squared deviation, so a
    single value yields ``0.0`` rather than an undefined sample std. This is the
    ``mean ± sigma`` reported in the results tables. Empty input raises ``ValueError`` —
    never a silent ``0.0`` that would flatter a table.
    """
    n = len(values)
    if n == 0:
        raise ValueError("mean_std requires at least one value")
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return mean, math.sqrt(var)


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile via linear interpolation between closest ranks.

    numpy ``type-7`` / "inclusive": ``rank = (q/100) * (n - 1)``, interpolated
    between the two surrounding order statistics. Pinned and documented so the R1
    latency figures are reproducible and agree with common tooling; changing the
    method silently shifts every published p95. A single value returns itself;
    empty input raises ``ValueError``.
    """
    n = len(values)
    if n == 0:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile q must be in [0, 100]")
    ordered = sorted(values)
    if n == 1:
        return ordered[0]
    rank = (q / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    frac = rank - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def p95(values: Sequence[float]) -> float:
    """The 95th percentile — the write/read latency figure reported in R1."""
    return percentile(values, 95.0)


def paired_delta(baseline: Sequence[float], variant: Sequence[float]) -> tuple[float, float]:
    """Mean/std of per-index (paired-by-seed) differences ``baseline - variant``.

    Pairing by seed removes cross-seed variance so an ablation's effect is read
    against its own reference run. Length mismatch raises ``ValueError`` — the
    two arms must have been run over the same seeds.
    """
    if len(baseline) != len(variant):
        raise ValueError(
            f"paired_delta needs equal-length inputs, got {len(baseline)} and {len(variant)}"
        )
    deltas = [b - v for b, v in zip(baseline, variant, strict=True)]
    return mean_std(deltas)


def is_significant(delta_mean: float, delta_std: float, k: float = 2.0) -> bool:
    """``|delta_mean| > k * delta_std`` — the §8.3 "< 2 sigma = not significant" rule."""
    return abs(delta_mean) > k * delta_std


# -----------------------------------------------------------------------------
# Text matching
# -----------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Casefold and collapse runs of whitespace to single spaces."""
    return " ".join(text.split()).casefold()


def _default_match(answer: str, expected: str) -> bool:
    """Lenient default: is ``expected`` a substring of ``answer`` once normalised?

    Case-insensitive and whitespace-insensitive so trivial formatting differences
    don't distort accuracy. A stricter matcher is injectable everywhere a text
    metric is scored.
    """
    return _normalize(expected) in _normalize(answer)


def _resolve(match: Matcher | None) -> Matcher:
    return match if match is not None else _default_match


# -----------------------------------------------------------------------------
# E1 — concurrency (fully offline-testable)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WriteObservation:
    """One acknowledged write as the E1 harness observed it, in ack order.

    ``key`` identifies the logical cell being written; ``value`` is what the
    writer acknowledged storing. ``has_vector`` marks writes that should also have
    produced a vector. ``dirty_read`` is set by the harness when its live
    read-modify-write observed a torn / never-committed intermediate state.
    """

    key: str
    value: Any
    has_vector: bool = False
    dirty_read: bool = False


@dataclass(frozen=True, slots=True)
class StoreState:
    """The backend's final state after an E1 run: committed rows and vector keys."""

    rows: Mapping[str, Any]
    vector_keys: Collection[str]


@dataclass(frozen=True, slots=True)
class InconsistencyReport:
    """Tallied inconsistencies for one E1 run (the metric behind Table R1)."""

    lost_updates: int
    divergence: int
    dirty_reads: int
    total: int
    per_1000: float


def count_inconsistencies(
    observed_writes: Sequence[WriteObservation], store_state: StoreState
) -> InconsistencyReport:
    """Count storage inconsistencies — the metric that separates the two backends.

    A serializable store commits in acknowledgement order, so the last ack for a
    key wins, every committed row keeps its vector, and no reader sees an
    intermediate state: this returns ``total == 0``. A naive, non-transactional
    store lets a stale concurrent writer clobber or drop the last ack, leaves rows
    without their vector (and vice versa), and exposes torn reads: this returns
    ``total > 0``. Empty input raises ``ValueError``.

    Three families are counted:

    1. **Lost updates** — per key, the value of the *last* observation in ack
       order must survive in ``rows``. A key missing entirely from ``rows``
       counts (a dropped write), as does one clobbered by a stale writer.
    2. **Vector↔row divergence** — both directions: a write that should have a
       vector whose key is absent from ``vector_keys`` (row without its vector —
       the gap this project's thesis targets), and an orphan vector key with no
       corresponding write.
    3. **Intermediate-state reads** — one per observation the harness flagged
       ``dirty_read``.
    """
    n = len(observed_writes)
    if n == 0:
        raise ValueError("count_inconsistencies requires at least one observed write")

    # (1) Lost updates: last-writer-in-ack-order must survive.
    last_write_value: dict[str, Any] = {}
    for w in observed_writes:
        last_write_value[w.key] = w.value
    lost_updates = sum(
        1
        for key, expected in last_write_value.items()
        if store_state.rows.get(key, _MISSING) != expected
    )

    # (2) Vector/row divergence, both directions.
    expected_vec_keys = {w.key for w in observed_writes if w.has_vector}
    stored_vec_keys = set(store_state.vector_keys)
    missing_vectors = len(expected_vec_keys - stored_vec_keys)
    orphan_vectors = len(stored_vec_keys - expected_vec_keys)
    divergence = missing_vectors + orphan_vectors

    # (3) Intermediate-state reads flagged live by the harness.
    dirty_reads = sum(1 for w in observed_writes if w.dirty_read)

    total = lost_updates + divergence + dirty_reads
    per_1000 = total / n * 1000
    return InconsistencyReport(
        lost_updates=lost_updates,
        divergence=divergence,
        dirty_reads=dirty_reads,
        total=total,
        per_1000=per_1000,
    )


# -----------------------------------------------------------------------------
# E3 — knowledge-update  (RESULT cells need Bedrock -> pendiente)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Answer:
    """An agent's answer about one fact, keyed by the fact it concerns."""

    fact_key: str
    text: str


def ku_accuracy(
    answers: Sequence[Answer],
    ground_truth: Mapping[str, str],
    match: Matcher | None = None,
) -> float:
    """Fraction of answers that match the *current* (post-update) fact.

    A missing ground-truth entry for an answered ``fact_key`` raises
    ``ValueError`` rather than silently skipping it (integrity: never score a
    subset). Empty ``answers`` raises ``ValueError``.
    """
    if not answers:
        raise ValueError("ku_accuracy requires at least one answer")
    matcher = _resolve(match)
    correct = 0
    for a in answers:
        if a.fact_key not in ground_truth:
            raise ValueError(f"no ground truth for fact_key {a.fact_key!r}")
        if matcher(a.text, ground_truth[a.fact_key]):
            correct += 1
    return correct / len(answers)


def stale_fact_rate(
    answers: Sequence[Answer],
    superseded_fact: Mapping[str, str],
    match: Matcher | None = None,
) -> float:
    """Fraction of answers that echo the *superseded* (obsolete) fact.

    Independent of :func:`ku_accuracy`: an answer may match neither the current
    nor the tracked stale value (a hallucinated third value lowers accuracy
    without raising this rate). Only ``fact_key`` present in ``superseded_fact``
    can count. Empty ``answers`` raises ``ValueError``.
    """
    if not answers:
        raise ValueError("stale_fact_rate requires at least one answer")
    matcher = _resolve(match)
    stale = 0
    for a in answers:
        obsolete = superseded_fact.get(a.fact_key)
        if obsolete is not None and matcher(a.text, obsolete):
            stale += 1
    return stale / len(answers)


# -----------------------------------------------------------------------------
# E4 — poisoning
# -----------------------------------------------------------------------------


def detection_rate(labels: Mapping[str, str], verdicts: Mapping[str, bool]) -> float:
    """Fraction of ground-truth attacks the immune gate quarantined.

    ``verdicts[id] is True`` means quarantined/detected — the runner maps
    ``ImmuneVerdict.accepted`` to ``not accepted``. Only ids labelled ``"attack"``
    count. A missing verdict for an attack id raises ``ValueError`` (every attack
    must have been adjudicated); zero attacks raises ``ValueError`` (no fake 1.0).
    """
    attack_ids = [id_ for id_, label in labels.items() if label == "attack"]
    if not attack_ids:
        raise ValueError("detection_rate requires at least one attack label")
    detected = 0
    for id_ in attack_ids:
        if id_ not in verdicts:
            raise ValueError(f"no verdict for attack case {id_!r}")
        if verdicts[id_] is True:
            detected += 1
    return detected / len(attack_ids)


def false_positive_rate(legit: Collection[str], verdicts: Mapping[str, bool]) -> float:
    """Fraction of *legitimate* memories the immune gate wrongly quarantined.

    A missing verdict for a legit id raises ``ValueError``; empty ``legit`` raises
    ``ValueError`` (no divide-by-zero).
    """
    if not legit:
        raise ValueError("false_positive_rate requires at least one legit case")
    false_positives = 0
    for id_ in legit:
        if id_ not in verdicts:
            raise ValueError(f"no verdict for legit case {id_!r}")
        if verdicts[id_] is True:
            false_positives += 1
    return false_positives / len(legit)


def fleet_contamination(
    agents: Mapping[str, Sequence[str]],
    false_fact: str,
    match: Matcher | None = None,
) -> float:
    """Fraction of agents whose memory contents echo the injected false fact.

    An agent is contaminated if *any* of its content strings matches
    ``false_fact``. Empty ``agents`` raises ``ValueError``.
    """
    if not agents:
        raise ValueError("fleet_contamination requires at least one agent")
    matcher = _resolve(match)
    contaminated = sum(
        1
        for contents in agents.values()
        if any(matcher(content, false_fact) for content in contents)
    )
    return contaminated / len(agents)


# -----------------------------------------------------------------------------
# E5 — cost  (RESULT cells need Bedrock -> pendiente)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryUsage:
    """Token usage for one query: prompt (input) and completion (output)."""

    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


def tokens_per_query(usage: Sequence[QueryUsage | int]) -> float:
    """Mean total tokens per query.

    Accepts :class:`QueryUsage` records (input+output) or bare pre-summed int
    totals, so both real usage records and pre-aggregated counts work. Empty input
    raises ``ValueError``.
    """
    if not usage:
        raise ValueError("tokens_per_query requires at least one usage record")
    totals = [u.total if isinstance(u, QueryUsage) else int(u) for u in usage]
    return sum(totals) / len(totals)


def footprint_mb(stats: MemoryStats | Sequence[MemoryStats]) -> float:
    """Retained-content footprint in MB, using the exact :class:`MemoryStats` formula.

    A single :class:`MemoryStats` returns its own ``footprint_mb``. A sequence
    (e.g. per-agent stats) is summed over ``total_cost_tokens`` and converted with
    the identical constant (``tokens * 4 / 1_048_576``, rounded to 4 places) so
    scoring and the model never disagree.
    """
    if isinstance(stats, MemoryStats):
        return stats.footprint_mb
    total = sum(s.total_cost_tokens for s in stats)
    return round(total * 4 / 1_048_576, 4)


def task_success(
    results: Mapping[str, str],
    ground_truth: Mapping[str, str],
    match: Matcher | None = None,
) -> float:
    """Fraction of tasks whose produced answer matches the expected one.

    Iterates over ``ground_truth`` so every graded task must have a result: a
    ground-truth id absent from ``results`` raises ``ValueError`` (never silently
    score a subset). Empty ``results`` raises ``ValueError``.
    """
    if not results:
        raise ValueError("task_success requires at least one result")
    if not ground_truth:
        raise ValueError("task_success requires ground truth")
    matcher = _resolve(match)
    successes = 0
    for task_id, expected in ground_truth.items():
        if task_id not in results:
            raise ValueError(f"no result for task {task_id!r}")
        if matcher(results[task_id], expected):
            successes += 1
    return successes / len(ground_truth)
