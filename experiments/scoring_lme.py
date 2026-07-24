"""Scoring adapter for the LongMemEval subset (experiment E3b, the project plan §8.4).

E3b replays the E3 arms (``A0_full``, ``A1_no_consolidation``, ``BL_rag``) on a
public benchmark so the knowledge-update effect is falsifiable by someone other
than us. This module is the translation layer between LongMemEval's format and
Aletheia's, for the *knowledge-update* and *temporal-reasoning* categories only.

Importable without the data
---------------------------
Nothing here reads the benchmark at import time, and the offline tests exercise
only the pure pieces (:func:`load_subset`, :func:`score`, :func:`to_metrics`). The
benchmark content is **not** committed (licensing — see
``scenarios/longmemeval/SUBSET.md``); :func:`load_benchmark_items` raises
:class:`LmeDataUnavailable` when the downloaded data is absent, so a run without it
fails loudly rather than fabricating a figure. Until the subset in ``SUBSET.md`` is
selected, :func:`load_subset` returns an empty subset and **no LongMemEval number
can leak** into ``experiment_runs`` (the project plan §3.1).

The session -> provenance mapping
---------------------------------
LongMemEval is organised as user/assistant *sessions*; Aletheia's write path is
organised around *which agent observed what*, with signed provenance. The mapping
is a real modelling decision that affects the E3b number, so it is named,
versioned (:data:`MEMORY_MAPPING_VERSION`) and written into the metrics row for
auditability — and stated as a limitation in the README/video, not buried:

``one_session_one_agent_v1`` — each conversational session maps to one synthetic
agent ``lme:<session_id>``; each fact-bearing turn becomes a :class:`MemoryEvent`
authored by that agent with ``hop_count=0`` and an HMAC signature over its content
under a per-session demo token; the turn's timestamp becomes ``created_at`` so the
consolidation cycle can see a fact revised across turns (the knowledge-update
signal E3b measures). This exercises the retrieval + consolidation path only, not
gossip or the immune gate — faithful to what E3b claims.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.models import MemoryEvent, MemoryKind, utcnow

__all__ = [
    "MEMORY_MAPPING_VERSION",
    "Answer",
    "LmeDataUnavailable",
    "LmeItem",
    "QuestionRef",
    "Subset",
    "build_provenance",
    "load_benchmark_items",
    "load_subset",
    "run_arm",
    "score",
    "to_metrics",
]

#: The versioned session->provenance mapping, recorded in every E3b metrics row.
MEMORY_MAPPING_VERSION = "one_session_one_agent_v1"

#: The two categories E3b covers (the consolidation cycle makes a claim about
#: exactly these). Canonicalised to their LongMemEval spellings.
KNOWLEDGE_UPDATE = "knowledge-update"
TEMPORAL_REASONING = "temporal-reasoning"
_CATEGORIES = (KNOWLEDGE_UPDATE, TEMPORAL_REASONING)

_DEFAULT_SUBSET_MD = Path(__file__).resolve().parents[1] / "scenarios" / "longmemeval" / "SUBSET.md"

#: Cells whose text marks "not selected yet" in SUBSET.md — treated as no data.
_PLACEHOLDERS = frozenset({"", "-", "—", "pendiente", "tbd", "n/a"})


class LmeDataUnavailable(Exception):
    """The downloaded LongMemEval data is missing — raised instead of a fake score."""


# -----------------------------------------------------------------------------
# Subset (pinned IDs) — parsed from SUBSET.md, never re-sampled
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuestionRef:
    """One pinned benchmark question: its id and its category."""

    qid: str
    category: str


@dataclass(frozen=True, slots=True)
class Subset:
    """The frozen E3b subset: the pinned refs plus the provenance of the selection."""

    refs: Sequence[QuestionRef]
    release: str | None = None
    seed: int | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.refs) == 0


def _normalize_category(text: str) -> str:
    """Canonicalise a category cell to its LongMemEval spelling.

    Accepts ``knowledge_update`` / ``knowledge update`` / ``knowledge-update`` and
    the temporal variants, so a subset table written with either separator parses.
    """
    t = text.strip().lower().replace("_", "-").replace(" ", "-")
    if t.startswith("knowledge"):
        return KNOWLEDGE_UPDATE
    if t.startswith("temporal"):
        return TEMPORAL_REASONING
    return t


def _table_cells(line: str) -> list[str] | None:
    """Split a markdown table row into trimmed cells, or ``None`` if not a row."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    # A separator row (---, :---:) carries no data.
    if all(set(c) <= set("-: ") and c for c in cells):
        return None
    return cells


def load_subset(subset_md_path: str | Path = _DEFAULT_SUBSET_MD) -> Subset:
    """Parse the pinned IDs and selection parameters from ``SUBSET.md``.

    Reads the two markdown tables (the 4-column "Selected question IDs" table and
    the 2-column selection-parameters table). Placeholder rows — the ``pendiente``
    markers present until the subset is drawn — are skipped, so on the real
    placeholder file this returns an empty :class:`Subset` cleanly, inventing no
    IDs. The subset is committed once and never re-rolled between arms or seeds.
    """
    text = Path(subset_md_path).read_text(encoding="utf-8")
    refs: list[QuestionRef] = []
    release: str | None = None
    seed: int | None = None

    for line in text.splitlines():
        cells = _table_cells(line)
        if cells is None:
            continue
        if len(cells) == 4:  # | # | Question ID | Category | Notes |
            _, qid, category, _notes = cells
            if qid.strip().lower() in _PLACEHOLDERS or qid.strip().lower() == "question id":
                continue
            refs.append(QuestionRef(qid=qid.strip(), category=_normalize_category(category)))
        elif len(cells) == 2:  # | Parameter | Value |
            key, value = cells[0].strip().lower(), cells[1].strip()
            if value.lower() in _PLACEHOLDERS:
                continue
            if "release" in key or "commit" in key:
                release = value
            elif "sampling seed" in key or key == "seed":
                try:
                    seed = int(value)
                except ValueError:
                    seed = None
    return Subset(refs=tuple(refs), release=release, seed=seed)


# -----------------------------------------------------------------------------
# Benchmark items (the downloaded data)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LmeItem:
    """One graded benchmark question, restricted to the fields E3b needs.

    ``sessions`` is the raw conversational history (list of sessions, each a
    mapping with ``session_id`` and ``turns``); :func:`build_provenance` turns it
    into memory writes. ``question`` / ``answer`` are graded by :func:`score`.
    """

    qid: str
    category: str
    question: str
    answer: str
    sessions: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


def load_benchmark_items(data_dir: str | Path | None, subset: Subset) -> list[LmeItem]:
    """Load the pinned subset's items from downloaded LongMemEval JSON.

    Raises :class:`LmeDataUnavailable` when ``data_dir`` is missing or empty — the
    benchmark content is not committed (licensing), so a real E3b run must point
    this at a local download. The offline tests assert the raise, documenting that
    dependency rather than papering over it.
    """
    if data_dir is None:
        raise LmeDataUnavailable(
            "no LongMemEval data_dir given; download the benchmark per "
            "scenarios/longmemeval/SUBSET.md (its content is not committed)"
        )
    root = Path(data_dir)
    json_files = sorted(root.glob("*.json")) if root.is_dir() else []
    if not json_files:
        raise LmeDataUnavailable(
            f"no LongMemEval JSON found under {root}; download the benchmark per "
            "scenarios/longmemeval/SUBSET.md"
        )

    wanted = {ref.qid: ref.category for ref in subset.refs}
    items: list[LmeItem] = []
    seen: set[str] = set()
    for path in json_files:
        records = json.loads(path.read_text(encoding="utf-8"))
        for rec in records:
            qid = str(rec.get("question_id") or rec.get("qid") or "")
            if qid not in wanted or qid in seen:
                continue
            seen.add(qid)
            items.append(
                LmeItem(
                    qid=qid,
                    category=wanted[qid],
                    question=str(rec.get("question", "")),
                    answer=str(rec.get("answer", "")),
                    sessions=tuple(rec.get("sessions") or rec.get("haystack_sessions") or ()),
                )
            )
    return items


# -----------------------------------------------------------------------------
# Session -> provenance mapping (one_session_one_agent_v1)
# -----------------------------------------------------------------------------


def build_provenance(
    session: Mapping[str, Any], *, token: str = "lme-demo-token"
) -> list[MemoryEvent]:
    """Map one LongMemEval session onto signed :class:`MemoryEvent` writes.

    ``one_session_one_agent_v1``: one synthetic agent per session, one memory per
    fact-bearing turn, HMAC-signed under a per-session demo token, timestamped from
    the turn so consolidation can observe a fact being revised across turns.
    """
    session_id = str(session.get("session_id") or session.get("id") or "unknown")
    agent_id = f"lme:{session_id}"
    events: list[MemoryEvent] = []
    for turn in session.get("turns", []):
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if not content:
            continue
        signature = hmac.new(token.encode(), content.encode(), hashlib.sha256).hexdigest()
        events.append(
            MemoryEvent(
                agent_id=agent_id,
                content=content,
                kind=MemoryKind.EPISODIC,
                hop_count=0,
                signature=signature,
                created_at=_parse_ts(turn.get("timestamp")),
                meta={"session_id": session_id, "mapping": MEMORY_MAPPING_VERSION},
            )
        )
    return events


def _parse_ts(value: Any) -> Any:
    from datetime import datetime

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return utcnow()
    return utcnow()


# -----------------------------------------------------------------------------
# Answering (needs Bedrock) and scoring (pure)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Answer:
    """The model's answer to one benchmark question, keyed by its id."""

    qid: str
    text: str


def run_arm(
    config: Any,
    adapter: Any,
    items: Sequence[LmeItem],
    embedder: Any,
    model: Any,
    *,
    seed: int,
) -> list[Answer]:  # pragma: no cover - needs Bedrock + a cluster
    """Answer every subset item through the memory layer for one arm.

    Writes each item's sessions via :func:`build_provenance` (registering each
    synthetic author first), runs the consolidation cycle when the arm enables it
    (the sole A0-vs-A1 difference), retrieves as a registered reader within the
    budget, and asks the model via ``complete``. Requires real embeddings and
    inference (Bedrock), so E3b RESULT cells stay ``pendiente`` here — this is the
    live path, never exercised offline, and it fabricates nothing.

    Isolation: for a clean per-question measurement the caller should pass an
    adapter scoped to one item (or a fresh run_id) so earlier items' memories do
    not leak into a later item's retrieval.
    """
    # Imported here (live path): keeps the module importable + unit-testable
    # without pulling the consolidation cycle or the model request type.
    from agents.model_client import ModelRequest
    from core.consolidation import consolidate

    reader_id = f"lme-reader-s{seed}"
    adapter.register_agent(reader_id, "sre", "lme-reader")

    answers: list[Answer] = []
    for item in items:
        # Register each session's synthetic author BEFORE its writes — both real
        # adapters reject an unregistered agent (the memories->agents FK), so the
        # first write would otherwise raise.
        for session in item.sessions:
            events = list(build_provenance(session))
            for event in events:
                adapter.register_agent(event.agent_id, "sre", "lme-session")
                event.embedding = embedder.embed(event.content)
                adapter.write_episode(event.agent_id, event)

        # Run the consolidation cycle when the arm enables it — this is the ONLY
        # thing that distinguishes A0_full from A1_no_consolidation, so E3b is
        # meaningless without it (external review: A0 and A1 were identical).
        if config.enable_consolidation:
            consolidate(adapter, config)

        # Retrieve as a registered reader (not an unregistered per-question agent),
        # under the shared retrieval budget (§8.7 control).
        context = adapter.query_semantic(
            reader_id,
            embedder.embed(item.question),
            k=5,
            budget_tokens=config.retrieval_budget_tokens,
        )
        prompt = "\n".join(f"- {hit.content}" for hit in context)
        response = model.complete(
            ModelRequest(
                system=(
                    "Answer the question using ONLY the retrieved memory below. "
                    "Prefer the most recent fact if they conflict."
                ),
                messages=[
                    {"role": "user", "content": f"Memory:\n{prompt}\n\nQuestion: {item.question}"}
                ],
                tag=item.qid,
                temperature=config.temperature,
            )
        )
        answers.append(Answer(qid=item.qid, text=response.text))
    return answers


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _default_correct(answer: str, gold: str) -> bool:
    """Lenient offline stand-in for the benchmark's official grader.

    A normalised substring test. The real E3b run injects LongMemEval's own
    correctness function via ``correctness=`` so the figure stays comparable to
    published results; the offline tests use this deterministic default.
    """
    return _normalize(gold) in _normalize(answer)


def score(
    answers: Sequence[Answer],
    items: Sequence[LmeItem],
    *,
    correctness: Callable[[str, str], bool] | None = None,
) -> dict[str, Any]:
    """Per-category accuracy for the subset. Pure; grades no cell it lacks an answer for.

    Returns ``{lme_ku_acc, lme_temporal_acc, n_ku, n_temporal}``. A category with
    no items scores ``None`` (never a flattering ``0``/``1``). Every item must have
    an answer — a missing one raises ``ValueError`` rather than silently scoring a
    subset (the project plan §3.1). ``correctness`` defaults to a normalised
    substring match; the live run injects the benchmark's official grader.
    """
    grade = correctness if correctness is not None else _default_correct
    by_qid = {a.qid: a.text for a in answers}
    correct = {KNOWLEDGE_UPDATE: 0, TEMPORAL_REASONING: 0}
    total = {KNOWLEDGE_UPDATE: 0, TEMPORAL_REASONING: 0}
    for item in items:
        if item.category not in correct:
            raise ValueError(f"item {item.qid!r} has category {item.category!r} outside E3b scope")
        if item.qid not in by_qid:
            raise ValueError(f"no answer for benchmark item {item.qid!r}")
        total[item.category] += 1
        if grade(by_qid[item.qid], item.answer):
            correct[item.category] += 1

    def _acc(cat: str) -> float | None:
        return correct[cat] / total[cat] if total[cat] else None

    return {
        "lme_ku_acc": _acc(KNOWLEDGE_UPDATE),
        "lme_temporal_acc": _acc(TEMPORAL_REASONING),
        "n_ku": total[KNOWLEDGE_UPDATE],
        "n_temporal": total[TEMPORAL_REASONING],
    }


def to_metrics(
    scored: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    mapping: str = MEMORY_MAPPING_VERSION,
    release: str | None = None,
) -> dict[str, Any]:
    """Shape a :func:`score` result into an ``experiment_runs`` metrics row (E3b).

    Tagged ``experiment="E3b"`` so :mod:`experiments.make_tables` routes it to the
    R3b columns, and stamped with the session->provenance ``mapping`` version and
    the benchmark ``release`` for reproducibility and audit.
    """
    return {
        "experiment": "E3b",
        "arm": arm,
        "seed": seed,
        "lme_ku_acc": scored.get("lme_ku_acc"),
        "lme_temporal_acc": scored.get("lme_temporal_acc"),
        "n_ku": scored.get("n_ku"),
        "n_temporal": scored.get("n_temporal"),
        "session_provenance_mapping": mapping,
        "benchmark_release": release,
    }
