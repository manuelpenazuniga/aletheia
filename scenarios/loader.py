"""Typed, validating access to the committed scenario datasets.

The datasets under ``scenarios/`` are seeded, committed JSON (the project plan
§8.4): the SRE incident corpus (with runbook revisions — the knowledge-update
signal E3 keys on) and the poison suite (the labelled attacks and legitimate-but-
unusual memories E4 scores against). This module is the *only* sanctioned way to
read them: it validates every record against the schema and returns frozen
dataclasses, so a malformed file fails loudly at load rather than silently
skewing an experiment.

Portable: this is a leaf helper that speaks only :mod:`core.models` types. The
optional :class:`~core.embeddings.Embedder` seam is injected by the caller, so
embeddings are computed offline in tests and are *never* persisted to the JSON
(1024 floats per row would bloat every diff and couple the corpus to an embedder
seed — the project plan generation note).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.embeddings import Embedder
from core.models import MemoryEvent, MemoryKind, QuarantineReason

# -----------------------------------------------------------------------------
# Committed-file locations — resolved relative to this module, no importlib magic.
# -----------------------------------------------------------------------------

_HERE = Path(__file__).parent
COMMITTED_INCIDENTS_PATH = _HERE / "incidents" / "sample.json"
POISON_DIR = _HERE / "poison"
DISTRIBUTED_CLUES_PATH = _HERE / "distributed_clues" / "sample.json"

#: The three attack files (basename == category) plus the legitimate control.
POISON_ATTACK_FILES: dict[str, str] = {
    QuarantineReason.BAD_PROVENANCE.value: "bad_provenance.json",
    QuarantineReason.SEMANTIC_ANOMALY.value: "semantic_anomaly.json",
    QuarantineReason.INJECTION_PATTERN.value: "injection_pattern.json",
}
POISON_LEGIT_FILE = "legit_unusual.json"
LEGIT_CATEGORY = "legit"

#: The minimum every dataset file must carry (the project plan §8.4). Enforced at
#: load so a truncated file cannot quietly weaken an experiment's statistics.
MIN_ATTACKS_PER_CATEGORY = 30
MIN_LEGIT_CASES = 30

_INCIDENT_ID_RE = re.compile(r"^inc-[0-9]{4}$")
_POISON_ID_RE = re.compile(r"^(bad_provenance|semantic_anomaly|injection_pattern|legit)-[0-9]{4}$")
_FACT_KEY_RE = re.compile(r"^runbook:.+:fix$")

_VALID_POISON_CATEGORIES = frozenset(POISON_ATTACK_FILES) | {LEGIT_CATEGORY}
_VALID_LABELS = frozenset({"attack", "legit"})


class ScenarioSchemaError(ValueError):
    """A committed dataset violated its schema — missing key, wrong type, bad id,
    count/label invariant. A subclass of ``ValueError`` so callers can catch either.
    """


# -----------------------------------------------------------------------------
# Typed records — frozen; the loader is the only constructor.
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunbookVersion:
    """The first, authored version of a runbook fix (``version == 1``)."""

    version: int
    content: str
    authored_by: str


@dataclass(frozen=True, slots=True)
class Revision:
    """A later runbook version that supersedes the one before it.

    ``version`` strictly increases and starts at 2; ``supersedes`` equals the
    immediately prior version. This is the knowledge-update chain consolidation
    detects and ``supersede()`` records.
    """

    version: int
    supersedes: int
    content: str
    reason: str


@dataclass(frozen=True, slots=True)
class Incident:
    """One simulated SRE incident: symptoms, a runbook, and its revisions."""

    id: str
    title: str
    category: str
    symptoms: tuple[str, ...]
    fact_key: str
    runbook: RunbookVersion
    revisions: tuple[Revision, ...]

    @property
    def latest_content(self) -> str:
        """The current runbook text — the last revision, or v1 if never revised."""
        return self.revisions[-1].content if self.revisions else self.runbook.content


@dataclass(frozen=True, slots=True)
class Provenance:
    """The provenance a poison case carries onto its :class:`MemoryEvent`."""

    parent_mem: str | None
    hop_count: int
    signature: str


@dataclass(frozen=True, slots=True)
class PoisonCase:
    """One labelled write for the E4 immune evaluation.

    ``label`` is the ground truth the E4 metrics score against: an ``attack`` the
    gate should quarantine, a ``legit`` case it should not (quarantining one is a
    true false positive). ``baseline_ref`` is populated only for semantic-anomaly
    cases — the agent's normal history the anomaly is measured against.
    """

    id: str
    category: str
    label: str
    agent_id: str
    content: str
    provenance: Provenance
    expected_detector: str
    baseline_ref: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_attack(self) -> bool:
        return self.label == "attack"


@dataclass(frozen=True, slots=True)
class PoisonSuite:
    """The whole poison suite: attacks grouped by category, plus the legit set."""

    attacks: dict[str, list[PoisonCase]]
    legit: list[PoisonCase] = field(default_factory=list)

    def all(self) -> list[PoisonCase]:
        """Every case, attacks first (category order) then the legit control."""
        out: list[PoisonCase] = []
        for category in POISON_ATTACK_FILES:
            out.extend(self.attacks.get(category, []))
        out.extend(self.legit)
        return out


# -----------------------------------------------------------------------------
# Low-level validation helpers — every one raises ScenarioSchemaError on failure.
# -----------------------------------------------------------------------------


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ScenarioSchemaError(f"scenario file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScenarioSchemaError(f"{path.name}: not valid JSON ({exc})") from exc
    if not isinstance(data, list):
        raise ScenarioSchemaError(f"{path.name}: top level must be a JSON array")
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ScenarioSchemaError(f"{path.name}[{i}]: each element must be an object")
    return data


def _require(obj: dict[str, Any], key: str, typ: type | tuple[type, ...], where: str) -> Any:
    if key not in obj:
        raise ScenarioSchemaError(f"{where}: missing required key {key!r}")
    value = obj[key]
    # bool is an int subclass; reject it where a real int is required.
    if typ is int and isinstance(value, bool):
        raise ScenarioSchemaError(f"{where}.{key}: expected int, got bool")
    if not isinstance(value, typ):
        want = typ if isinstance(typ, tuple) else (typ,)
        names = "/".join(t.__name__ for t in want)
        raise ScenarioSchemaError(f"{where}.{key}: expected {names}, got {type(value).__name__}")
    return value


def _require_nonempty_str(obj: dict[str, Any], key: str, where: str) -> str:
    value = _require(obj, key, str, where)
    if not value.strip():
        raise ScenarioSchemaError(f"{where}.{key}: must be a non-empty string")
    return value


# -----------------------------------------------------------------------------
# Incident parsing
# -----------------------------------------------------------------------------


def _parse_runbook(obj: Any, where: str) -> RunbookVersion:
    if not isinstance(obj, dict):
        raise ScenarioSchemaError(f"{where}.runbook: must be an object")
    version = _require(obj, "version", int, f"{where}.runbook")
    if version != 1:
        raise ScenarioSchemaError(f"{where}.runbook.version: must be 1, got {version}")
    return RunbookVersion(
        version=version,
        content=_require_nonempty_str(obj, "content", f"{where}.runbook"),
        authored_by=_require_nonempty_str(obj, "authored_by", f"{where}.runbook"),
    )


def _parse_revisions(raw: Any, where: str) -> tuple[Revision, ...]:
    if not isinstance(raw, list) or not raw:
        raise ScenarioSchemaError(f"{where}.revisions: must be a non-empty array")
    revisions: list[Revision] = []
    expected_version = 2
    for i, obj in enumerate(raw):
        rw = f"{where}.revisions[{i}]"
        if not isinstance(obj, dict):
            raise ScenarioSchemaError(f"{rw}: must be an object")
        version = _require(obj, "version", int, rw)
        supersedes = _require(obj, "supersedes", int, rw)
        if version != expected_version:
            raise ScenarioSchemaError(
                f"{rw}.version: expected strictly increasing from 2 "
                f"(want {expected_version}, got {version})"
            )
        if supersedes != version - 1:
            raise ScenarioSchemaError(
                f"{rw}.supersedes: must equal the prior version {version - 1}, got {supersedes}"
            )
        revisions.append(
            Revision(
                version=version,
                supersedes=supersedes,
                content=_require_nonempty_str(obj, "content", rw),
                reason=_require_nonempty_str(obj, "reason", rw),
            )
        )
        expected_version += 1
    return tuple(revisions)


def _parse_incident(obj: dict[str, Any], where: str) -> Incident:
    inc_id = _require_nonempty_str(obj, "id", where)
    if not _INCIDENT_ID_RE.match(inc_id):
        raise ScenarioSchemaError(f"{where}.id: {inc_id!r} does not match ^inc-[0-9]{{4}}$")
    symptoms_raw = _require(obj, "symptoms", list, where)
    if not symptoms_raw:
        raise ScenarioSchemaError(f"{where}.symptoms: must be a non-empty array")
    symptoms: list[str] = []
    for i, s in enumerate(symptoms_raw):
        if not isinstance(s, str) or not s.strip():
            raise ScenarioSchemaError(f"{where}.symptoms[{i}]: must be a non-empty string")
        symptoms.append(s)
    fact_key = _require_nonempty_str(obj, "fact_key", where)
    if not _FACT_KEY_RE.match(fact_key):
        raise ScenarioSchemaError(f"{where}.fact_key: {fact_key!r} does not match ^runbook:.+:fix$")
    return Incident(
        id=inc_id,
        title=_require_nonempty_str(obj, "title", where),
        category=_require_nonempty_str(obj, "category", where),
        symptoms=tuple(symptoms),
        fact_key=fact_key,
        runbook=_parse_runbook(obj.get("runbook"), where),
        revisions=_parse_revisions(obj.get("revisions"), where),
    )


def load_incidents(path: Path | None = None) -> list[Incident]:
    """Load and validate an incident file (default: committed ``sample.json``).

    Enforces unique, well-formed ids and a runbook-plus-revision chain on every
    record. Raises :class:`ScenarioSchemaError` on the first violation.
    """
    path = path or COMMITTED_INCIDENTS_PATH
    raw = _read_json_array(path)
    incidents: list[Incident] = []
    seen: set[str] = set()
    for i, obj in enumerate(raw):
        incident = _parse_incident(obj, f"{path.name}[{i}]")
        if incident.id in seen:
            raise ScenarioSchemaError(f"{path.name}: duplicate incident id {incident.id!r}")
        seen.add(incident.id)
        incidents.append(incident)
    return incidents


# -----------------------------------------------------------------------------
# Poison parsing
# -----------------------------------------------------------------------------


def _parse_provenance(obj: Any, where: str) -> Provenance:
    if not isinstance(obj, dict):
        raise ScenarioSchemaError(f"{where}.provenance: must be an object")
    if "parent_mem" not in obj:
        raise ScenarioSchemaError(f"{where}.provenance: missing required key 'parent_mem'")
    parent = obj["parent_mem"]
    if parent is not None and not isinstance(parent, str):
        raise ScenarioSchemaError(f"{where}.provenance.parent_mem: must be a string or null")
    hop_count = _require(obj, "hop_count", int, f"{where}.provenance")
    if hop_count < 0:
        raise ScenarioSchemaError(f"{where}.provenance.hop_count: must be >= 0, got {hop_count}")
    # signature may be empty on purpose (a 'missing signature' bad_provenance
    # attack), so require the key and str type but not non-emptiness.
    signature = _require(obj, "signature", str, f"{where}.provenance")
    return Provenance(parent_mem=parent, hop_count=hop_count, signature=signature)


def _parse_poison_case(obj: dict[str, Any], where: str, *, expected_category: str) -> PoisonCase:
    case_id = _require_nonempty_str(obj, "id", where)
    if not _POISON_ID_RE.match(case_id):
        raise ScenarioSchemaError(f"{where}.id: {case_id!r} has an invalid poison id")
    category = _require_nonempty_str(obj, "category", where)
    if category not in _VALID_POISON_CATEGORIES:
        raise ScenarioSchemaError(f"{where}.category: {category!r} is not a valid poison category")
    if category != expected_category:
        raise ScenarioSchemaError(
            f"{where}.category: {category!r} does not match this file's category "
            f"{expected_category!r}"
        )
    # The id prefix must name the same category as the file.
    if not case_id.startswith(f"{category}-"):
        raise ScenarioSchemaError(
            f"{where}.id: prefix of {case_id!r} does not match category {category!r}"
        )
    label = _require_nonempty_str(obj, "label", where)
    if label not in _VALID_LABELS:
        raise ScenarioSchemaError(f"{where}.label: {label!r} not in {sorted(_VALID_LABELS)}")
    expected_label = "legit" if category == LEGIT_CATEGORY else "attack"
    if label != expected_label:
        raise ScenarioSchemaError(
            f"{where}.label: category {category!r} requires label {expected_label!r}, got {label!r}"
        )

    baseline_raw = obj.get("baseline_ref", [])
    if not isinstance(baseline_raw, list):
        raise ScenarioSchemaError(f"{where}.baseline_ref: must be an array")
    baseline: list[str] = []
    for i, ref in enumerate(baseline_raw):
        if not isinstance(ref, str) or not ref.strip():
            raise ScenarioSchemaError(f"{where}.baseline_ref[{i}]: must be a non-empty string")
        baseline.append(ref)
    if category == QuarantineReason.SEMANTIC_ANOMALY.value and not baseline:
        raise ScenarioSchemaError(
            f"{where}.baseline_ref: a semantic_anomaly case must carry a non-empty baseline_ref"
        )

    return PoisonCase(
        id=case_id,
        category=category,
        label=label,
        agent_id=_require_nonempty_str(obj, "agent_id", where),
        content=_require_nonempty_str(obj, "content", where),
        provenance=_parse_provenance(obj.get("provenance"), where),
        expected_detector=_require_nonempty_str(obj, "expected_detector", where),
        baseline_ref=tuple(baseline),
        notes=str(obj.get("notes", "")),
    )


def _load_poison_file(category: str, filename: str, *, min_count: int) -> list[PoisonCase]:
    path = POISON_DIR / filename
    raw = _read_json_array(path)
    cases: list[PoisonCase] = []
    seen: set[str] = set()
    for i, obj in enumerate(raw):
        case = _parse_poison_case(obj, f"{filename}[{i}]", expected_category=category)
        if case.id in seen:
            raise ScenarioSchemaError(f"{filename}: duplicate poison id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if len(cases) < min_count:
        raise ScenarioSchemaError(
            f"{filename}: expected at least {min_count} cases, found {len(cases)}"
        )
    return cases


def load_poison_suite() -> PoisonSuite:
    """Load and validate the whole poison suite from the committed files.

    Enforces the per-file minimums (>= 30 attacks per category, >= 30 legit) and
    every per-case invariant. Raises :class:`ScenarioSchemaError` on any violation.
    """
    attacks: dict[str, list[PoisonCase]] = {}
    for category, filename in POISON_ATTACK_FILES.items():
        attacks[category] = _load_poison_file(
            category, filename, min_count=MIN_ATTACKS_PER_CATEGORY
        )
    legit = _load_poison_file(LEGIT_CATEGORY, POISON_LEGIT_FILE, min_count=MIN_LEGIT_CASES)
    return PoisonSuite(attacks=attacks, legit=legit)


def load_poison_cases(category: str | None = None, label: str | None = None) -> list[PoisonCase]:
    """Flat, filtered view over the poison suite.

    ``category`` selects one file's cases; ``label`` filters to 'attack' or
    'legit'. Both default to no filter (every case).
    """
    if category is not None and category not in _VALID_POISON_CATEGORIES:
        raise ScenarioSchemaError(f"unknown poison category {category!r}")
    if label is not None and label not in _VALID_LABELS:
        raise ScenarioSchemaError(f"unknown label {label!r}")
    suite = load_poison_suite()
    cases = suite.all()
    if category is not None:
        cases = [c for c in cases if c.category == category]
    if label is not None:
        cases = [c for c in cases if c.label == label]
    return cases


# -----------------------------------------------------------------------------
# Mapping to core MemoryEvents — embeddings only when an Embedder is injected.
# -----------------------------------------------------------------------------


def _embed(texts: Iterable[str], embedder: Embedder | None) -> list[list[float] | None]:
    materialised = list(texts)
    if embedder is None:
        return [None] * len(materialised)
    return list(embedder.embed_batch(materialised))


def incident_to_events(inc: Incident, embedder: Embedder | None = None) -> list[MemoryEvent]:
    """Map an incident's runbook v1 and its revisions to episodic MemoryEvents.

    The runbook author owns every event (revisions do not carry their own author
    in the corpus). ``meta`` carries the incident id, fact_key and version so a
    consuming experiment can drive ``supersede()``/``set_canonical()``. Embeddings
    are attached only when an :class:`Embedder` is injected — never read from JSON.
    """
    contents = [inc.runbook.content, *(r.content for r in inc.revisions)]
    vectors = _embed(contents, embedder)
    author = inc.runbook.authored_by

    events: list[MemoryEvent] = []
    versions = [inc.runbook.version, *(r.version for r in inc.revisions)]
    for content, vec, version in zip(contents, vectors, versions, strict=True):
        events.append(
            MemoryEvent(
                agent_id=author,
                content=content,
                kind=MemoryKind.EPISODIC,
                embedding=vec,
                meta={
                    "incident_id": inc.id,
                    "fact_key": inc.fact_key,
                    "runbook_version": version,
                    "category": inc.category,
                },
            )
        )
    return events


def poison_to_event(case: PoisonCase, embedder: Embedder | None = None) -> MemoryEvent:
    """Map one poison case to a MemoryEvent carrying its attack provenance intact.

    ``parent_mem``/``hop_count``/``signature`` are copied verbatim — including
    values that intentionally violate runtime invariants (an over-propagated hop
    count, an empty signature): that is the attack data the immune gate must
    catch, not a schema fault. A missing signature is carried as ``None`` so the
    provenance detector's ``if not event.signature`` check fires as designed.
    """
    (vec,) = _embed([case.content], embedder)
    return MemoryEvent(
        agent_id=case.agent_id,
        content=case.content,
        kind=MemoryKind.EPISODIC,
        embedding=vec,
        parent_mem=case.provenance.parent_mem,
        signature=case.provenance.signature or None,
        hop_count=case.provenance.hop_count,
        meta={
            "poison_id": case.id,
            "poison_category": case.category,
            "label": case.label,
            "expected_detector": case.expected_detector,
        },
    )
