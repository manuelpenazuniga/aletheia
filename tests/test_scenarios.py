"""Scenario-dataset tests — schema, labels, counts, determinism. Fully offline.

Everything runs against the committed JSON and the seeded
:class:`~core.embeddings.DeterministicEmbedder`: no network, no Bedrock, no live
DB. These tests are the guard that a committed dataset can never silently drift
from its schema or its generator (the project plan §3.1 / §8.4).
"""

from __future__ import annotations

import json
import re

import pytest

from core.config import DEFAULT_EMBEDDING_DIM
from core.embeddings import DeterministicEmbedder
from core.immune import detect_injection_pattern
from core.models import MemoryEvent, MemoryKind, QuarantineReason
from scenarios.generate_incidents import dumps_incidents, generate_incidents
from scenarios.loader import (
    COMMITTED_INCIDENTS_PATH,
    MIN_ATTACKS_PER_CATEGORY,
    MIN_LEGIT_CASES,
    POISON_ATTACK_FILES,
    POISON_DIR,
    ScenarioSchemaError,
    incident_to_events,
    load_incidents,
    load_poison_cases,
    load_poison_suite,
    poison_to_event,
)
from scenarios.model_client import OfflineTemplateModel

_INCIDENT_ID_RE = re.compile(r"^inc-[0-9]{4}$")
_FACT_KEY_RE = re.compile(r"^runbook:.+:fix$")


# ---------------------------------------------------------------------------
# Smoke: every committed dataset the loader owns parses without raising.
# ---------------------------------------------------------------------------


def test_loader_reads_every_committed_dataset():
    incidents = load_incidents()
    suite = load_poison_suite()
    assert incidents
    assert suite.all()


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------


def test_incident_ids_unique_and_well_formed():
    incidents = load_incidents()
    ids = [inc.id for inc in incidents]
    assert len(ids) == len(set(ids)), "incident ids must be unique"
    for inc in incidents:
        assert _INCIDENT_ID_RE.match(inc.id)
        assert inc.symptoms, "every incident needs at least one symptom"
        assert all(s.strip() for s in inc.symptoms)
        assert _FACT_KEY_RE.match(inc.fact_key)


def test_incident_runbook_and_revision_chain_is_well_formed():
    for inc in load_incidents():
        assert inc.runbook.version == 1
        assert inc.revisions, "every incident carries at least one runbook revision"
        expected = 2
        for rev in inc.revisions:
            assert rev.version == expected, "revision versions strictly increase from 2"
            assert rev.supersedes == rev.version - 1, "each revision supersedes the prior version"
            assert rev.content.strip()
            assert rev.reason.strip()
            expected += 1


# ---------------------------------------------------------------------------
# Poison suite — counts, labels, categories.
# ---------------------------------------------------------------------------


def test_poison_suite_meets_minimum_counts():
    suite = load_poison_suite()
    for category in POISON_ATTACK_FILES:
        assert len(suite.attacks[category]) >= MIN_ATTACKS_PER_CATEGORY
    assert len(suite.legit) >= MIN_LEGIT_CASES


def test_attack_files_category_matches_quarantine_reason():
    """Each attack file's category is a QuarantineReason value == its basename."""
    valid_reasons = {r.value for r in QuarantineReason}
    suite = load_poison_suite()
    for category, cases in suite.attacks.items():
        assert category in valid_reasons
        for case in cases:
            assert case.category == category
            assert case.label == "attack"
            assert case.content.strip()


def test_legit_cases_all_labelled_legit():
    for case in load_poison_cases(category="legit"):
        assert case.label == "legit"
        assert case.content.strip()


def test_poison_ids_unique_per_file_and_prefixed_by_category():
    suite = load_poison_suite()
    for category, cases in suite.attacks.items():
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))
        assert all(c.id.startswith(f"{category}-") for c in cases)
    legit_ids = [c.id for c in suite.legit]
    assert len(legit_ids) == len(set(legit_ids))
    assert all(c.id.startswith("legit-") for c in suite.legit)


def test_every_label_is_in_the_allowed_domain():
    for case in load_poison_cases():
        assert case.label in {"attack", "legit"}


def test_semantic_anomaly_cases_carry_a_baseline_ref():
    for case in load_poison_cases(category="semantic_anomaly"):
        assert case.baseline_ref, "a semantic_anomaly case needs a baseline history"
        assert all(ref.strip() for ref in case.baseline_ref)


def test_injection_cases_actually_trip_the_immune_scanner():
    """Ground-truth check: every injection attack matches the real detector."""
    for case in load_poison_cases(category="injection_pattern"):
        assert detect_injection_pattern(case.content) is not None, case.content


#: Domain-agnostic stop-words removed before the off-topic lexical check.
_STOP = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "on",
    "in",
    "at",
    "and",
    "for",
    "after",
    "before",
    "with",
    "is",
    "was",
    "no",
    "it",
    "its",
    "this",
    "that",
    "we",
}


def _content_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOP


def test_semantic_anomaly_attacks_are_lexically_off_topic():
    """Ground-truth check, offline and embedder-independent: every semantic_anomaly
    attack must share NO domain vocabulary with its baseline history.

    The absolute anomaly *distance* depends on the embedder — the seeded offline
    embedder does not separate topics as sharply as Titan does at the production
    threshold, so validating the detector's verdict offline would require a fudged
    threshold (dishonest). What we CAN validate offline, and what the label
    actually claims, is that the attack is off-topic: disjoint content tokens. The
    real-detector verdict is validated against Bedrock in
    ``test_semantic_anomaly_attacks_trip_the_real_detector_on_bedrock`` (E4)."""
    for case in load_poison_cases(category="semantic_anomaly"):
        attack = _content_tokens(case.content)
        baseline = set().union(*(_content_tokens(ref) for ref in case.baseline_ref))
        overlap = len(attack & baseline) / max(1, len(attack))
        assert overlap == 0.0, (case.id, attack & baseline)


def _has_aws_credentials() -> bool:
    try:
        import botocore.session

        return botocore.session.get_session().get_credentials() is not None
    except Exception:
        return False


@pytest.mark.aws
@pytest.mark.skipif(not _has_aws_credentials(), reason="no AWS credentials — Bedrock unavailable")
def test_semantic_anomaly_attacks_trip_the_real_detector_on_bedrock():
    """The true ground-truth check for semantic_anomaly labels: with REAL Titan
    embeddings and the production threshold, each attack must trip the detector.

    Credential-gated (skips without AWS), like the CockroachDB integration tests.
    Bounded to a stratified sample to keep Bedrock cost small."""
    from adapters.bedrock_embedder import BedrockEmbedder
    from adapters.memory_inmem import InMemoryAdapter
    from core.config import AletheiaConfig
    from core.immune import ImmuneSystem

    dim = 1024
    embedder = BedrockEmbedder(dim=dim)
    config = AletheiaConfig(embedding_dim=dim)
    cases = load_poison_cases(category="semantic_anomaly")[:8]  # cost-bounded spot-check

    for case in cases:
        adapter = InMemoryAdapter(embedding_dim=dim)
        adapter.register_agent(case.agent_id, "sre", "hash")
        for i, ref in enumerate(case.baseline_ref):
            adapter.write_episode(
                case.agent_id,
                MemoryEvent(
                    agent_id=case.agent_id,
                    content=ref,
                    embedding=embedder.embed(ref),
                    mem_id=f"base-{i}",
                ),
            )
        verdict = ImmuneSystem(adapter, embedder, config).inspect(poison_to_event(case))
        assert not verdict.accepted, case.content
        assert verdict.reason is QuarantineReason.SEMANTIC_ANOMALY, (case.id, verdict.reason)


def test_load_poison_cases_filters_by_label():
    attacks = load_poison_cases(label="attack")
    legit = load_poison_cases(label="legit")
    assert attacks and legit
    assert all(c.label == "attack" for c in attacks)
    assert all(c.label == "legit" for c in legit)


# ---------------------------------------------------------------------------
# Committed JSON never carries embeddings.
# ---------------------------------------------------------------------------


def test_committed_json_contains_no_embeddings():
    paths = [COMMITTED_INCIDENTS_PATH, *(POISON_DIR / f for f in POISON_ATTACK_FILES.values())]
    paths.append(POISON_DIR / "legit_unusual.json")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "embedding" not in text, f"{path.name} must not persist vectors"
        # structural guard, not just a substring match
        for obj in json.loads(text):
            assert "embedding" not in obj


# ---------------------------------------------------------------------------
# Mapping to core MemoryEvents.
# ---------------------------------------------------------------------------


def test_incident_to_events_yields_valid_memory_events():
    embedder = DeterministicEmbedder()
    inc = load_incidents()[0]
    events = incident_to_events(inc, embedder)
    assert len(events) == 1 + len(inc.revisions)  # runbook v1 + each revision
    for ev in events:
        assert isinstance(ev, MemoryEvent)
        assert ev.kind is MemoryKind.EPISODIC
        assert ev.cost_tokens and ev.cost_tokens > 0
        assert ev.embedding is not None
        assert len(ev.embedding) == DEFAULT_EMBEDDING_DIM
        assert ev.meta["incident_id"] == inc.id
        assert ev.meta["fact_key"] == inc.fact_key


def test_incident_to_events_without_embedder_has_no_vectors():
    inc = load_incidents()[0]
    for ev in incident_to_events(inc):
        assert ev.embedding is None


def test_poison_to_event_carries_attack_provenance_intact():
    case = load_poison_cases(category="bad_provenance")[0]
    ev = poison_to_event(case)
    assert ev.agent_id == case.agent_id
    assert ev.content == case.content
    assert ev.hop_count == case.provenance.hop_count
    assert ev.parent_mem == case.provenance.parent_mem
    # an empty signature in the dataset becomes None so the provenance detector fires
    expected_sig = case.provenance.signature or None
    assert ev.signature == expected_sig


def test_poison_to_event_preserves_over_propagated_hop_count():
    """An hop_count past gossip_max_hops is attack data, not a construction error."""
    hoppy = [c for c in load_poison_cases(category="bad_provenance") if c.provenance.hop_count > 3]
    assert hoppy, "the suite should include over-propagated provenance attacks"
    ev = poison_to_event(hoppy[0])
    assert ev.hop_count > 3  # MemoryEvent accepts it; only the immune gate rejects it


# ---------------------------------------------------------------------------
# Generator determinism + no-drift.
# ---------------------------------------------------------------------------


def test_offline_model_is_deterministic_across_instances():
    a = OfflineTemplateModel()
    b = OfflineTemplateModel()
    for prompt in ("incident one", "incident two", "a totally different prompt"):
        assert a.complete(prompt, seed=0) == b.complete(prompt, seed=0)
    # a different seed changes the output (entropy is seed-driven)
    assert a.complete("incident one", seed=0) != a.complete("incident one", seed=1)


def test_committed_sample_matches_offline_generator_exactly():
    """Regenerating (offline, seed 0, n 10) reproduces the committed sample byte-for-byte."""
    regenerated = dumps_incidents(generate_incidents(10, seed=0, client=OfflineTemplateModel()))
    committed = COMMITTED_INCIDENTS_PATH.read_text(encoding="utf-8")
    assert regenerated == committed, "sample.json has drifted from its generator"


class _MalformedModel:
    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> str:
        return "this is not json"


def test_generate_rejects_malformed_model_output():
    with pytest.raises(ScenarioSchemaError):
        generate_incidents(1, seed=0, client=_MalformedModel())


class _SchemaViolatingModel:
    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> str:
        # valid JSON, but a runbook with no revisions violates the schema
        return json.dumps(
            {
                "title": "x",
                "category": "db-latency",
                "symptoms": ["s"],
                "fact_key": "runbook:db-latency:fix",
                "runbook": {"version": 1, "content": "c", "authored_by": "sre-01"},
                "revisions": [],
            }
        )


def test_generate_rejects_schema_violating_model_output():
    with pytest.raises(ScenarioSchemaError):
        generate_incidents(1, seed=0, client=_SchemaViolatingModel())
