"""Domain models: validation, defaults and the invariants they encode."""

from __future__ import annotations

import pytest

from core.models import (
    CanonicalFact,
    MemoryEvent,
    MemoryKind,
    MemoryStats,
    MemoryStatus,
    ProvenanceLink,
    QuarantineReason,
    estimate_tokens,
)


def test_memory_event_defaults():
    event = MemoryEvent(agent_id="sre-1", content="disk full on node 3")
    assert event.kind is MemoryKind.EPISODIC
    assert event.status is MemoryStatus.ACTIVE
    assert event.mem_id is None
    assert event.hop_count == 0
    assert event.is_direct_observation is True


def test_cost_tokens_is_derived_from_content_when_absent():
    event = MemoryEvent(agent_id="sre-1", content="x" * 400)
    assert event.cost_tokens == estimate_tokens("x" * 400) == 100


def test_explicit_cost_tokens_is_preserved():
    event = MemoryEvent(agent_id="sre-1", content="short", cost_tokens=99)
    assert event.cost_tokens == 99


def test_string_kind_and_status_are_coerced_to_enums():
    """Payloads arriving as JSON from the ingest service must land as enums."""
    event = MemoryEvent(agent_id="a", content="c", kind="semantic", status="active")
    assert event.kind is MemoryKind.SEMANTIC
    assert event.status is MemoryStatus.ACTIVE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"agent_id": "", "content": "c"},
        {"agent_id": "a", "content": ""},
        {"agent_id": "a", "content": "c", "importance": 1.5},
        {"agent_id": "a", "content": "c", "importance": -0.1},
        {"agent_id": "a", "content": "c", "hop_count": -1},
    ],
)
def test_invalid_memory_events_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MemoryEvent(**kwargs)


def test_gossiped_memory_is_not_a_direct_observation():
    event = MemoryEvent(agent_id="a", content="c", parent_mem="mem-1", hop_count=2)
    assert event.is_direct_observation is False


def test_estimate_tokens_is_deterministic_and_non_zero_for_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("same") == estimate_tokens("same")


def test_provenance_link_root_detection():
    assert ProvenanceLink(mem_id="m1", agent_id="a", signature="s").is_root is True
    assert (
        ProvenanceLink(mem_id="m2", agent_id="a", signature="s", parent_mem="m1").is_root is False
    )


def test_canonical_fact_starts_at_version_one():
    assert CanonicalFact(fact_key="runbook:x:fix", content="restart").version == 1


def test_enum_values_match_the_schema_strings():
    """These strings are written into CockroachDB; they are part of the schema."""
    assert MemoryStatus.ACTIVE == "active"
    assert MemoryStatus.SUPERSEDED == "superseded"
    assert MemoryStatus.QUARANTINED == "quarantined"
    assert MemoryStatus.ARCHIVED == "archived"
    assert MemoryKind.CANONICAL_REF == "canonical_ref"
    assert QuarantineReason.BAD_PROVENANCE == "bad_provenance"


def test_memory_stats_footprint():
    stats = MemoryStats(
        agent_id=None,
        total_memories=2,
        active=2,
        superseded=0,
        quarantined=0,
        archived=0,
        total_cost_tokens=1_048_576 // 4,
    )
    assert stats.footprint_mb == 1.0
