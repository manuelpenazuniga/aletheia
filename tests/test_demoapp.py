"""Demo dashboard tests: TestClient over the seeded offline demo world.

The demo app is a read-only view over a real memory layer. These tests assert the
HTTP surface serves real, non-empty content built by the actual core cycles
(consolidation produced canonical facts with version history; the immune system
produced a quarantine feed) — not a mock — and that it stays honest: every
results number traces to committed run rows, no endpoint fabricates data.

Everything is offline and deterministic (InMemoryAdapter + DeterministicEmbedder);
no CockroachDB, no Bedrock.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from demoapp.app import create_app
from demoapp.data import build_demo_world


@pytest.fixture(scope="module")
def world():
    return build_demo_world()


@pytest.fixture(scope="module")
def client(world) -> TestClient:
    return TestClient(create_app(world.adapter, world.embedder))


# --------------------------------------------------------------------- health
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Aletheia" in r.text


# ------------------------------------------------------------------- overview
def test_overview_has_real_fleet(client):
    d = client.get("/api/overview").json()
    t = d["totals"]
    assert t["memories"] > 0
    assert t["active"] > 0
    assert t["canonical_facts"] > 0  # consolidation ran
    assert t["quarantined"] > 0  # immune system ran
    # More than one SRE author, so the fleet view is genuinely a fleet.
    sre_agents = [a for a in d["agents"] if a["agent_id"].startswith("sre-")]
    assert len(sre_agents) >= 2
    assert all(a["total"] == a["active"] + a["superseded"] + a["quarantined"] for a in d["agents"])


# -------------------------------------------------------------------- memory
def test_memories_listed(client):
    d = client.get("/api/memories?limit=10").json()
    assert 0 < len(d["memories"]) <= 10
    m = d["memories"][0]
    assert {"mem_id", "agent_id", "content", "status"} <= m.keys()


def test_search_returns_ranked_hits(client):
    r = client.post("/api/search", json={"query": "database latency runbook", "k": 5})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert hits, "semantic search should find seeded runbook memories"
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)  # ranked best-first


def test_search_rejects_empty_query(client):
    assert client.post("/api/search", json={"query": "   "}).status_code == 422


# ---------------------------------------------------------------- provenance
def test_provenance_chain_for_real_memory(client):
    mem_id = client.get("/api/memories?limit=1").json()["memories"][0]["mem_id"]
    d = client.get(f"/api/provenance/{mem_id}").json()
    assert d["mem_id"] == mem_id
    assert len(d["chain"]) >= 1


def test_provenance_unknown_memory_404(client):
    r = client.get("/api/provenance/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


# ----------------------------------------------------------------- canonical
def test_canonical_facts_have_version_history(client):
    facts = client.get("/api/canonical").json()["canonical_facts"]
    assert facts, "consolidation should have promoted canonical facts"
    key = facts[0]["fact_key"]
    hist = client.get(f"/api/canonical/{key}/history").json()
    assert hist["fact_key"] == key
    statuses = [v["status"] for v in hist["timeline"]]
    # A knowledge-update: at least one superseded version and exactly one current.
    assert "active" in statuses
    assert statuses.count("active") == 1
    assert any(s == "superseded" for s in statuses)
    # Timeline is oldest-first; the current version is last.
    assert hist["timeline"][-1]["status"] == "active"


def test_canonical_history_unknown_key_404(client):
    assert client.get("/api/canonical/runbook:does-not-exist/history").status_code == 404


# ------------------------------------------------------------------- immune
def test_quarantine_feed_is_real(client):
    feed = client.get("/api/quarantine").json()["quarantine"]
    assert feed, "the immune system should have quarantined seeded poison"
    rec = feed[0]
    assert rec["reason"]
    assert rec["detector"]


# ------------------------------------------------------------------ results
def test_results_trace_to_committed_runs(client):
    d = client.get("/api/results").json()
    assert d["available"] is True
    # R1: CockroachDB SERIALIZABLE is inconsistency-free; naive is not.
    crdb = [r for r in d["r1"] if r["backend"] == "cockroachdb"]
    naive = [r for r in d["r1"] if r["backend"] == "naive"]
    assert crdb and all(r["inconsistencies_per_1000"] == 0 for r in crdb)
    assert naive and all(r["inconsistencies_per_1000"] > 0 for r in naive)
    # R2: the kill-node arm lost zero memories.
    kill = [r for r in d["r2"] if r.get("event") == "kill_node"]
    assert kill and all(r["memories_lost"] == 0 for r in kill)


def test_default_app_builds_offline():
    # create_app() with no injection builds the demo world itself.
    c = TestClient(create_app())
    assert c.get("/healthz").status_code == 200
    assert c.get("/api/overview").json()["totals"]["active"] > 0
