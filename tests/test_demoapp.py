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
def client() -> TestClient:
    # No injection: create_app builds the seeded offline world itself — exactly what
    # `python -m demoapp` serves, and correctly reported as mode "offline-demo".
    return TestClient(create_app())


# --------------------------------------------------------------------- health
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "mode": "offline-demo"}


def test_injected_adapter_reports_live_mode(world):
    # When a caller injects an adapter+embedder (the production wiring), the app
    # reports mode "live" — the honest signal that data is not the seeded demo.
    c = TestClient(create_app(world.adapter, world.embedder))
    assert c.get("/healthz").json()["mode"] == "live"


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Aletheia" in r.text


# ------------------------------------------------------------------- overview
def test_overview_has_real_fleet(client):
    d = client.get("/api/overview").json()
    assert d["mode"] == "offline-demo"
    t = d["totals"]
    assert t["memories"] > 0
    assert t["active"] > 0
    assert t["archived"] > 0  # forgetting ran
    assert t["canonical_facts"] > 0  # consolidation ran
    assert t["quarantined"] > 0  # immune system ran
    # More than one SRE author, so the fleet view is genuinely a fleet.
    sre_agents = [a for a in d["agents"] if a["agent_id"].startswith("sre-")]
    assert len(sre_agents) >= 2
    # Each agent's total covers every status (incl. archived), and the per-agent
    # totals reconcile with the fleet-wide memory count — no rows silently dropped.
    for a in d["agents"]:
        assert a["total"] == a["active"] + a["superseded"] + a["quarantined"] + a["archived"]
    assert sum(a["total"] for a in d["agents"]) == t["memories"]


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
    # Backend is derived from the recorded field, and N from n_agents — not guessed.
    assert all(r["backend"] in {"naive", "cockroachdb"} for r in d["r1"])
    assert all(isinstance(r["n"], int) for r in d["r1"])
    # R2: the kill-node arm lost zero memories.
    kill = [r for r in d["r2"] if r.get("event") == "kill_node"]
    assert kill and all(r["memories_lost"] == 0 for r in kill)


def test_results_integrity_filters():
    # A dry-run or non-authoritative row must never be treated as a real result,
    # and the backend must come from the recorded field, not the arm string.
    from demoapp.app import _authoritative, _backend_of

    assert _authoritative({"metrics": {"authoritative": True, "dry_run": False}}) is True
    assert _authoritative({"metrics": {"authoritative": True, "dry_run": True}}) is False
    assert _authoritative({"metrics": {"authoritative": False, "dry_run": False}}) is False
    assert _authoritative({"metrics": {}}) is False  # fail closed when unflagged
    assert _backend_of({"backend": "crdb_serializable"}, "misleading_naive_arm") == "cockroachdb"
    assert _backend_of({"backend": "naive_nontxn"}, "arm") == "naive"
    assert _backend_of({}, "unlabeled") == "unknown"


def test_default_app_builds_offline():
    # create_app() with no injection builds the demo world itself.
    c = TestClient(create_app())
    assert c.get("/healthz").status_code == 200
    assert c.get("/api/overview").json()["totals"]["active"] > 0


# --------------------------------------------------------------- kill-switch
def test_ablation_wall_diverges(client):
    panels = client.get("/api/ablation").json()["panels"]
    by = {p["label"]: p for p in panels}
    assert set(by) == {"full", "no-consolidation", "no-immune", "no-forgetting"}
    full = by["full"]["metrics"]
    # full is the clean reference.
    assert full["stale_active"] == 0
    assert full["poison_active"] == 0
    # Each ablation degrades its own target, measurably, vs full.
    assert by["no-consolidation"]["metrics"]["canonical_facts"] < full["canonical_facts"]
    assert by["no-immune"]["metrics"]["poison_active"] > full["poison_active"]
    assert by["no-forgetting"]["metrics"]["active_cost_tokens"] > full["active_cost_tokens"]


def test_ablation_and_killswitch_labelled_simulated(client):
    # Both counterfactual endpoints must disclose they are seeded offline sims, so
    # their numbers are never mistaken for live fleet state.
    assert client.get("/api/ablation").json()["source"] == "seeded-offline"
    ks = client.post("/api/killswitch", json={}).json()
    assert ks["simulated"] is True and ks["source"] == "seeded-offline"


def test_killswitch_all_on_matches_baseline(client):
    d = client.post("/api/killswitch", json={}).json()  # defaults = all on
    assert d["flags"] == {
        "enable_consolidation": True,
        "enable_forgetting": True,
        "enable_gossip": True,
        "enable_immune": True,
    }
    assert d["metrics"] == d["baseline"]  # all-on IS the baseline


def test_killswitch_disable_immune_contaminates(client):
    d = client.post("/api/killswitch", json={"enable_immune": False}).json()
    assert d["flags"]["enable_immune"] is False
    # Poison now sits in the active fleet; the baseline had none.
    assert d["metrics"]["poison_active"] > 0
    assert d["baseline"]["poison_active"] == 0
    assert d["metrics"]["quarantined"] == 0


def test_killswitch_disable_forgetting_grows_footprint(client):
    d = client.post("/api/killswitch", json={"enable_forgetting": False}).json()
    assert d["metrics"]["active_cost_tokens"] > d["baseline"]["active_cost_tokens"]
    assert d["metrics"]["archived"] == 0


# ----------------------------------------------------------- launch attack
def test_attack_categories_offered(client):
    d = client.get("/api/attack/categories").json()
    # Only the offline-reliable families are offered; semantic_anomaly is excluded.
    assert d["categories"] == ["injection_pattern", "bad_provenance"]
    assert d["token_required"] is False  # no ALETHEIA_DEMO_TOKEN set in tests


def test_attack_is_caught_and_hits_the_feed(client):
    before = len(client.get("/api/quarantine").json()["quarantine"])
    r = client.post("/api/attack", json={"category": "injection_pattern"})
    assert r.status_code == 200
    d = r.json()
    assert d["detected"] is True
    assert d["detector"] and d["reason"]
    assert d["quarantined_mem_id"]
    # The caught attack really lands in the quarantine feed (not staged).
    after = client.get("/api/quarantine").json()["quarantine"]
    assert len(after) == before + 1
    assert any(q["mem_id"] == d["quarantined_mem_id"] for q in after)


def test_attack_bad_provenance_also_caught(client):
    d = client.post("/api/attack", json={"category": "bad_provenance"}).json()
    assert d["detected"] is True


def test_attack_rejects_unknown_category(client):
    assert client.post("/api/attack", json={"category": "nope"}).status_code == 422


def test_attack_token_gate(world, monkeypatch):
    # With a demo token configured, an attack without the right token is refused.
    monkeypatch.setenv("ALETHEIA_DEMO_TOKEN", "s3cr3t")
    c = TestClient(create_app(world.adapter, world.embedder))
    assert c.get("/api/attack/categories").json()["token_required"] is True
    assert c.post("/api/attack", json={"category": "injection_pattern"}).status_code == 403
    assert (
        c.post("/api/attack", json={"category": "injection_pattern", "token": "wrong"}).status_code
        == 403
    )
    ok = c.post("/api/attack", json={"category": "injection_pattern", "token": "s3cr3t"})
    assert ok.status_code == 200 and ok.json()["detected"] is True


# ------------------------------------------------------------- determinism
def test_demo_world_is_deterministic():
    # The integrity policy forbids results that wander between runs. The demo world
    # must score identically every build (strict prune order, no random tie-break).
    from demoapp.data import build_demo_world, world_metrics

    runs = [world_metrics(build_demo_world().adapter) for _ in range(5)]
    assert all(r == runs[0] for r in runs), runs


def test_canonical_content_is_deterministic_and_latest():
    # A subtler integrity trap: consolidation picks the newest by (created_at,
    # mem_id); if versions tie on created_at the random mem_id could crown a STALE
    # version, and the demo would show an obsolete runbook as current. Deterministic
    # version-ordered timestamps must make the true latest win, every build.
    from collections import defaultdict

    from demoapp.data import build_demo_world

    contents = set()
    for _ in range(8):
        world = build_demo_world()
        canon = tuple(sorted((k, v.content) for k, v in world.adapter._canonical.items()))
        contents.add(canon)
    assert len(contents) == 1, "canonical content must not vary between builds"

    # And the surviving canonical version is the highest for its fact_key.
    world = build_demo_world()
    max_v: dict[str, int] = defaultdict(int)
    active_v: dict[str, int] = {}
    for m in world.adapter.list_memories(status=None):
        if isinstance(m.meta, dict) and m.meta.get("fact_key"):
            key = m.meta["fact_key"]
            v = int(m.meta.get("runbook_version") or 0)
            max_v[key] = max(max_v[key], v)
            if str(m.status) == "active":
                active_v[key] = v
    assert active_v and all(active_v[k] == max_v[k] for k in active_v)


# ------------------------------------------------------------- input bounds
def test_memories_limit_is_bounded(client):
    assert client.get("/api/memories?limit=0").status_code == 422  # ge=1
    assert client.get("/api/memories?limit=100000").status_code == 422  # le cap


def test_search_k_is_clamped_and_query_bounded(client):
    # A huge k is clamped, not honoured unboundedly.
    r = client.post("/api/search", json={"query": "latency", "k": 999999})
    assert r.status_code == 200
    assert len(r.json()["hits"]) <= 50
    # An over-long query is rejected rather than embedded.
    assert client.post("/api/search", json={"query": "x" * 10000}).status_code == 422
