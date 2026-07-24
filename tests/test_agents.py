"""Offline, deterministic tests for the SRE agent fleet (Unit D).

Everything here runs with no network: a :class:`DeterministicEmbedder`, an
:class:`InMemoryAdapter`, and a :class:`ScriptedModel`. The immune gate under test
is the real, portable :class:`core.immune.ImmuneSystem` (also offline) for the
structural/regex attacks, plus a tiny rejecting double where the point is flag
gating rather than detection.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from adapters.memory_inmem import InMemoryAdapter
from agents.adversary import (
    ATTACK_BAD_PROVENANCE,
    ATTACK_CANONICAL_REWRITE,
    ATTACK_CONTENT_INJECTION,
    NONEXISTENT_PARENT,
    AdversaryAgent,
)
from agents.fleet import make_local_adversary, make_local_sre_agent, run_fleet
from agents.model_client import ModelRequest, ScriptedModel
from agents.read_client import LocalMemoryReader
from agents.signing import HmacSigner, derive_agent_token, sign_message
from agents.types import Incident
from agents.write_client import LocalMemoryWriter, PoisonRejected
from core.config import AletheiaConfig
from core.embeddings import DeterministicEmbedder
from core.immune import ImmuneSystem
from core.memory import MemoryService
from core.models import MemoryKind, MemoryStatus

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DIM = 8
SECRET = "test-secret"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def cfg(**overrides) -> AletheiaConfig:
    base = {"embedding_dim": DIM}
    base.update(overrides)
    return AletheiaConfig().with_overrides(**base)


def make_counter():
    counter = itertools.count(1)
    lock = threading.Lock()

    def _next_id() -> str:
        with lock:
            return f"m-{next(counter):04d}"

    return _next_id


def build_service(config: AletheiaConfig, *, id_factory=None):
    adapter = InMemoryAdapter(embedding_dim=DIM, id_factory=id_factory)
    embedder = DeterministicEmbedder(dim=DIM, seed=0)
    return MemoryService(adapter, embedder, config), adapter, embedder


def diagnosis_responder(req: ModelRequest) -> str:
    """A deterministic JSON diagnosis keyed on the incident id (req.tag)."""
    return json.dumps(
        {
            "summary": f"diagnosed {req.tag}",
            "action": "mitigate and monitor",
            "learned_fact": f"Runbook for {req.tag}: rotate the pool and restart the sidecar.",
            "importance": 0.6,
            "kind": "episodic",
        }
    )


def make_incidents(n: int) -> list[Incident]:
    return [
        Incident(
            incident_id=f"inc-{i:02d}",
            title=f"Incident {i}: service degraded",
            description="a production service is degraded",
            signals=[f"error signal {i}", "latency climbing"],
            meta={"scenario": "unit"},
        )
        for i in range(n)
    ]


class RejectAll:
    """A minimal :class:`ImmuneGate` double that rejects every write."""

    def __init__(self, reason: str = "canonical_rewrite") -> None:
        self._reason = reason

    def inspect(self, event):
        return SimpleNamespace(
            accepted=False, reason=self._reason, detector="test_double", payload={}
        )


# -----------------------------------------------------------------------------
# ScriptedModel
# -----------------------------------------------------------------------------


def test_scripted_model_routes_by_tag_and_is_deterministic():
    m1 = ScriptedModel({"a": "resp-a"}, default="fallback")
    m2 = ScriptedModel({"a": "resp-a"}, default="fallback")
    req = ModelRequest(system="s", messages=[{"role": "user", "content": "hi"}], tag="a")
    assert m1.complete(req).text == "resp-a"
    assert m2.complete(req).text == "resp-a"

    unknown = ModelRequest(system="s", messages=[{"role": "user", "content": "hi"}], tag="z")
    assert m1.complete(unknown).text == "fallback"


def test_scripted_model_hash_fallback_is_stable_and_nonjson():
    req = ModelRequest(system="sys", messages=[{"role": "user", "content": "x"}], tag=None)
    t1 = ScriptedModel().complete(req).text
    t2 = ScriptedModel().complete(req).text
    assert t1 == t2
    assert t1.startswith("UNSCRIPTED:")
    # Different prompt -> different fallback (a function of system+messages).
    other = ModelRequest(system="sys", messages=[{"role": "user", "content": "y"}], tag=None)
    assert ScriptedModel().complete(other).text != t1


# -----------------------------------------------------------------------------
# Read seam
# -----------------------------------------------------------------------------


def test_local_reader_delegates_budget_and_exclusion():
    config = cfg()
    svc, adapter, _ = build_service(config)
    adapter.register_agent("sre-1", "sre", "h")
    m1 = svc.remember("sre-1", "database latency spike on the primary")
    m2 = svc.remember("sre-1", "database latency spike on the primary again worse")
    adapter.supersede(m1, m2)

    reader = LocalMemoryReader(svc)
    hits = reader.recall("sre-1", "database latency")
    ids = {h.mem_id for h in hits}
    assert m2 in ids
    assert m1 not in ids  # superseded is excluded from retrieval

    tiny = reader.recall("sre-1", "database latency", budget_tokens=1)
    assert len(tiny) <= len(hits)  # budget truncation is inherited from the service


# -----------------------------------------------------------------------------
# SRE loop
# -----------------------------------------------------------------------------


def test_sre_cold_start_writes_direct_observation():
    config = cfg()
    svc, adapter, _ = build_service(config)
    agent = make_local_sre_agent(
        "sre-01", svc, ScriptedModel(diagnosis_responder), config, secret=SECRET
    )
    agent.ensure_registered()

    inc = make_incidents(1)[0]
    step = agent.handle_incident(inc)

    assert step.rejected is False
    assert step.mem_id is not None
    assert step.recalled_ids == []  # nothing to recall on the first incident
    assert step.diagnosis is not None

    stored = adapter.get_memory(step.mem_id)
    assert stored.kind == MemoryKind.EPISODIC
    assert stored.meta["incident_id"] == inc.incident_id
    assert stored.meta["arm"] == config.ablation_name

    (link, *_rest) = adapter.provenance_chain(step.mem_id)
    assert link.parent_mem is None and link.hop_count == 0
    assert link.signature  # signed by the writer's Signer
    assert adapter.stats(None).active == 1


def test_cross_agent_institutional_memory():
    config = cfg()
    svc, _adapter, _ = build_service(config)
    model = ScriptedModel(diagnosis_responder)
    a = make_local_sre_agent("sre-01", svc, model, config, secret=SECRET)
    b = make_local_sre_agent("sre-02", svc, model, config, secret=SECRET)
    a.ensure_registered()
    b.ensure_registered()

    step_a = a.handle_incident(
        Incident("inc-01", "DB latency", "primary db slow", ["p99 high"], {})
    )
    step_b = b.handle_incident(
        Incident("inc-02", "DB latency recurs", "primary db slow", ["p99 high"], {})
    )
    # B recalls A's memory: read and write seams share one backing store.
    assert step_a.mem_id in step_b.recalled_ids


def test_model_parse_fallback_writes_raw_text():
    config = cfg()
    svc, adapter, _ = build_service(config)
    model = ScriptedModel(default="this is not json at all {oops")
    agent = make_local_sre_agent("sre-01", svc, model, config, secret=SECRET)
    agent.ensure_registered()

    step = agent.handle_incident(make_incidents(1)[0])
    assert step.mem_id is not None  # never crashes on bad model output
    stored = adapter.get_memory(step.mem_id)
    assert stored.content == "this is not json at all {oops"
    assert step.diagnosis.learned_fact == "this is not json at all {oops"


# -----------------------------------------------------------------------------
# Fleet runner
# -----------------------------------------------------------------------------


def test_fleet_end_to_end_resolves_incidents_with_audit():
    config = cfg()
    svc, adapter, _ = build_service(config)
    model = ScriptedModel(diagnosis_responder)
    agents = [
        make_local_sre_agent(f"sre-{i:02d}", svc, model, config, secret=SECRET) for i in range(5)
    ]
    incidents = make_incidents(12)

    result = run_fleet(agents, incidents)

    assert result.writes == 12
    assert result.rejects == 0
    assert result.recalls == 12
    assert all(s.diagnosis is not None and s.mem_id is not None for s in result.steps)
    assert adapter.stats(None).active == 12

    # Two audit records per incident, in strictly monotonic seq order.
    assert len(result.audit) == 24
    assert [r.seq for r in result.audit] == list(range(1, 25))
    ops = [r.op for r in result.audit]
    assert ops[0::2] == ["recall"] * 12
    assert ops[1::2] == ["remember"] * 12


def test_fleet_broadcast_assignment():
    config = cfg()
    svc, adapter, _ = build_service(config)
    model = ScriptedModel(diagnosis_responder)
    agents = [
        make_local_sre_agent(f"sre-{i:02d}", svc, model, config, secret=SECRET) for i in range(3)
    ]
    result = run_fleet(agents, make_incidents(4), assignment="broadcast")
    assert result.writes == 12  # every agent handles every incident
    assert adapter.stats(None).active == 12


def test_fleet_run_is_reproducible_under_the_same_seed():
    def run_once():
        config = cfg()
        svc, _adapter, _ = build_service(config, id_factory=make_counter())
        model = ScriptedModel(diagnosis_responder)
        agents = [
            make_local_sre_agent(f"sre-{i:02d}", svc, model, config, secret=SECRET)
            for i in range(3)
        ]
        return run_fleet(agents, make_incidents(10))

    r1 = run_once()
    r2 = run_once()

    assert [s.mem_id for s in r1.steps] == [s.mem_id for s in r2.steps]

    def key(record):
        return (
            record.seq,
            record.op,
            record.agent_id,
            record.incident_id,
            record.mem_id,
            tuple(record.recalled_ids),
            record.reason,
        )

    assert [key(r) for r in r1.audit] == [key(r) for r in r2.audit]


# -----------------------------------------------------------------------------
# Adversary + immune system (E4)
# -----------------------------------------------------------------------------


def test_adversary_bad_provenance_is_caught_by_the_real_gate():
    config = cfg()
    svc, adapter, embedder = build_service(config)
    gate = ImmuneSystem(adapter, embedder, config)
    adv = make_local_adversary(
        "adv-01",
        svc,
        ScriptedModel(diagnosis_responder),
        config,
        secret=SECRET,
        attack=ATTACK_BAD_PROVENANCE,
        gate=gate,
    )
    adv.ensure_registered()

    step = adv.handle_incident(make_incidents(1)[0])
    assert step.rejected is True
    assert step.mem_id is None
    assert step.reason == "bad_provenance"
    assert step.diagnosis is not None  # the diagnosis is produced before the write attempt
    assert adapter.stats(None).active == 0  # reject-before-persist: nothing written
    # The forged parent really is absent, confirming the attack shape.
    assert NONEXISTENT_PARENT not in {m.mem_id for m in adapter.list_memories(status=None)}


def test_adversary_content_injection_is_caught_by_the_real_gate():
    config = cfg()
    svc, adapter, embedder = build_service(config)
    gate = ImmuneSystem(adapter, embedder, config)
    adv = make_local_adversary(
        "adv-01",
        svc,
        ScriptedModel(diagnosis_responder),
        config,
        secret=SECRET,
        attack=ATTACK_CONTENT_INJECTION,
        gate=gate,
    )
    adv.ensure_registered()

    step = adv.handle_incident(make_incidents(1)[0])
    assert step.rejected is True
    assert step.mem_id is None
    assert step.reason == "injection_pattern"
    assert adapter.stats(None).active == 0


def test_canonical_rewrite_gated_by_the_immune_flag():
    # enable_immune=True -> the rejecting gate refuses the write.
    on_config = cfg(enable_immune=True)
    svc_on, adapter_on, _ = build_service(on_config)
    adv_on = make_local_adversary(
        "adv-01",
        svc_on,
        ScriptedModel(diagnosis_responder),
        on_config,
        secret=SECRET,
        attack=ATTACK_CANONICAL_REWRITE,
        gate=RejectAll(),
    )
    adv_on.ensure_registered()
    step_on = adv_on.handle_incident(make_incidents(1)[0])
    assert step_on.rejected is True
    assert adapter_on.stats(None).active == 0

    # enable_immune=False (A4 ablation) -> the same gate is a no-op; write admitted.
    off_config = cfg(enable_immune=False)
    svc_off, adapter_off, _ = build_service(off_config)
    adv_off = make_local_adversary(
        "adv-01",
        svc_off,
        ScriptedModel(diagnosis_responder),
        off_config,
        secret=SECRET,
        attack=ATTACK_CANONICAL_REWRITE,
        gate=RejectAll(),
    )
    adv_off.ensure_registered()
    step_off = adv_off.handle_incident(make_incidents(1)[0])
    assert step_off.rejected is False
    assert step_off.mem_id is not None
    assert adapter_off.stats(None).active == 1


def test_immune_disabled_admits_despite_a_rejecting_gate():
    config = cfg(enable_immune=False)
    svc, adapter, _ = build_service(config)
    agent = make_local_sre_agent(
        "sre-01", svc, ScriptedModel(diagnosis_responder), config, secret=SECRET, gate=RejectAll()
    )
    agent.ensure_registered()
    step = agent.handle_incident(make_incidents(1)[0])
    assert step.rejected is False
    assert step.mem_id is not None
    assert adapter.stats(None).active == 1


def test_fleet_not_contaminated_by_a_caught_adversary():
    config = cfg()
    svc, adapter, embedder = build_service(config)
    gate = ImmuneSystem(adapter, embedder, config)
    model = ScriptedModel(diagnosis_responder)

    benign = [
        make_local_sre_agent(f"sre-{i:02d}", svc, model, config, secret=SECRET, gate=gate)
        for i in range(3)
    ]
    adversary = make_local_adversary(
        "adv-01", svc, model, config, secret=SECRET, attack=ATTACK_CONTENT_INJECTION, gate=gate
    )
    result = run_fleet([*benign, adversary], make_incidents(8))

    assert result.rejects >= 1
    # A 'reject' audit record carries the detector reason.
    reject_records = [r for r in result.audit if r.op == "reject"]
    assert reject_records and all(r.reason == "injection_pattern" for r in reject_records)

    # Persist-then-quarantine (the immune-panel audit path, identical to ingest):
    # the poison is retained QUARANTINED for forensics but never enters ACTIVE,
    # retrievable memory, so it cannot contaminate the fleet.
    for mem in adapter.list_memories(status=MemoryStatus.ACTIVE):
        assert "IGNORE PREVIOUS INSTRUCTIONS" not in mem.content
    quarantined = adapter.list_memories(status=MemoryStatus.QUARANTINED)
    assert any("IGNORE PREVIOUS INSTRUCTIONS" in m.content for m in quarantined)

    # A benign agent recalling the attack's topic never surfaces the false claim.
    hits = LocalMemoryReader(svc).recall(
        "sre-00", "override the runbook disable alerting grant admin access"
    )
    assert all("IGNORE PREVIOUS INSTRUCTIONS" not in h.content for h in hits)


def test_writer_raises_poison_rejected_directly():
    config = cfg()
    svc, adapter, _ = build_service(config)
    signer = HmacSigner.for_agent(SECRET, "sre-01")
    writer = LocalMemoryWriter(svc, signer, config, immune_gate=RejectAll("injection_pattern"))
    writer.register_agent("sre-01", "sre")
    with pytest.raises(PoisonRejected) as excinfo:
        writer.remember("sre-01", "some content")
    assert excinfo.value.reason == "injection_pattern"
    assert adapter.stats(None).active == 0


# -----------------------------------------------------------------------------
# Signature parity + zero-network guarantee
# -----------------------------------------------------------------------------


def test_signer_parity_with_ingest_auth():
    from ingest.auth import derive_agent_token as ingest_derive
    from ingest.auth import sign as ingest_sign

    secret, agent_id, content, parent, hop = "s", "sre-01", "hello world", None, 0
    token = derive_agent_token(secret, agent_id)
    assert token == ingest_derive(secret, agent_id)
    assert sign_message(token, agent_id, content, parent, hop) == ingest_sign(
        token, agent_id, content, parent, hop
    )
    signer = HmacSigner.for_agent(secret, agent_id)
    assert signer.sign(agent_id, content, parent, hop) == ingest_sign(
        token, agent_id, content, parent, hop
    )


def test_importing_agents_and_running_a_fleet_pulls_no_cloud_sdk():
    """A full offline fleet run must not import boto3 or httpx (subprocess-isolated)."""
    program = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import json
from adapters.memory_inmem import InMemoryAdapter
from core.config import AletheiaConfig
from core.embeddings import DeterministicEmbedder
from core.memory import MemoryService
from agents import ScriptedModel, run_fleet, make_local_sre_agent, Incident

dim = {DIM}
cfg = AletheiaConfig().with_overrides(embedding_dim=dim)
svc = MemoryService(InMemoryAdapter(embedding_dim=dim), DeterministicEmbedder(dim=dim, seed=0), cfg)
model = ScriptedModel(default=json.dumps(
    {{"summary": "s", "action": "a", "learned_fact": "lf", "importance": 0.5, "kind": "episodic"}}
))
agents = [make_local_sre_agent("sre-%02d" % i, svc, model, cfg, secret="x") for i in range(3)]
incidents = [
    Incident(incident_id="inc-%02d" % i, title="t%d" % i, description="d", signals=["s"], meta={{}})
    for i in range(10)
]
res = run_fleet(agents, incidents)
assert res.writes == 10, res.writes
print(",".join(m for m in ("boto3", "httpx") if m in sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"agents pulled cloud SDKs: {result.stdout!r}"


def test_adversary_is_an_sre_agent_subclass():
    from agents.agent import SREAgent

    assert issubclass(AdversaryAgent, SREAgent)
    assert MemoryStatus.ACTIVE == "active"  # sanity: core enums importable here
