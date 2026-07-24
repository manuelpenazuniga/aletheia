"""Ingest write-path tests: TestClient over InMemoryAdapter + DeterministicEmbedder.

Everything here is offline and deterministic — the whole point of injecting the
adapter and embedder into ``create_app`` is that the HTTP surface, the auth, and
the write path are exercised without CockroachDB or Bedrock.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from adapters.memory_inmem import InMemoryAdapter
from core.config import AletheiaConfig
from core.embeddings import DeterministicEmbedder
from ingest.app import create_app
from ingest.auth import Unauthorized, derive_agent_token, sign, verify
from ingest.immune import ImmuneGate, ImmuneVerdict, PassthroughImmuneGate
from ingest.schemas import Provenance, RememberRequest

TEST_DIM = 8
SECRET = "test-server-secret"


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def adapter() -> InMemoryAdapter:
    store = InMemoryAdapter(embedding_dim=TEST_DIM)
    store.register_agent("sre-1", "sre", "hash-1")
    store.register_agent("sre-2", "sre", "hash-2")
    return store


@pytest.fixture
def config() -> AletheiaConfig:
    # embedding_dim MUST match the test embedder or MemoryService raises at build.
    return AletheiaConfig(embedding_dim=TEST_DIM)


@pytest.fixture
def make_client(adapter, config):
    """Factory: build a TestClient over the ingest app, overriding pieces per test."""

    def _make(
        *,
        cfg: AletheiaConfig | None = None,
        immune_gate: ImmuneGate | None = None,
        store: InMemoryAdapter | None = None,
        raise_server_exceptions: bool = True,
    ) -> TestClient:
        app = create_app(
            store or adapter,
            DeterministicEmbedder(dim=TEST_DIM, seed=0),
            hmac_secret=SECRET,
            config=cfg or config,
            immune_gate=immune_gate,
        )
        return TestClient(app, raise_server_exceptions=raise_server_exceptions)

    return _make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def signed_body(
    agent_id: str,
    content: str,
    *,
    kind: str = "episodic",
    importance: float = 0.5,
    parent_mem: str | None = None,
    hop_count: int = 0,
    meta: dict | None = None,
    signature: str | None = None,
) -> tuple[str, dict]:
    """Return (bearer_token, request_body) for a correctly signed write."""
    token = derive_agent_token(SECRET, agent_id)
    sig = (
        signature
        if signature is not None
        else sign(token, agent_id, content, parent_mem, hop_count)
    )
    body = {
        "agent_id": agent_id,
        "content": content,
        "kind": kind,
        "importance": importance,
        "provenance": {"parent_mem": parent_mem, "hop_count": hop_count, "signature": sig},
        "meta": meta or {},
    }
    return token, body


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------- happy path
def test_healthz_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_remember_persists_memory(client, adapter):
    token, body = signed_body("sre-1", "disk full on node 3")
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 201
    mem_id = resp.json()["mem_id"]

    stored = adapter.get_memory(mem_id)
    assert stored.content == "disk full on node 3"
    chain = adapter.provenance_chain(mem_id)
    assert len(chain) == 1
    assert chain[0].mem_id == mem_id


def test_embedding_computed_server_side(client, adapter):
    token, body = signed_body("sre-1", "cert expired on the gateway")
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 201
    stored = adapter.get_memory(resp.json()["mem_id"])
    assert stored.embedding is not None
    assert len(stored.embedding) == TEST_DIM


def test_meta_is_persisted(client, adapter):
    token, body = signed_body("sre-1", "deploy rolled back", meta={"incident": "INC-1"})
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 201
    stored = adapter.get_memory(resp.json()["mem_id"])
    assert stored.meta.get("incident") == "INC-1"


def test_gossip_write_with_parent_and_hop(client, adapter):
    token, root_body = signed_body("sre-1", "root observation")
    root_id = client.post("/v1/memories", json=root_body, headers=auth_headers(token)).json()[
        "mem_id"
    ]

    child_token, child_body = signed_body(
        "sre-2", "heard from sre-1", parent_mem=root_id, hop_count=1
    )
    resp = client.post("/v1/memories", json=child_body, headers=auth_headers(child_token))
    assert resp.status_code == 201
    child_id = resp.json()["mem_id"]

    chain = adapter.provenance_chain(child_id)
    assert len(chain) == 2
    assert chain[0].mem_id == child_id
    assert chain[0].hop_count == 1
    assert chain[1].mem_id == root_id


# ---------------------------------------------------------- body / schema (422)
def test_raw_embedding_field_rejected(client, adapter):
    token, body = signed_body("sre-1", "some content")
    body["embedding"] = [0.1] * TEST_DIM
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 422
    assert adapter.stats().total_memories == 0


def test_empty_content_422(client):
    token, body = signed_body("sre-1", "x")
    body["content"] = ""
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 422


def test_empty_signature_422(client):
    token, body = signed_body("sre-1", "x")
    body["provenance"]["signature"] = ""
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 422


def test_importance_out_of_range_422(client):
    token, body = signed_body("sre-1", "x", importance=1.5)
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 422


def test_invalid_kind_422(client):
    token, body = signed_body("sre-1", "x", kind="not-a-kind")
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 422


def test_negative_hop_422(client):
    token, body = signed_body("sre-1", "x", hop_count=-1)
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 422


# ----------------------------------------------------------------- auth (401/403)
def test_missing_authorization_header(client, adapter):
    _, body = signed_body("sre-1", "content")
    resp = client.post("/v1/memories", json=body)
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"
    assert adapter.stats().total_memories == 0


def test_malformed_authorization_scheme(client):
    token, body = signed_body("sre-1", "content")
    resp = client.post("/v1/memories", json=body, headers={"Authorization": f"Token {token}"})
    assert resp.status_code == 401


def test_wrong_bearer_token(client, adapter):
    _, body = signed_body("sre-1", "content")
    resp = client.post("/v1/memories", json=body, headers=auth_headers("deadbeef"))
    assert resp.status_code == 401
    assert adapter.stats().total_memories == 0


def test_tampered_content_signature_mismatch(client, adapter):
    # Sign "A", send "B".
    token, body = signed_body("sre-1", "content A")
    body["content"] = "content B"
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 401
    assert adapter.stats().total_memories == 0


def test_tampered_parent_or_hop_signature_mismatch(client):
    token, body = signed_body("sre-1", "content", hop_count=0)
    body["provenance"]["hop_count"] = 2  # altered after signing
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 401


def test_cross_agent_token_rejected(client):
    # Token derived for sre-1, body claims sre-2.
    token = derive_agent_token(SECRET, "sre-1")
    _, body = signed_body("sre-2", "content")  # body + sig for sre-2
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 401


def test_unregistered_agent_forbidden(client):
    token, body = signed_body("ghost", "content")  # correctly signed, but not registered
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 403


def test_parent_mem_not_found_404(client):
    token, body = signed_body("sre-1", "content", parent_mem="nope", hop_count=1)
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 404


# ---------------------------------------------------------------------- immune
def test_immune_disabled_passthrough(make_client, config):
    client = make_client(cfg=config.with_overrides(enable_immune=False))
    token, body = signed_body("sre-1", "content")
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 201


def test_immune_enabled_passthrough_phase1(make_client, config):
    client = make_client(
        cfg=config.with_overrides(enable_immune=True), immune_gate=PassthroughImmuneGate()
    )
    token, body = signed_body("sre-1", "content")
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 201


def test_immune_reject_returns_422(make_client, config, adapter):
    class RejectingGate:
        def inspect(self, event) -> ImmuneVerdict:
            return ImmuneVerdict(allowed=False, reason="bad_provenance", detector="test")

    client = make_client(cfg=config.with_overrides(enable_immune=True), immune_gate=RejectingGate())
    token, body = signed_body("sre-1", "content")
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["reason"] == "bad_provenance"
    assert detail["detector"] == "test"
    assert adapter.stats().total_memories == 0  # rejected before any write


# -------------------------------------------------------------- error isolation
def test_internal_error_leaks_nothing(make_client):
    dsn = "postgresql://user:pw@secret-host:26257/aletheia"

    class ExplodingAdapter(InMemoryAdapter):
        def write_episode(self, agent_id, event):
            raise RuntimeError(f"boom secret={SECRET} dsn={dsn}")

    store = ExplodingAdapter(embedding_dim=TEST_DIM)
    store.register_agent("sre-1", "sre", "hash-1")
    client = make_client(store=store, raise_server_exceptions=False)

    token, body = signed_body("sre-1", "content")
    resp = client.post("/v1/memories", json=body, headers=auth_headers(token))
    assert resp.status_code == 500
    assert resp.json() == {"detail": "internal error"}
    assert SECRET not in resp.text
    assert "postgresql://" not in resp.text
    assert "secret-host" not in resp.text


# ------------------------------------------------------------------- auth unit
def test_auth_unit_roundtrip():
    agent_id, content = "sre-1", "the runbook fix changed"
    token = derive_agent_token(SECRET, agent_id)
    sig = sign(token, agent_id, content, None, 0)
    req = RememberRequest(
        agent_id=agent_id,
        content=content,
        provenance=Provenance(parent_mem=None, hop_count=0, signature=sig),
    )
    ctx = verify(SECRET, req, token)
    assert ctx.agent_id == agent_id
    assert ctx.agent_token == token

    tampered = RememberRequest(
        agent_id=agent_id,
        content=content + "!",  # one char flipped
        provenance=Provenance(parent_mem=None, hop_count=0, signature=sig),
    )
    with pytest.raises(Unauthorized):
        verify(SECRET, tampered, token)


# ------------------------------------------------------------ infra confinement
def test_importing_ingest_app_does_not_pull_infra():
    """Importing ingest.app must not drag psycopg/boto3 into the interpreter."""
    program = (
        "import sys; import ingest.app; "
        "leaked=[m for m in ('psycopg','boto3','botocore') if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"ingest.app pulled in infra: {result.stdout.strip()}"
