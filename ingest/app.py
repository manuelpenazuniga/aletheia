"""The ingest write-path service (FastAPI) — the sole holder of DB write creds.

Separation of paths (CLAUDE.md §5.3): agents *read* through the managed MCP
server (read-only) and *write* only through this service, which fronts the
database with authentication, the immune gate, and a single atomic write. The
managed MCP is never given write access; this validated write path is the
minimum-privilege alternative.

This module imports **nothing** from psycopg or boto3. The adapter and embedder
are injected into :func:`create_app`, so the same app runs over the in-memory
adapter with a deterministic embedder (tests) and over CockroachDB with Bedrock
(production, wired only in :mod:`ingest.main`). That confinement is guarded by
``tests/test_ingest.py::test_importing_ingest_app_does_not_pull_infra``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from core.adapter import (
    AgentNotRegistered,
    DuplicateMemory,
    MemoryNotFound,
    StorageAdapter,
)
from core.config import AletheiaConfig
from core.embeddings import Embedder
from core.memory import MemoryService
from core.models import MemoryEvent
from ingest.auth import Unauthorized, get_hmac_secret, require_bearer, verify
from ingest.immune import ImmuneGate, PassthroughImmuneGate
from ingest.schemas import HealthResponse, RememberRequest, RememberResponse

logger = logging.getLogger("aletheia.ingest")
if not logger.handlers:  # pragma: no cover - logging setup, not behaviour
    logger.setLevel(logging.INFO)


def _log(op: str, **fields: Any) -> None:
    """One structured JSON line. Never logs secrets, tokens, signatures or full content."""
    logger.info(json.dumps({"op": op, **fields}, default=str))


# --------------------------------------------------------------------- providers
# Each reads what create_app stashed on app.state. Kept tiny so the endpoint's
# dependencies are explicit and individually overridable in tests.
def get_config(request: Request) -> AletheiaConfig:
    return request.app.state.config


def get_memory_service(request: Request) -> MemoryService:
    return request.app.state.memory_service


def get_immune_gate(request: Request) -> ImmuneGate:
    return request.app.state.immune_gate


def get_adapter(request: Request) -> StorageAdapter:
    return request.app.state.adapter


def create_app(
    adapter: StorageAdapter,
    embedder: Embedder,
    *,
    hmac_secret: str,
    config: AletheiaConfig | None = None,
    immune_gate: ImmuneGate | None = None,
) -> FastAPI:
    """Build the ingest app over injected infrastructure.

    Args:
        adapter: any :class:`~core.adapter.StorageAdapter` (in-memory in tests,
            CockroachDB in production).
        embedder: the trusted :class:`~core.embeddings.Embedder`; the vector is
            computed here, never accepted from the client.
        hmac_secret: server signing secret. Must be non-empty — an empty key
            would make every derived token and signature forgeable.
        config: feature flags + dims. Defaults to :class:`AletheiaConfig`.
        immune_gate: the C5 seam. Defaults to :class:`PassthroughImmuneGate`.

    The :class:`MemoryService` constructor raises here (startup, not per-request)
    if ``embedder.dim != config.embedding_dim`` — a dimension mismatch must fail
    the process, not corrupt the index one write at a time.
    """
    if not hmac_secret:
        raise ValueError("hmac_secret must be non-empty (never sign with an empty key)")

    cfg = config or AletheiaConfig()
    service = MemoryService(adapter, embedder, cfg)

    app = FastAPI(title="Aletheia Ingest", version="1.0.0")
    app.state.adapter = adapter
    app.state.memory_service = service
    app.state.config = cfg
    app.state.hmac_secret = hmac_secret
    app.state.immune_gate = immune_gate or PassthroughImmuneGate()

    _register_exception_handlers(app)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse()

    @app.post("/v1/memories", response_model=RememberResponse, status_code=201)
    async def remember(
        body: RememberRequest,
        bearer: str = Depends(require_bearer),
        secret: str = Depends(get_hmac_secret),
        svc: MemoryService = Depends(get_memory_service),
        adapter: StorageAdapter = Depends(get_adapter),
        gate: ImmuneGate = Depends(get_immune_gate),
        cfg: AletheiaConfig = Depends(get_config),
    ) -> RememberResponse:
        started = time.perf_counter()

        # Authenticate the per-agent token and the signed provenance against the
        # single parsed body. Raises Unauthorized -> 401 on any mismatch.
        auth = verify(secret, body, bearer)

        # Immune gate — only when enabled. Phase 1's passthrough allows every
        # write, so this is a no-op today; the seam is here for Phase 2's real
        # detector. Runs BEFORE any write, so a rejection persists nothing.
        if cfg.enable_immune:
            provisional = MemoryEvent(
                agent_id=auth.agent_id,
                content=body.content,
                kind=body.kind,
                importance=body.importance,
                parent_mem=body.provenance.parent_mem,
                hop_count=body.provenance.hop_count,
                signature=body.provenance.signature,
                meta=dict(body.meta),
            )
            verdict = gate.inspect(provisional)
            if not verdict.allowed:
                # NOTE (Phase 2): persisting this rejection to quarantine_log has
                # no home in the current adapter contract (no mem_id exists yet).
                # Flagged in the design; today we reject without an audit write.
                _log(
                    "ingest.remember",
                    agent_id=auth.agent_id,
                    immune="reject",
                    reason=verdict.reason,
                    detector=verdict.detector,
                    status_code=422,
                )
                raise HTTPException(
                    status_code=422,
                    detail={"reason": verdict.reason, "detector": verdict.detector},
                )

        # One atomic memory + provenance write. The embedding is computed inside
        # remember() from content — the agent never supplies a vector.
        mem_id = svc.remember(
            auth.agent_id,
            body.content,
            kind=body.kind,
            importance=body.importance,
            parent_mem=body.provenance.parent_mem,
            hop_count=body.provenance.hop_count,
            signature=body.provenance.signature,
            meta=body.meta,
        )
        stored = adapter.get_memory(mem_id)

        _log(
            "ingest.remember",
            agent_id=auth.agent_id,
            mem_id=mem_id,
            kind=str(stored.kind),
            immune="allow",
            content_len=len(body.content),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            status_code=201,
        )
        return RememberResponse(
            mem_id=mem_id,
            agent_id=stored.agent_id,
            kind=stored.kind,
            status=stored.status,
            cost_tokens=stored.cost_tokens or 0,
            created_at=stored.created_at,
        )


def _register_exception_handlers(app: FastAPI) -> None:
    """Map core adapter errors to HTTP, never leaking secrets or DSNs to the client."""

    @app.exception_handler(Unauthorized)
    async def _unauthorized(request: Request, exc: Unauthorized) -> JSONResponse:
        # Log the specific reason server-side; return a generic body + the
        # challenge header. A prober learns only "401", not which check failed.
        _log("ingest.auth_failed", detail=str(exc), status_code=401)
        return JSONResponse(
            status_code=401,
            content={"detail": "unauthorized"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(AgentNotRegistered)
    async def _agent_not_registered(request: Request, exc: AgentNotRegistered) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": "agent not authorized to write"})

    @app.exception_handler(MemoryNotFound)
    async def _memory_not_found(request: Request, exc: MemoryNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "parent memory not found"})

    @app.exception_handler(DuplicateMemory)
    async def _duplicate(request: Request, exc: DuplicateMemory) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": "memory already exists"})

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        # Defense in depth: a core-model constraint that slipped past pydantic.
        # str(exc) here is a domain message ("importance must be in ..."), never
        # a secret — core models never see the signing key or DSN.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the real exception server-side only; the response is generic so a
        # DSN, secret, or stack detail can never leak through the body.
        _log("ingest.error", error_type=type(exc).__name__, status_code=500)
        return JSONResponse(status_code=500, content={"detail": "internal error"})
