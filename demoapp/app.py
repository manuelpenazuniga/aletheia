"""The public demo app — a fleet + memory dashboard over a real memory layer.

The whole app is a thin read-only view over a :class:`~core.adapter.StorageAdapter`
and an :class:`~core.embeddings.Embedder`, injected via :func:`create_app`. Offline
(the default) it reads the seeded :func:`demoapp.data.build_demo_world`; in
production it is handed a CockroachDBAdapter + a Bedrock embedder and the exact
same endpoints serve live fleet memory. No endpoint invents data — every number
comes from the adapter or from the committed ``docs/experiment_data`` run rows.

Views served (the project plan §9.1): the fleet and what it remembers, live
semantic search against the vector memory, a memory's provenance chain, a
canonical fact's version timeline (the knowledge-update "git of beliefs"), the
immune system's quarantine feed, and the real R1/R2 experiment numbers.

Import-safe: this module imports fastapi (it is not ``core/``); the demo world is
built lazily so importing the module pulls no scenario data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from core.adapter import MemoryNotFound, StorageAdapter
from core.embeddings import Embedder

_STATIC = Path(__file__).resolve().parent / "static"
_EXPERIMENT_DATA = Path(__file__).resolve().parents[1] / "docs" / "experiment_data"
_DEFAULT_SEARCH_K = 8
_DEFAULT_BUDGET = 4000


class SearchRequest(BaseModel):
    query: str
    k: int = _DEFAULT_SEARCH_K


def create_app(
    adapter: StorageAdapter | None = None,
    embedder: Embedder | None = None,
    *,
    reader_agent: str = "sre-01",
) -> FastAPI:
    """Build the dashboard over an injected memory layer.

    With no arguments it builds the offline demo world; pass a CockroachDBAdapter
    + a Bedrock embedder to serve live fleet memory. ``reader_agent`` is the
    (registered) identity the read views query as — institutional memory is
    shared, so this is a caller label, not a filter.
    """
    if adapter is None or embedder is None:
        from demoapp.data import build_demo_world

        world = build_demo_world()
        adapter = adapter or world.adapter
        embedder = embedder or world.embedder

    app = FastAPI(title="Aletheia — fleet memory dashboard", version="1.0.0")
    app.state.adapter = adapter
    app.state.embedder = embedder
    app.state.reader_agent = reader_agent

    _register_routes(app)
    return app


def _adapter(request: Request) -> StorageAdapter:
    return request.app.state.adapter


def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/overview")
    async def overview(request: Request) -> dict[str, Any]:
        adapter = _adapter(request)
        stats = adapter.stats(None)
        agents = _agent_rows(adapter)
        return {
            "totals": {
                "memories": stats.total_memories,
                "active": stats.active,
                "superseded": stats.superseded,
                "quarantined": stats.quarantined,
                "archived": stats.archived,
                "canonical_facts": stats.canonical_facts,
                "cost_tokens": stats.total_cost_tokens,
                "footprint_mb": stats.footprint_mb,
            },
            "agents": agents,
        }

    @app.get("/api/memories")
    async def memories(request: Request, limit: int = 50) -> dict[str, Any]:
        adapter = _adapter(request)
        rows = adapter.list_memories(status=None)[-limit:]
        rows.reverse()
        return {"memories": [_mem_dict(m) for m in rows]}

    @app.post("/api/search")
    async def search(request: Request, body: SearchRequest) -> dict[str, Any]:
        adapter = _adapter(request)
        embedder = request.app.state.embedder
        if not body.query.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")
        vec = embedder.embed(body.query)
        hits = adapter.query_semantic(request.app.state.reader_agent, vec, body.k, _DEFAULT_BUDGET)
        return {"query": body.query, "hits": [_hit_dict(h) for h in hits]}

    @app.get("/api/provenance/{mem_id}")
    async def provenance(request: Request, mem_id: str) -> dict[str, Any]:
        adapter = _adapter(request)
        try:
            chain = adapter.provenance_chain(mem_id)
        except MemoryNotFound:
            raise HTTPException(status_code=404, detail="no such memory") from None
        return {
            "mem_id": mem_id,
            "chain": [
                {
                    "mem_id": link.mem_id,
                    "agent_id": link.agent_id,
                    "parent_mem": link.parent_mem,
                    "hop_count": link.hop_count,
                    "is_root": link.is_root,
                }
                for link in chain
            ],
        }

    @app.get("/api/canonical")
    async def canonical(request: Request) -> dict[str, Any]:
        adapter = _adapter(request)
        # Every fact_key seen in memory metadata is a candidate canonical fact.
        keys = sorted(
            {
                m.meta["fact_key"]
                for m in adapter.list_memories(status=None)
                if isinstance(m.meta, dict) and m.meta.get("fact_key")
            }
        )
        facts = []
        for key in keys:
            fact = adapter.get_canonical(key)
            if fact is not None:
                facts.append(
                    {
                        "fact_key": fact.fact_key,
                        "content": fact.content,
                        "version": fact.version,
                        "source_mem": fact.source_mem,
                    }
                )
        return {"canonical_facts": facts}

    @app.get("/api/canonical/{fact_key:path}/history")
    async def canonical_history(request: Request, fact_key: str) -> dict[str, Any]:
        """The supersede timeline for a fact: every memory carrying this fact_key,
        oldest first, showing which are superseded and which is current."""
        adapter = _adapter(request)
        rows = [
            m
            for m in adapter.list_memories(status=None)
            if isinstance(m.meta, dict) and m.meta.get("fact_key") == fact_key
        ]
        rows.sort(key=lambda m: (m.created_at, m.mem_id or ""))
        if not rows:
            raise HTTPException(status_code=404, detail="no such fact_key")
        return {
            "fact_key": fact_key,
            "timeline": [
                {
                    "mem_id": m.mem_id,
                    "content": m.content,
                    "status": str(m.status),
                    "version": (
                        m.meta.get("runbook_version") if isinstance(m.meta, dict) else None
                    ),
                    "superseded_by": m.superseded_by,
                    "created_at": _iso(m.created_at),
                }
                for m in rows
            ],
        }

    @app.get("/api/quarantine")
    async def quarantine(request: Request) -> dict[str, Any]:
        adapter = _adapter(request)
        log = getattr(adapter, "quarantine_log", None)
        records = log() if callable(log) else []
        feed = []
        for rec in reversed(records):  # newest first
            content = None
            if rec.mem_id:
                try:
                    content = adapter.get_memory(rec.mem_id).content
                except MemoryNotFound:
                    content = None
            feed.append(
                {
                    "mem_id": rec.mem_id,
                    "reason": rec.reason.value if hasattr(rec.reason, "value") else rec.reason,
                    "detector": rec.detector,
                    "payload": rec.payload,
                    "content": content,
                    "created_at": _iso(rec.created_at),
                }
            )
        return {"quarantine": feed}

    @app.get("/api/results")
    async def results() -> JSONResponse:
        """The real R1/R2 experiment numbers, from the committed run rows."""
        return JSONResponse(_load_results())


# --------------------------------------------------------------------------- #
# Serialisation helpers (pure)
# --------------------------------------------------------------------------- #
def _agent_rows(adapter: StorageAdapter) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = {}
    for m in adapter.list_memories(status=None):
        row = counts.setdefault(m.agent_id, {"active": 0, "quarantined": 0, "superseded": 0})
        status = str(m.status)
        if status in row:
            row[status] += 1
    return [
        {"agent_id": aid, **row, "total": sum(row.values())} for aid, row in sorted(counts.items())
    ]


def _mem_dict(m: Any) -> dict[str, Any]:
    return {
        "mem_id": m.mem_id,
        "agent_id": m.agent_id,
        "content": m.content,
        "kind": str(m.kind),
        "status": str(m.status),
        "importance": m.importance,
        "cost_tokens": m.cost_tokens,
        "created_at": _iso(m.created_at),
        "fact_key": m.meta.get("fact_key") if isinstance(m.meta, dict) else None,
    }


def _hit_dict(h: Any) -> dict[str, Any]:
    return {
        "mem_id": h.mem_id,
        "agent_id": h.agent_id,
        "content": h.content,
        "kind": str(h.kind),
        "score": round(h.score, 4),
        "distance": round(h.distance, 4),
        "cost_tokens": h.cost_tokens,
    }


def _iso(ts: Any) -> str | None:
    return ts.isoformat() if ts is not None else None


def _load_results() -> dict[str, Any]:
    """Summarise R1 (E1) and R2 (E2) from the committed experiment_runs export."""
    out: dict[str, Any] = {"r1": [], "r2": [], "available": False}
    e1 = _EXPERIMENT_DATA / "e1_experiment_runs.json"
    e2 = _EXPERIMENT_DATA / "e2_experiment_runs.json"
    if e1.exists():
        rows = json.loads(e1.read_text())
        by_arm: dict[str, list[float]] = {}
        for r in rows:
            m = r["metrics"]
            by_arm.setdefault(r["arm"], []).append(float(m["inconsistencies_per_1000"]))
        for arm in sorted(by_arm):
            vals = by_arm[arm]
            out["r1"].append(
                {
                    "arm": arm,
                    "backend": "naive" if "naive" in arm else "cockroachdb",
                    "n": _n_from_arm(arm),
                    "inconsistencies_per_1000": round(sum(vals) / len(vals), 1),
                    "reps": len(vals),
                }
            )
        out["available"] = True
    if e2.exists():
        for r in json.loads(e2.read_text()):
            m = r["metrics"]
            out["r2"].append(
                {
                    "arm": r["arm"],
                    "event": m.get("event"),
                    "memories_lost": m.get("memories_lost"),
                    "integrity_ok": m.get("integrity_ok"),
                    "recovery_s": m.get("recovery_s"),
                    "fleet_kept_operating": m.get("fleet_kept_operating"),
                    "writes_acknowledged": m.get("writes_acknowledged"),
                }
            )
        out["available"] = True
    return out


def _n_from_arm(arm: str) -> int | None:
    for token in arm.split("_"):
        if token.startswith("n") and token[1:].isdigit():
            return int(token[1:])
    return None
