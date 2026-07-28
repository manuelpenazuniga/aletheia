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
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from core.adapter import MemoryNotFound, StorageAdapter
from core.config import AletheiaConfig
from core.embeddings import Embedder

_STATIC = Path(__file__).resolve().parent / "static"
_EXPERIMENT_DATA = Path(__file__).resolve().parents[1] / "docs" / "experiment_data"
_DEFAULT_SEARCH_K = 8
_MAX_SEARCH_K = 50
_MAX_QUERY_LEN = 500
_MAX_LIST_LIMIT = 500
_DEFAULT_BUDGET = 4000


class SearchRequest(BaseModel):
    query: str
    k: int = _DEFAULT_SEARCH_K


class KillSwitchRequest(BaseModel):
    enable_consolidation: bool = True
    enable_forgetting: bool = True
    enable_gossip: bool = True
    enable_immune: bool = True


class AttackRequest(BaseModel):
    category: str = "injection_pattern"
    token: str | None = None


_DEMO_TOKEN_ENV = "ALETHEIA_DEMO_TOKEN"
# Cap on attack-generated quarantine rows per process, so a public "launch attack"
# button cannot grow memory without bound (the project plan §9.1 security note).
_MAX_LAUNCHED_ATTACKS = 100


def create_app(
    adapter: StorageAdapter | None = None,
    embedder: Embedder | None = None,
    *,
    reader_agent: str = "sre-01",
    config: Any = None,
) -> FastAPI:
    """Build the dashboard over an injected memory layer.

    With no arguments it builds the offline demo world; pass a CockroachDBAdapter
    + a Bedrock embedder to serve live fleet memory. ``reader_agent`` is the
    (registered) identity the read views query as — institutional memory is
    shared, so this is a caller label, not a filter. ``config`` (the demo config
    when offline) drives the immune gate behind the launch-attack button.
    """
    from demoapp.data import demo_config

    injected = adapter is not None and embedder is not None
    if not injected:
        from demoapp.data import build_demo_world

        world = build_demo_world()
        adapter = adapter or world.adapter
        embedder = embedder or world.embedder
        config = config or world.config

    # The data source, surfaced on every relevant response and in the UI so
    # seeded offline data is NEVER mistaken for a live fleet (the project plan §3.2).
    mode = "live" if injected else "offline-demo"

    app = FastAPI(title="Aletheia — fleet memory dashboard", version="1.0.0")
    app.state.adapter = adapter
    app.state.embedder = embedder
    app.state.reader_agent = reader_agent
    app.state.mode = mode
    app.state.config = config or demo_config()
    app.state.attacks_launched = 0

    _register_routes(app)
    return app


def _adapter(request: Request) -> StorageAdapter:
    return request.app.state.adapter


def _projection(app: FastAPI):
    """Fit (once, cached) a 2D landmark projection over every memory that carries an
    embedding, and return it with those memories. Deterministic, so caching is safe."""
    cache = getattr(app.state, "_projection", None)
    if cache is None:
        from demoapp.projection import fit

        mems = [m for m in app.state.adapter.list_memories(status=None) if m.embedding]
        proj = fit([m.embedding for m in mems])
        cache = (proj, mems)
        app.state._projection = cache
    return cache


def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, str]:
        return {"status": "ok", "mode": request.app.state.mode}

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/overview")
    async def overview(request: Request) -> dict[str, Any]:
        adapter = _adapter(request)
        stats = adapter.stats(None)
        agents = _agent_rows(adapter)
        return {
            "mode": request.app.state.mode,
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
    async def memories(
        request: Request,
        limit: int = Query(default=50, ge=1, le=_MAX_LIST_LIMIT),
    ) -> dict[str, Any]:
        adapter = _adapter(request)
        rows = adapter.list_memories(status=None)[-limit:]
        rows.reverse()
        return {"memories": [_mem_dict(m) for m in rows]}

    @app.post("/api/search")
    async def search(request: Request, body: SearchRequest) -> dict[str, Any]:
        adapter = _adapter(request)
        embedder = request.app.state.embedder
        query = body.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        if len(query) > _MAX_QUERY_LEN:
            raise HTTPException(status_code=422, detail=f"query exceeds {_MAX_QUERY_LEN} chars")
        k = max(1, min(body.k, _MAX_SEARCH_K))
        vec = embedder.embed(query)
        hits = adapter.query_semantic(request.app.state.reader_agent, vec, k, _DEFAULT_BUDGET)
        return {"query": query, "hits": [_hit_dict(h) for h in hits]}

    @app.get("/api/vector_map")
    async def vector_map(request: Request) -> dict[str, Any]:
        """The vector memory space, projected to 2D — every memory as a point,
        placed by its real embedding. Powers the interactive map view. Coordinates
        come from a deterministic landmark projection cached on first call."""
        proj, mems = _projection(request.app)
        points = [
            {
                "mem_id": m.mem_id,
                "x": round(x, 4),
                "y": round(y, 4),
                "agent_id": m.agent_id,
                "status": str(m.status),
                "kind": str(m.kind),
                "content": m.content,
            }
            for m in mems
            for x, y in [proj.project(m.embedding)]
        ]
        return {"points": points, "count": len(points)}

    @app.post("/api/vector_search")
    async def vector_search(request: Request, body: SearchRequest) -> dict[str, Any]:
        """Semantic search for the map: the real query_semantic nearest neighbours,
        plus the query's own 2D position under the SAME projection so the map can
        drop the query point and light up the memories it actually retrieved."""
        embedder = request.app.state.embedder
        query = body.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        if len(query) > _MAX_QUERY_LEN:
            raise HTTPException(status_code=422, detail=f"query exceeds {_MAX_QUERY_LEN} chars")
        k = max(1, min(body.k, _MAX_SEARCH_K))
        proj, _ = _projection(request.app)
        vec = embedder.embed(query)
        hits = _adapter(request).query_semantic(
            request.app.state.reader_agent, vec, k, _DEFAULT_BUDGET
        )
        qx, qy = proj.project(vec)
        return {
            "query": query,
            "query_xy": [round(qx, 4), round(qy, 4)],
            "hits": [{"mem_id": h.mem_id, "score": round(h.score, 4)} for h in hits],
        }

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

    @app.get("/api/attack/categories")
    async def attack_categories() -> dict[str, Any]:
        from demoapp.data import poison_categories

        return {"categories": poison_categories(), "token_required": _token_required()}

    @app.post("/api/attack")
    async def attack(request: Request, body: AttackRequest) -> dict[str, Any]:
        """Release the adversary (§9.1.5): run one committed poison case through the
        REAL immune gate against this app's memory. A caught attack lands in the
        quarantine feed exactly as a production interception would — not staged.

        Token-gated (the project plan §9.1): destructive/write demo actions require
        ``ALETHEIA_DEMO_TOKEN`` when it is set. Rate-capped per process so a public
        button cannot grow memory without bound."""
        from demoapp.data import launch_attack, poison_categories

        _require_demo_token(body.token)
        if body.category not in poison_categories():
            raise HTTPException(status_code=422, detail=f"unknown attack category: {body.category}")
        if request.app.state.attacks_launched >= _MAX_LAUNCHED_ATTACKS:
            raise HTTPException(status_code=429, detail="attack rate cap reached for this session")

        index = request.app.state.attacks_launched
        request.app.state.attacks_launched += 1
        result = launch_attack(
            request.app.state.adapter,
            request.app.state.embedder,
            request.app.state.config,
            category=body.category,
            index=index,
        )
        result["simulated"] = request.app.state.mode != "live"
        return result

    @app.get("/api/contagion")
    async def contagion(request: Request) -> dict[str, Any]:
        """Hallucination contagion: how one poisoned fact spreads through the fleet
        with the immune system off vs on (§0 — the adverse phenomenon the immune
        system solves). Both scenarios are real gossip propagation traced from
        provenance — a seeded offline simulation, cached (deterministic)."""
        cache = getattr(request.app.state, "_contagion", None)
        if cache is None:
            from demoapp.data import build_contagion

            cache = {
                "simulated": True,
                "source": "seeded-offline",
                "immune_off": build_contagion(enable_immune=False),
                "immune_on": build_contagion(enable_immune=True),
            }
            request.app.state._contagion = cache
        return cache

    @app.get("/api/results")
    async def results() -> JSONResponse:
        """The real R1/R2 experiment numbers, from the committed run rows."""
        return JSONResponse(_load_results())

    @app.post("/api/killswitch")
    async def killswitch(body: KillSwitchRequest) -> dict[str, Any]:
        """Flip the four feature flags and rebuild the world for real, returning the
        recomputed metrics next to the all-on baseline. This is the interactive
        proof (§9.1.3) that each component is necessary: the judge turns one off and
        watches stale knowledge, poison, or footprint climb.

        ALWAYS a seeded offline simulation — it rebuilds an InMemoryAdapter world
        regardless of the injected adapter, so it never reads or mutates live fleet
        memory. The ``simulated`` flag says so on every response; the UI labels it."""
        from demoapp.data import build_demo_world, demo_config, flags_of, world_metrics

        cfg = demo_config(
            AletheiaConfig(
                enable_consolidation=body.enable_consolidation,
                enable_forgetting=body.enable_forgetting,
                enable_gossip=body.enable_gossip,
                enable_immune=body.enable_immune,
            )
        )
        world = build_demo_world(config=cfg)
        return {
            "simulated": True,
            "source": "seeded-offline",
            "flags": flags_of(cfg),
            "metrics": world_metrics(world.adapter),
            "baseline": world_metrics(_baseline_world(app).adapter),
        }

    @app.get("/api/ablation")
    async def ablation(request: Request) -> dict[str, Any]:
        """The ablation wall (§9.1.6): the full world and each single-component
        ablation, same scenario, scored by the same metrics. Always a seeded offline
        simulation (never live memory). Cached on first call — the worlds are
        deterministic."""
        cache = getattr(request.app.state, "_ablation", None)
        if cache is None:
            from demoapp.data import build_ablation_panels

            cache = build_ablation_panels()
            request.app.state._ablation = cache
        return {"simulated": True, "source": "seeded-offline", "panels": cache}


def _token_required() -> bool:
    return bool(os.environ.get(_DEMO_TOKEN_ENV))


def _require_demo_token(supplied: str | None) -> None:
    """Enforce the demo token on write/attack actions when one is configured.

    When ``ALETHEIA_DEMO_TOKEN`` is unset (local dev) the gate is open; a public
    deployment sets it, and every attack must present it. Fail-closed on mismatch."""
    required = os.environ.get(_DEMO_TOKEN_ENV)
    if required and supplied != required:
        raise HTTPException(status_code=403, detail="invalid or missing demo token")


def _baseline_world(app: FastAPI):
    """The all-on seeded world, built once and cached. Deterministic, so caching it
    is safe and makes /api/killswitch comparisons against a single fixed baseline."""
    cache = getattr(app.state, "_baseline", None)
    if cache is None:
        from demoapp.data import build_demo_world, demo_config

        cache = build_demo_world(config=demo_config())
        app.state._baseline = cache
    return cache


# --------------------------------------------------------------------------- #
# Serialisation helpers (pure)
# --------------------------------------------------------------------------- #
_AGENT_STATUSES = ("active", "superseded", "quarantined", "archived")


def _agent_rows(adapter: StorageAdapter) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = {}
    for m in adapter.list_memories(status=None):
        row = counts.setdefault(m.agent_id, dict.fromkeys(_AGENT_STATUSES, 0))
        status = str(m.status)
        if status in row:
            row[status] += 1
    # total covers EVERY status (incl. archived), so agent rows reconcile with the
    # fleet-wide total_memories — no rows silently dropped.
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


def _authoritative(row: dict) -> bool:
    """Only real, non-dry-run measurements count — the integrity policy (§3.1, §8.6).
    A future dry run in the export must never be averaged in as a real result."""
    m = row.get("metrics", {})
    return bool(m.get("authoritative")) and not m.get("dry_run", False)


def _backend_of(metrics: dict, arm: str) -> str:
    """Derive the backend from the recorded field, not the arm name."""
    backend = str(metrics.get("backend") or arm)
    if "naive" in backend:
        return "naive"
    if "crdb" in backend or "cockroach" in backend:
        return "cockroachdb"
    return "unknown"


def _load_results() -> dict[str, Any]:
    """Summarise R1 (E1) and R2 (E2) from the committed experiment_runs export.

    Fails closed on integrity: only rows flagged ``authoritative`` and not
    ``dry_run`` are read; backend and N are taken from the recorded metrics fields,
    never guessed from the arm string."""
    out: dict[str, Any] = {"r1": [], "r2": [], "available": False}
    e1 = _EXPERIMENT_DATA / "e1_experiment_runs.json"
    e2 = _EXPERIMENT_DATA / "e2_experiment_runs.json"
    if e1.exists():
        rows = [r for r in json.loads(e1.read_text()) if _authoritative(r)]
        by_arm: dict[str, list[float]] = {}
        meta: dict[str, dict] = {}
        for r in rows:
            m = r["metrics"]
            by_arm.setdefault(r["arm"], []).append(float(m["inconsistencies_per_1000"]))
            meta.setdefault(r["arm"], m)
        for arm in sorted(by_arm):
            vals = by_arm[arm]
            out["r1"].append(
                {
                    "arm": arm,
                    "backend": _backend_of(meta[arm], arm),
                    "n": meta[arm].get("n_agents"),
                    "inconsistencies_per_1000": round(sum(vals) / len(vals), 1),
                    "reps": len(vals),
                }
            )
        out["available"] = out["available"] or bool(rows)
    if e2.exists():
        for r in json.loads(e2.read_text()):
            if not _authoritative(r):
                continue
            m = r["metrics"]
            out["r2"].append(
                {
                    "arm": r["arm"],
                    "backend": _backend_of(m, r["arm"]),
                    "event": m.get("event"),
                    "memories_lost": m.get("memories_lost"),
                    "integrity_ok": m.get("integrity_ok"),
                    "recovery_s": m.get("recovery_s"),
                    "fleet_kept_operating": m.get("fleet_kept_operating"),
                    "writes_acknowledged": m.get("writes_acknowledged"),
                }
            )
        out["available"] = out["available"] or bool(out["r2"])
    return out
