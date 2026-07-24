"""AWS Lambda entry point for the consolidation cycle (C2).

The handler stays thin on purpose. All memory logic lives in the portable core
(:func:`core.consolidation.consolidate`); this file only translates between the
Lambda runtime and that core, so the same cycle can run from the experiment
runner or the demo app without AWS.

Because this module lives under ``lambdas/`` (not ``core/``), it may import the
CockroachDB adapter directly — psycopg is fine here. The adapter is built lazily
*inside* the handler from ``ALETHEIA_CRDB_DSN`` so the module stays importable
without a database configured.

Trigger: EventBridge schedule (every 10 minutes in the demo).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from core.config import AletheiaConfig
from core.consolidation import consolidate

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _log(op: str, **fields: Any) -> None:
    """Structured JSON logging, the observability contract of CLAUDE.md §3.8."""
    logger.info(json.dumps({"op": op, **fields}))


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    started = time.monotonic()
    cfg = AletheiaConfig.from_env()
    request_id = getattr(context, "aws_request_id", None)

    if not cfg.enable_consolidation:
        _log("consolidation.skipped", reason="flag_disabled", request_id=request_id)
        return {"status": "skipped", "reason": "enable_consolidation=false"}

    dsn = os.environ.get("ALETHEIA_CRDB_DSN")
    if not dsn:
        _log(
            "consolidation.error",
            reason="missing_dsn",
            request_id=request_id,
            latency_ms=round((time.monotonic() - started) * 1000, 2),
        )
        raise RuntimeError(
            "ALETHEIA_CRDB_DSN is not set; the consolidation cycle needs a "
            "CockroachDB connection string to run"
        )

    # Import here, not at module top: keeps the module importable without psycopg
    # or a live database, so the flag-off path and unit tests carry no infra cost.
    from adapters.cockroach import CockroachDBAdapter

    adapter = CockroachDBAdapter(dsn, embedding_dim=cfg.embedding_dim)
    try:
        summary = consolidate(adapter, cfg)
    finally:
        adapter.close()

    latency_ms = round((time.monotonic() - started) * 1000, 2)
    _log(
        "consolidation.completed",
        request_id=request_id,
        groups=summary.groups,
        supersedes=summary.supersedes,
        canonical_updates=summary.canonical_updates,
        latency_ms=latency_ms,
    )
    return {
        "status": "ok",
        "groups": summary.groups,
        "supersedes": summary.supersedes,
        "canonical_updates": summary.canonical_updates,
        "latency_ms": latency_ms,
    }
