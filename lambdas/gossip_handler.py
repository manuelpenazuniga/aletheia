"""AWS Lambda entry point for the gossip tick (C4).

The handler stays thin on purpose. All propagation logic lives in the portable
core (:func:`core.gossip.gossip_tick`); this file only translates between the
Lambda runtime and that core, so the same tick can run from the experiment runner
or the demo app without AWS.

Because this module lives under ``lambdas/`` (not ``core/``), it may import the
CockroachDB adapter and the Bedrock embedder directly — psycopg and boto3 are
fine here. Both are built lazily *inside* the handler so the module stays
importable without a database or AWS credentials configured (the flag-off path
and unit tests carry no infra cost).

Trigger: EventBridge schedule.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from core.config import AletheiaConfig
from core.gossip import gossip_tick

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _log(op: str, **fields: Any) -> None:
    """Structured JSON logging, the observability contract of the project plan §3.8."""
    logger.info(json.dumps({"op": op, **fields}))


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    started = time.monotonic()
    cfg = AletheiaConfig.from_env()
    request_id = getattr(context, "aws_request_id", None)

    if not cfg.enable_gossip:
        _log("gossip.skipped", reason="flag_disabled", request_id=request_id)
        return {"status": "skipped", "reason": "enable_gossip=false"}

    dsn = os.environ.get("ALETHEIA_CRDB_DSN")
    if not dsn:
        _log(
            "gossip.error",
            reason="missing_dsn",
            request_id=request_id,
            latency_ms=round((time.monotonic() - started) * 1000, 2),
        )
        raise RuntimeError(
            "ALETHEIA_CRDB_DSN is not set; the gossip tick needs a "
            "CockroachDB connection string to run"
        )

    # Import here, not at module top: keeps the module importable without psycopg,
    # boto3 or a live database, so the flag-off path and unit tests carry no infra
    # cost.
    from adapters.bedrock_embedder import BedrockEmbedder
    from adapters.cockroach import CockroachDBAdapter

    adapter = CockroachDBAdapter(dsn, embedding_dim=cfg.embedding_dim)
    embedder = BedrockEmbedder(model_id=cfg.embedding_model_id, dim=cfg.embedding_dim)
    try:
        summary = gossip_tick(adapter, embedder, cfg)
    finally:
        adapter.close()

    latency_ms = round((time.monotonic() - started) * 1000, 2)
    _log(
        "gossip.completed",
        request_id=request_id,
        candidates=summary.candidates,
        propagations=summary.propagations,
        capped=summary.capped,
        peers=summary.peers,
        latency_ms=latency_ms,
    )
    return {
        "status": "ok",
        "candidates": summary.candidates,
        "propagations": summary.propagations,
        "capped": summary.capped,
        "peers": summary.peers,
        "latency_ms": latency_ms,
    }
