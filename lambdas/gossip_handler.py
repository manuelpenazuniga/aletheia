"""AWS Lambda entry point for the gossip tick (C4).

Phase 0 skeleton, same contract as ``consolidation_handler``: configuration, flag
check and structured logging now; the propagation logic lands in Phase 2 in
``core/gossip.py``.

Trigger: EventBridge schedule.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from core.config import AletheiaConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _log(op: str, **fields: Any) -> None:
    logger.info(json.dumps({"op": op, **fields}))


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    started = time.monotonic()
    cfg = AletheiaConfig.from_env()
    request_id = getattr(context, "aws_request_id", None)

    if not cfg.enable_gossip:
        _log("gossip.skipped", reason="flag_disabled", request_id=request_id)
        return {"status": "skipped", "reason": "enable_gossip=false"}

    _log(
        "gossip.not_implemented",
        request_id=request_id,
        phase="2",
        max_hops=cfg.gossip_max_hops,
        dsn_configured=bool(os.environ.get("ALETHEIA_CRDB_DSN")),
        latency_ms=round((time.monotonic() - started) * 1000, 2),
    )
    raise NotImplementedError(
        "the gossip tick lands in Phase 2 (core/gossip.py); this handler is the deployment skeleton"
    )
