"""AWS Lambda entry point for the consolidation cycle (C2).

Phase 0 skeleton: it wires up configuration, the flag check and the structured
logging contract, and it deliberately does **no** consolidation work yet — the
logic lands in Phase 1 in ``core/consolidation.py`` and is called from here.

The handler stays thin on purpose. All memory logic lives in the portable core;
this file only translates between the Lambda runtime and that core, so the same
cycle can run from the experiment runner or the demo app without AWS.

Trigger: EventBridge schedule (every 10 minutes in the demo).
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
    """Structured JSON logging, the observability contract of CLAUDE.md §3.8."""
    logger.info(json.dumps({"op": op, **fields}))


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    started = time.monotonic()
    cfg = AletheiaConfig.from_env()
    request_id = getattr(context, "aws_request_id", None)

    if not cfg.enable_consolidation:
        _log("consolidation.skipped", reason="flag_disabled", request_id=request_id)
        return {"status": "skipped", "reason": "enable_consolidation=false"}

    # Phase 1: build the CockroachDB adapter from ALETHEIA_CRDB_DSN and call
    # core.consolidation.run_cycle(adapter, cfg). Kept unimplemented rather than
    # stubbed with fake counts — an invented metric is worse than a missing one.
    _log(
        "consolidation.not_implemented",
        request_id=request_id,
        phase="1",
        dsn_configured=bool(os.environ.get("ALETHEIA_CRDB_DSN")),
        latency_ms=round((time.monotonic() - started) * 1000, 2),
    )
    raise NotImplementedError(
        "the consolidation cycle lands in Phase 1 (core/consolidation.py); "
        "this handler is the deployment skeleton"
    )
