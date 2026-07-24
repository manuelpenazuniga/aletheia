"""Production wiring for the ingest service — the ONLY place ingest meets infra.

``ingest/app.py`` is infrastructure-free so tests run it over the in-memory
adapter. This module is where the real CockroachDB adapter and the Bedrock
embedder are constructed and injected, and it is the only ingest module that
imports psycopg/boto3 (transitively, through those adapters). ``uvicorn
ingest.main:app`` serves it.
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from adapters.bedrock_embedder import BedrockEmbedder
from adapters.cockroach import CockroachDBAdapter
from core.config import AletheiaConfig
from ingest.app import create_app


def build_from_env() -> FastAPI:
    """Construct the app from ``ALETHEIA_*`` environment variables.

    Fails fast on a missing DSN or an empty signing secret: the service must
    never come up able to accept writes it cannot persist, nor sign with an
    empty key.
    """
    config = AletheiaConfig.from_env()

    dsn = os.environ.get("ALETHEIA_CRDB_DSN", "").strip()
    if not dsn:
        raise RuntimeError(
            "ALETHEIA_CRDB_DSN is not set: the ingest service needs the CockroachDB "
            "connection string to persist writes (see .env.example)."
        )

    secret = os.environ.get("ALETHEIA_INGEST_HMAC_SECRET", "").strip()
    if not secret:
        raise RuntimeError(
            "ALETHEIA_INGEST_HMAC_SECRET is not set: refusing to start with an empty "
            "signing key (every agent token and signature would be forgeable)."
        )

    adapter = CockroachDBAdapter(dsn=dsn, embedding_dim=config.embedding_dim)
    embedder = BedrockEmbedder(
        model_id=config.embedding_model_id,
        dim=config.embedding_dim,
    )
    return create_app(adapter, embedder, hmac_secret=secret, config=config)


app = build_from_env() if os.environ.get("ALETHEIA_CRDB_DSN") else None
"""ASGI entrypoint for ``uvicorn ingest.main:app``.

Built only when a DSN is present so merely importing this module (e.g. tooling,
IDE, a dependency graph scan) does not require live cloud credentials.
"""
