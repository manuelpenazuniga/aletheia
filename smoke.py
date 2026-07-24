"""End-to-end smoke test: write and read one memory, with a real embedding.

This is the Phase 0 acceptance criterion (CLAUDE.md §10). It proves the whole
vertical slice is wired: configuration -> embedder -> serializable transaction ->
`memories` + `provenance` -> vector search over the C-SPANN index -> read back.

    python smoke.py --local     # local docker CockroachDB + deterministic embedder
    python smoke.py --cloud     # CockroachDB Cloud + Amazon Bedrock (Titan)

Two modes, both real, neither faked:

* ``--local`` uses the seeded :class:`DeterministicEmbedder` — a real embedder,
  documented as such, so the loop can be exercised offline without spending
  cloud credit. It never touches Bedrock and never claims to.
* ``--cloud`` uses Amazon Bedrock Titan embeddings against the cloud cluster.
  This is the mode the acceptance criterion refers to. If credentials, region or
  model access are missing it fails loudly with the remediation step — it does
  not degrade to the deterministic embedder (CLAUDE.md §3.1, §3.2).

Rows written here are smoke fixtures, not fleet memory, and are removed at the
end unless ``--keep`` is passed. The "nothing is ever deleted" rule governs
memory semantics in the running system, not the fixtures of a connectivity test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from typing import Any

from core.config import AletheiaConfig
from core.embeddings import DeterministicEmbedder, Embedder
from core.models import MemoryEvent, MemoryKind, estimate_tokens

SMOKE_AGENT_ID = "smoke-agent"
SMOKE_CONTENT = (
    "Runbook db-latency: when p99 read latency exceeds 400ms on the primary shard, "
    "drain the node before restarting it; a bare restart re-triggers the stampede."
)
DECOY_CONTENT = (
    "Runbook cert-expiry: the gateway TLS certificate expired; rotate it via ACM "
    "and redeploy the listener."
)


class SmokeFailure(RuntimeError):
    """A smoke step did not produce the expected result."""


def redact(dsn: str) -> str:
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", dsn)


def step(message: str) -> None:
    print(f"[smoke] {message}", flush=True)


def to_vector_literal(vec: list[float]) -> str:
    """CockroachDB VECTOR input format."""
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


def build_embedder(args: argparse.Namespace, cfg: AletheiaConfig) -> tuple[Embedder, str]:
    if args.embedder == "bedrock":
        from adapters.bedrock_embedder import BedrockEmbedder

        embedder = BedrockEmbedder(
            model_id=cfg.embedding_model_id,
            dim=cfg.embedding_dim,
            region_name=args.region or os.environ.get("AWS_REGION"),
        )
        label = f"Amazon Bedrock {cfg.embedding_model_id} (real model, region {args.region or os.environ.get('AWS_REGION') or 'default'})"
        return embedder, label

    embedder = DeterministicEmbedder(dim=cfg.embedding_dim, seed=cfg.seed)
    return (
        embedder,
        f"DeterministicEmbedder(dim={cfg.embedding_dim}, seed={cfg.seed}) — offline, no Bedrock",
    )


def run(args: argparse.Namespace) -> int:
    import psycopg

    cfg = AletheiaConfig.from_env(seed=args.seed)
    dsn = args.dsn or os.environ.get("ALETHEIA_CRDB_DSN")
    if not dsn:
        raise SmokeFailure(
            "no DSN: set ALETHEIA_CRDB_DSN in .env (copy .env.example) or pass --dsn"
        )

    embedder, embedder_label = build_embedder(args, cfg)
    step(f"target   : {redact(dsn)}")
    step(f"embedder : {embedder_label}")

    # 1. Embed first, before touching the database: an unavailable model should
    #    fail before any row is written, leaving nothing behind to clean up.
    vectors = embedder.embed_batch([SMOKE_CONTENT, DECOY_CONTENT])
    for vec in vectors:
        if len(vec) != cfg.embedding_dim:
            raise SmokeFailure(
                f"embedder returned {len(vec)} dims, config expects {cfg.embedding_dim}"
            )
    step(f"embedded : 2 texts -> {cfg.embedding_dim}-dim vectors")

    # 2. Connect and report what we are actually talking to.
    with psycopg.connect(dsn, autocommit=True) as conn:
        version = conn.execute("SELECT version()").fetchone()[0]
        step(f"connected: {version.split(' (')[0]}")

        isolation = conn.execute("SHOW transaction_isolation").fetchone()[0]
        step(f"isolation: {isolation}")
        if isolation != "serializable":
            raise SmokeFailure(
                f"expected serializable isolation, got {isolation!r} — the whole "
                "concurrency argument of this project depends on it"
            )

        # 3. Register the smoke agent (the memories -> agents foreign key).
        conn.execute(
            """
            UPSERT INTO agents (agent_id, role, token_hash) VALUES (%s, %s, %s)
            """,
            (SMOKE_AGENT_ID, "sre", "smoke-not-a-real-token-hash"),
        )
        step(f"agent    : {SMOKE_AGENT_ID} registered")

        # 4. Write memory + provenance in ONE serializable transaction. This is
        #    the property that removes the vector/row consistency gap.
        written: list[str] = []
        with conn.transaction():
            for content, vec in zip([SMOKE_CONTENT, DECOY_CONTENT], vectors, strict=True):
                event = MemoryEvent(
                    agent_id=SMOKE_AGENT_ID,
                    content=content,
                    kind=MemoryKind.EPISODIC,
                    embedding=vec,
                    importance=0.8,
                )
                mem_id = conn.execute(
                    """
                    INSERT INTO memories
                        (agent_id, kind, content, embedding, importance, cost_tokens)
                    VALUES (%s, %s, %s, %s::VECTOR, %s, %s)
                    RETURNING mem_id
                    """,
                    (
                        event.agent_id,
                        str(event.kind),
                        event.content,
                        to_vector_literal(list(event.embedding)),
                        event.importance,
                        event.cost_tokens,
                    ),
                ).fetchone()[0]
                conn.execute(
                    """
                    INSERT INTO provenance (mem_id, parent_mem, agent_id, signature, hop_count)
                    VALUES (%s, NULL, %s, %s, 0)
                    """,
                    (mem_id, SMOKE_AGENT_ID, f"smoke-hmac-{uuid.uuid4().hex[:12]}"),
                )
                written.append(str(mem_id))
        step(f"wrote    : {len(written)} memories + provenance in one transaction")

        target_id = written[0]

        # 5. Read it back and verify the round-trip, vector included.
        row = conn.execute(
            """
            SELECT content, cost_tokens, status, vector_dims(embedding)
            FROM memories WHERE mem_id = %s
            """,
            (target_id,),
        ).fetchone()
        if row is None:
            raise SmokeFailure(f"memory {target_id} was not readable after commit")
        content, cost_tokens, status, dims = row
        if content != SMOKE_CONTENT:
            raise SmokeFailure("content did not round-trip intact")
        if dims != cfg.embedding_dim:
            raise SmokeFailure(f"stored vector has {dims} dims, expected {cfg.embedding_dim}")
        if cost_tokens != estimate_tokens(SMOKE_CONTENT):
            raise SmokeFailure("cost_tokens did not round-trip")
        step(
            f"read back: content ok, status={status}, vector_dims={dims}, cost_tokens={cost_tokens}"
        )

        # 6. Semantic retrieval exactly as the fleet performs it — filtered to
        #    active memories — plus proof that the C-SPANN index served it.
        #    idx_mem_embedding carries `status` as a prefix column precisely so
        #    this filtered query uses the index; without the prefix the same
        #    query plans as a FULL SCAN. `<=>` is cosine, matching the index
        #    opclass and InMemoryAdapter's scoring.
        query_vec = embedder.embed("p99 read latency is spiking on the primary database shard")
        search_sql = """
            SELECT mem_id, content, embedding <=> %s::VECTOR AS distance
            FROM memories
            WHERE status = 'active'
            ORDER BY distance
            LIMIT 10
        """
        plan = "\n".join(
            str(r[0])
            for r in conn.execute("EXPLAIN " + search_sql, (to_vector_literal(query_vec),))
        )
        used_index = "vector search" in plan and "idx_mem_embedding" in plan
        pruned = "prefix spans" in plan
        step(
            f"plan     : vector index {'USED' if used_index else 'NOT used (full scan)'}"
            + (", pruned to the active partition" if pruned else "")
        )
        if not used_index:
            raise SmokeFailure(
                "the query planner did not use idx_mem_embedding — the vector index "
                f"is not serving retrieval:\n{plan}"
            )
        if not pruned:
            raise SmokeFailure(
                "the vector index was used but not pruned by status — the prefix "
                f"column is not doing its job:\n{plan}"
            )

        hits = conn.execute(search_sql, (to_vector_literal(query_vec),)).fetchall()
        if not hits:
            raise SmokeFailure("vector search returned nothing")
        ranked = [str(h[0]) for h in hits]
        step(f"searched : {len(hits)} hits, nearest distance={hits[0][2]:.4f}")
        if target_id not in ranked:
            raise SmokeFailure(
                f"the written memory {target_id} was not retrievable by vector search"
            )
        if ranked.index(target_id) > ranked.index(written[1]):
            raise SmokeFailure(
                "vector search ranked the certificate decoy above the latency runbook"
            )
        step("ranking  : the latency runbook outranked the certificate decoy")

        # 7. Provenance is queryable — the audit path the immune system relies on.
        links = conn.execute(
            "SELECT agent_id, hop_count, parent_mem FROM provenance WHERE mem_id = %s",
            (target_id,),
        ).fetchall()
        if len(links) != 1 or links[0][1] != 0 or links[0][2] is not None:
            raise SmokeFailure(f"unexpected provenance for {target_id}: {links}")
        step("provenance: 1 root link, hop_count=0, direct observation")

        # 8. Status transition, not deletion.
        conn.execute(
            "UPDATE memories SET status = 'superseded', superseded_by = %s WHERE mem_id = %s",
            (written[1], target_id),
        )
        remaining = conn.execute(
            "SELECT count(*) FROM memories WHERE mem_id = %s AND status = 'superseded'",
            (target_id,),
        ).fetchone()[0]
        if remaining != 1:
            raise SmokeFailure("supersede did not preserve the row")
        step("supersede: row preserved with status='superseded'")

        # 8b. The knowledge-update property, end to end: the superseded runbook
        #     must disappear from retrieval *immediately*, with no reindex step.
        #     This is the whole reason `status` is a prefix column — the entry
        #     leaves the searched partition the moment the status changes.
        after = conn.execute(search_sql, (to_vector_literal(query_vec),)).fetchall()
        if target_id in [str(r[0]) for r in after]:
            raise SmokeFailure(
                "the superseded memory is still retrievable — knowledge-update is "
                "not reflected in the vector index"
            )
        step(f"freshness: superseded memory gone from search immediately ({len(after)} hits left)")

        summary: dict[str, Any] = {
            "server": version.split(" (")[0],
            "isolation": isolation,
            "embedder": embedder_label,
            "embedding_dim": cfg.embedding_dim,
            "memories_written": len(written),
            "nearest_distance": round(float(hits[0][2]), 6),
        }

        # 9. Clean up the fixtures.
        if args.keep:
            step(f"kept     : rows left in place for inspection (agent {SMOKE_AGENT_ID})")
        else:
            with conn.transaction():
                conn.execute("DELETE FROM provenance WHERE agent_id = %s", (SMOKE_AGENT_ID,))
                conn.execute("DELETE FROM memories WHERE agent_id = %s", (SMOKE_AGENT_ID,))
                conn.execute("DELETE FROM agents WHERE agent_id = %s", (SMOKE_AGENT_ID,))
            step("cleaned  : smoke fixtures removed")

    print(json.dumps(summary, indent=2))
    step("PASS — write and read of a real memory with a real embedding succeeded")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--local",
        action="store_true",
        help="local docker CockroachDB + deterministic offline embedder",
    )
    mode.add_argument(
        "--cloud",
        action="store_true",
        help="CockroachDB Cloud + Amazon Bedrock Titan embeddings (Phase 0 acceptance criterion)",
    )
    parser.add_argument("--dsn", default=None, help="overrides ALETHEIA_CRDB_DSN")
    parser.add_argument(
        "--embedder",
        choices=["deterministic", "bedrock"],
        default=None,
        help="defaults to deterministic with --local, bedrock with --cloud",
    )
    parser.add_argument("--region", default=None, help="AWS region for Bedrock")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", action="store_true", help="do not remove the smoke rows")
    args = parser.parse_args(argv)

    if not args.local and not args.cloud:
        parser.error("choose a mode: --local or --cloud")
    if args.embedder is None:
        args.embedder = "bedrock" if args.cloud else "deterministic"

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        pass

    try:
        return run(args)
    except RuntimeError as exc:
        # SmokeFailure and BedrockUnavailable both land here: a missing credential
        # or an unmet expectation is a loud, actionable failure — never a fallback.
        print(f"[smoke] FAIL — {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
