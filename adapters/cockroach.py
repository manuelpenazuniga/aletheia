"""CockroachDB implementation of :class:`core.adapter.StorageAdapter`.

This is the production backend. It reproduces the *observable* behaviour of
:class:`adapters.memory_inmem.InMemoryAdapter` — the reference oracle — exactly,
while adding the two things the in-memory store cannot give: durable, distributed
storage and genuine serializable concurrency control. Every write of a memory and
its provenance happens in one serializable transaction, so a reader never sees a
vector without its operational row (CLAUDE.md §6 rule (b) — the CockroachDB
argument).

Design notes
------------
* **Connections.** ``psycopg_pool`` is not a dependency; instead each thread gets
  its own :class:`psycopg.Connection` via :class:`threading.local`, so the N
  concurrent agents of experiment E1 each own a connection. All live connections
  are tracked so :meth:`close` can release them.
* **Serialization retries.** CockroachDB surfaces a serialization conflict as
  SQLSTATE ``40001``; :meth:`_txn` retries such transactions with exponential
  backoff and jitter. No other error is retried — a unique/foreign-key violation
  is a real answer, translated to the adapter's typed error.
* **Vectors** are bound as the literal ``'[...]'`` string with a ``::VECTOR``
  cast (the smoke.py pattern) and read back via ``embedding::STRING``. Retrieval
  ranks with the cosine operator ``<=>`` so distances agree with the oracle's
  :func:`~adapters.memory_inmem.cosine_distance` by contract, not by coincidence.
* **meta round-trips** through the ``memories.meta`` JSONB column, so the
  consolidation cycle can key on ``meta['fact_key']`` against this backend exactly
  as it does against the in-memory oracle.

This module lives under ``adapters/`` and may import psycopg; ``core/`` may not
(enforced by tests/test_architecture.py).
"""

from __future__ import annotations

import contextlib
import random
import threading
import time
from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from core.adapter import (
    AdapterError,
    AgentNotRegistered,
    DuplicateMemory,
    MemoryNotFound,
)
from core.config import DEFAULT_EMBEDDING_DIM
from core.models import (
    CanonicalFact,
    MemoryEvent,
    MemoryHit,
    MemoryKind,
    MemoryStats,
    MemoryStatus,
    ProvenanceLink,
    QuarantineReason,
    QuarantineRecord,
)

# Column projection shared by every full-MemoryEvent read: memories LEFT JOIN
# provenance so parent_mem / signature / hop_count come back on the same row.
# `meta` (JSONB) round-trips — the consolidation cycle keys on meta['fact_key'].
_MEM_COLS = (
    "m.mem_id, m.agent_id, m.kind, m.content, m.embedding::STRING, "
    "m.importance, m.cost_tokens, m.status, m.superseded_by, m.created_at, "
    "m.last_accessed, m.access_count, m.meta, p.parent_mem, p.signature, p.hop_count"
)
_MEM_FROM = "FROM memories m LEFT JOIN provenance p ON p.mem_id = m.mem_id"


def to_vector_literal(vec: list[float]) -> str:
    """CockroachDB ``VECTOR`` input format — identical to smoke.py's helper."""
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


class CockroachDBAdapter:
    """Serializable, vector-indexed storage implementing the frozen §7 contract."""

    def __init__(
        self,
        dsn: str,
        *,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        archive_callback: Callable[[MemoryEvent], None] | None = None,
        max_retries: int = 5,
        retry_base_delay: float = 0.05,
        retry_max_delay: float = 1.0,
    ) -> None:
        """
        Args:
            dsn: PostgreSQL-wire connection string to the CockroachDB cluster.
            embedding_dim: enforced on every embedding write, so a dimension
                mismatch fails at the boundary instead of corrupting the index.
                Must equal the ``VECTOR(n)`` column width.
            archive_callback: invoked with the pre-flip snapshot when a memory is
                archived — the S3 offload seam, kept out of the core.
            max_retries: how many times a ``40001`` transaction is retried before
                surfacing an :class:`AdapterError`.
            retry_base_delay / retry_max_delay: exponential backoff bounds (s).
        """
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        self._archive_callback = archive_callback
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay

        self._local = threading.local()
        self._registry_lock = threading.Lock()
        self._connections: list[psycopg.Connection] = []

    # ------------------------------------------------------------- connections
    def _conn(self) -> psycopg.Connection:
        """Return this thread's live connection, opening/recovering it as needed.

        One connection per thread (via ``threading.local``) so concurrent agents
        never share a session. A cached connection that has been closed or left in
        a failed transaction state is discarded and reopened, so a prior error on
        this thread can never poison a later operation.
        """
        conn: psycopg.Connection | None = getattr(self._local, "conn", None)
        if conn is not None and not conn.closed:
            status = conn.info.transaction_status
            if status in (
                psycopg.pq.TransactionStatus.IDLE,
                psycopg.pq.TransactionStatus.INTRANS,
            ):
                return conn
            # ACTIVE / INERROR / UNKNOWN: unusable — drop and reopen.
            with contextlib.suppress(Exception):  # pragma: no cover - best-effort teardown
                conn.close()
        conn = psycopg.connect(self.dsn, autocommit=False)
        self._local.conn = conn
        with self._registry_lock:
            self._connections.append(conn)
        return conn

    def _txn(self, fn: Callable[[psycopg.Connection], Any]) -> Any:
        """Run ``fn`` inside a transaction, retrying only serialization failures.

        ``conn.transaction()`` opens a BEGIN/COMMIT block and rolls back on any
        exception. SQLSTATE ``40001`` (serialization) is retried with exponential
        backoff + jitter; everything else propagates to the caller, which maps it
        to a typed adapter error. Retries exhausted -> :class:`AdapterError`.
        """
        for attempt in range(self._max_retries + 1):
            conn = self._conn()
            try:
                with conn.transaction():
                    return fn(conn)
            except psycopg.errors.SerializationFailure:
                if attempt >= self._max_retries:
                    raise AdapterError(
                        f"serialization conflict not resolved after {self._max_retries} retries"
                    ) from None
                delay = min(
                    self._retry_max_delay,
                    self._retry_base_delay * (2**attempt),
                )
                time.sleep(delay * (0.5 + random.random()))
        # Unreachable: the loop either returns or raises. Present for type-checkers.
        raise AdapterError("transaction retry loop exited unexpectedly")  # pragma: no cover

    # --------------------------------------------------------- existence checks
    @staticmethod
    def _require_agent(conn: psycopg.Connection, agent_id: str) -> None:
        row = conn.execute("SELECT 1 FROM agents WHERE agent_id = %s", (agent_id,)).fetchone()
        if row is None:
            raise AgentNotRegistered(f"agent not registered: {agent_id!r}")

    @staticmethod
    def _require_memory(conn: psycopg.Connection, mem_id: str) -> None:
        row = conn.execute("SELECT 1 FROM memories WHERE mem_id = %s::UUID", (mem_id,)).fetchone()
        if row is None:
            raise MemoryNotFound(f"no such memory: {mem_id!r}")

    # --------------------------------------------------------------- rebuilders
    @staticmethod
    def _parse_vector(text: str | None) -> list[float] | None:
        if text is None:
            return None
        s = text.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        s = s.strip()
        if not s:
            return []
        return [float(part) for part in s.split(",")]

    def _row_to_event(self, row: tuple) -> MemoryEvent:
        (
            mem_id,
            agent_id,
            kind,
            content,
            embedding_str,
            importance,
            cost_tokens,
            status,
            superseded_by,
            created_at,
            last_accessed,
            access_count,
            meta,
            parent_mem,
            signature,
            hop_count,
        ) = row
        return MemoryEvent(
            agent_id=agent_id,
            content=content,
            kind=kind,
            embedding=self._parse_vector(embedding_str),
            importance=importance,
            cost_tokens=cost_tokens,
            status=status,
            mem_id=str(mem_id),
            superseded_by=str(superseded_by) if superseded_by is not None else None,
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=access_count,
            parent_mem=str(parent_mem) if parent_mem is not None else None,
            signature=signature,
            hop_count=hop_count if hop_count is not None else 0,
            meta=dict(meta) if meta else {},
        )

    def _check_dim(self, vec: Any) -> None:
        if len(vec) != self.embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch: got {len(vec)}, adapter expects "
                f"{self.embedding_dim} (see AletheiaConfig.embedding_dim)"
            )

    # ------------------------------------------------------------------ agents
    def register_agent(self, agent_id: str, role: str, token_hash: str) -> None:
        if not agent_id:
            raise ValueError("agent_id is required")

        def fn(conn: psycopg.Connection) -> None:
            # ON CONFLICT DO UPDATE keeps the original created_at, matching the
            # oracle which preserves it across re-registration.
            conn.execute(
                """
                INSERT INTO agents (agent_id, role, token_hash)
                VALUES (%s, %s, %s)
                ON CONFLICT (agent_id)
                DO UPDATE SET role = excluded.role, token_hash = excluded.token_hash
                """,
                (agent_id, role, token_hash),
            )

        self._txn(fn)

    # ------------------------------------------------------------------ writes
    def write_episode(self, agent_id: str, event: MemoryEvent) -> str:
        """Persist memory + embedding + provenance in one serializable txn."""
        if event.agent_id != agent_id:
            raise ValueError(
                f"agent_id mismatch: called with {agent_id!r}, event carries {event.agent_id!r}"
            )
        if event.embedding is not None:
            self._check_dim(event.embedding)

        embedding_literal = (
            to_vector_literal(list(event.embedding)) if event.embedding is not None else None
        )

        def fn(conn: psycopg.Connection) -> str:
            self._require_agent(conn, agent_id)
            # Validate the parent explicitly (oracle order) so an unknown parent
            # is MemoryNotFound rather than a generic provenance FK violation.
            if event.parent_mem is not None:
                self._require_memory(conn, event.parent_mem)

            if event.mem_id:
                row = conn.execute(
                    """
                    INSERT INTO memories
                        (mem_id, agent_id, kind, content, embedding, importance,
                         cost_tokens, status, meta)
                    VALUES (%s::UUID, %s, %s, %s, %s::VECTOR, %s, %s, %s, %s)
                    RETURNING mem_id
                    """,
                    (
                        event.mem_id,
                        agent_id,
                        str(event.kind),
                        event.content,
                        embedding_literal,
                        event.importance,
                        event.cost_tokens,
                        str(event.status),
                        Jsonb(dict(event.meta)),
                    ),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    INSERT INTO memories
                        (agent_id, kind, content, embedding, importance,
                         cost_tokens, status, meta)
                    VALUES (%s, %s, %s, %s::VECTOR, %s, %s, %s, %s)
                    RETURNING mem_id
                    """,
                    (
                        agent_id,
                        str(event.kind),
                        event.content,
                        embedding_literal,
                        event.importance,
                        event.cost_tokens,
                        str(event.status),
                        Jsonb(dict(event.meta)),
                    ),
                ).fetchone()
            mem_id = str(row[0])
            conn.execute(
                """
                INSERT INTO provenance
                    (mem_id, parent_mem, agent_id, signature, hop_count)
                VALUES (%s::UUID, %s::UUID, %s, %s, %s)
                """,
                (mem_id, event.parent_mem, agent_id, event.signature or "", event.hop_count),
            )
            return mem_id

        try:
            return self._txn(fn)
        except psycopg.errors.UniqueViolation:
            # PRIMARY KEY on memories (or the UNIQUE provenance.mem_id) collided:
            # overwriting would destroy a memory + its provenance. Never silent.
            raise DuplicateMemory(f"memory already exists: {event.mem_id!r}") from None

    def upsert_embedding(self, mem_id: str, vec: list[float], meta: dict) -> None:
        self._check_dim(vec)
        literal = to_vector_literal(list(vec))

        def fn(conn: psycopg.Connection) -> None:
            self._require_memory(conn, mem_id)
            if meta:
                # Merge into the existing JSONB, matching the oracle's meta.update.
                conn.execute(
                    "UPDATE memories SET embedding = %s::VECTOR, meta = meta || %s "
                    "WHERE mem_id = %s::UUID",
                    (literal, Jsonb(dict(meta)), mem_id),
                )
            else:
                conn.execute(
                    "UPDATE memories SET embedding = %s::VECTOR WHERE mem_id = %s::UUID",
                    (literal, mem_id),
                )

        self._txn(fn)

    # ------------------------------------------------------------------- reads
    def read_recent(self, agent_id: str, k: int) -> list[MemoryEvent]:
        if k <= 0:
            return []

        def fn(conn: psycopg.Connection) -> list[MemoryEvent]:
            self._require_agent(conn, agent_id)
            rows = conn.execute(
                f"""
                SELECT {_MEM_COLS} {_MEM_FROM}
                WHERE m.agent_id = %s AND m.status = 'active'
                ORDER BY m.created_at DESC, m.mem_id DESC
                LIMIT %s
                """,
                (agent_id, k),
            ).fetchall()
            return [self._row_to_event(r) for r in rows]

        return self._txn(fn)

    def query_semantic(
        self, agent_id: str, vec: list[float], k: int, budget_tokens: int
    ) -> list[MemoryHit]:
        self._check_dim(vec)
        if k <= 0 or budget_tokens <= 0:
            return []
        literal = to_vector_literal(list(vec))

        def fn(conn: psycopg.Connection) -> list[MemoryHit]:
            self._require_agent(conn, agent_id)
            # Distance-only ORDER keeps the C-SPANN vector plan (a secondary sort
            # key would force a full scan). status='active' is pruned by the
            # index's prefix column. The Python budget loop below mirrors the
            # oracle exactly: stop before the first hit that would overflow.
            rows = conn.execute(
                """
                SELECT m.mem_id, m.agent_id, m.content, m.kind, m.cost_tokens,
                       m.importance, m.status, m.created_at, m.meta, p.hop_count,
                       m.embedding <=> %s::VECTOR AS distance
                FROM memories m LEFT JOIN provenance p ON p.mem_id = m.mem_id
                WHERE m.status = 'active'
                ORDER BY distance
                LIMIT %s
                """,
                (literal, k),
            ).fetchall()

            hits: list[MemoryHit] = []
            spent = 0
            for (
                mem_id,
                mem_agent_id,
                content,
                kind,
                cost_tokens,
                importance,
                status,
                created_at,
                meta,
                hop_count,
                distance,
            ) in rows:
                cost = cost_tokens or 0
                if spent + cost > budget_tokens:
                    break
                spent += cost
                hits.append(
                    MemoryHit(
                        mem_id=str(mem_id),
                        agent_id=mem_agent_id,
                        content=content,
                        kind=MemoryKind(kind),
                        score=max(0.0, min(1.0, 1.0 - distance)),
                        distance=distance,
                        cost_tokens=cost,
                        importance=importance,
                        status=MemoryStatus(status),
                        created_at=created_at,
                        hop_count=hop_count if hop_count is not None else 0,
                        meta=dict(meta) if meta else {},
                    )
                )

            # Retrieval is an access — record it only for the hits that fit the
            # budget, exactly like the oracle, so forgetting sees the same signal.
            if hits:
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1, last_accessed = now()
                    WHERE mem_id = ANY(%s::UUID[])
                    """,
                    ([h.mem_id for h in hits],),
                )
            return hits

        return self._txn(fn)

    def get_memory(self, mem_id: str) -> MemoryEvent:
        """Fetch by id regardless of status — the audit path. (Extension.)"""

        def fn(conn: psycopg.Connection) -> MemoryEvent:
            row = conn.execute(
                f"SELECT {_MEM_COLS} {_MEM_FROM} WHERE m.mem_id = %s::UUID",
                (mem_id,),
            ).fetchone()
            if row is None:
                raise MemoryNotFound(f"no such memory: {mem_id!r}")
            return self._row_to_event(row)

        return self._txn(fn)

    def list_memories(
        self,
        *,
        status: MemoryStatus | str | None = MemoryStatus.ACTIVE,
        kind: MemoryKind | str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryEvent]:
        """Enumerate memories filtered by status/kind/agent — powers C2 and C3.

        Filters are conjunctive; ``None`` on an axis means "no filter". Returns
        events in deterministic ascending ``(created_at, mem_id)`` order, matching
        the oracle's twin method. (Extension beyond §7.)
        """
        want_status = None if status is None else str(MemoryStatus(status))
        want_kind = None if kind is None else str(MemoryKind(kind))

        clauses: list[str] = []
        params: list[Any] = []
        if want_status is not None:
            clauses.append("m.status = %s")
            params.append(want_status)
        if want_kind is not None:
            clauses.append("m.kind = %s")
            params.append(want_kind)
        if agent_id is not None:
            clauses.append("m.agent_id = %s")
            params.append(agent_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        def fn(conn: psycopg.Connection) -> list[MemoryEvent]:
            rows = conn.execute(
                f"""
                SELECT {_MEM_COLS} {_MEM_FROM}
                {where}
                ORDER BY m.created_at ASC, m.mem_id ASC
                """,
                tuple(params),
            ).fetchall()
            return [self._row_to_event(r) for r in rows]

        return self._txn(fn)

    # ------------------------------------------------------ status transitions
    def supersede(self, old_mem_id: str, new_mem_id: str) -> None:
        if old_mem_id == new_mem_id:
            raise ValueError("a memory cannot supersede itself")

        def fn(conn: psycopg.Connection) -> None:
            self._require_memory(conn, old_mem_id)
            self._require_memory(conn, new_mem_id)
            conn.execute(
                """
                UPDATE memories
                SET status = 'superseded', superseded_by = %s::UUID
                WHERE mem_id = %s::UUID
                """,
                (new_mem_id, old_mem_id),
            )

        self._txn(fn)

    def quarantine(self, mem_id: str, reason: str, detector: str, payload: dict) -> None:
        # Validate the reason BEFORE any DB mutation: an invalid reason must leave
        # the memory exactly as it was, with no quarantine_log row written.
        valid_reason = QuarantineReason(reason)

        def fn(conn: psycopg.Connection) -> None:
            self._require_memory(conn, mem_id)
            conn.execute(
                "UPDATE memories SET status = 'quarantined' WHERE mem_id = %s::UUID",
                (mem_id,),
            )
            conn.execute(
                """
                INSERT INTO quarantine_log (mem_id, reason, detector, payload)
                VALUES (%s::UUID, %s, %s, %s)
                """,
                (mem_id, str(valid_reason), detector, Jsonb(dict(payload or {}))),
            )

        self._txn(fn)

    def archive(self, mem_id: str) -> None:
        # Outbox ordering: snapshot -> offload (may raise, nothing committed) ->
        # flip status in a separate txn. A failed S3 upload must never leave a
        # memory hidden as 'archived' with nothing behind it.
        snapshot = self.get_memory(mem_id)
        if self._archive_callback is not None:
            self._archive_callback(snapshot)

        def fn(conn: psycopg.Connection) -> None:
            self._require_memory(conn, mem_id)
            conn.execute(
                "UPDATE memories SET status = 'archived' WHERE mem_id = %s::UUID",
                (mem_id,),
            )

        self._txn(fn)

    # --------------------------------------------------------------- canonical
    def get_canonical(self, key: str) -> CanonicalFact | None:
        def fn(conn: psycopg.Connection) -> CanonicalFact | None:
            row = conn.execute(
                """
                SELECT fact_key, content, version, source_mem, updated_at
                FROM canonical_facts WHERE fact_key = %s
                """,
                (key,),
            ).fetchone()
            if row is None:
                return None
            fact_key, content, version, source_mem, updated_at = row
            return CanonicalFact(
                fact_key=fact_key,
                content=content,
                version=version,
                source_mem=str(source_mem) if source_mem is not None else None,
                updated_at=updated_at,
            )

        return self._txn(fn)

    def set_canonical(self, key: str, content: str, source_mem: str) -> None:
        def fn(conn: psycopg.Connection) -> None:
            if source_mem is not None:
                self._require_memory(conn, source_mem)
            # Insert path -> version 1 (the column DEFAULT). Conflict path ->
            # current version + 1. Matches the oracle's increment semantics.
            conn.execute(
                """
                INSERT INTO canonical_facts (fact_key, content, version, source_mem, updated_at)
                VALUES (%s, %s, 1, %s::UUID, now())
                ON CONFLICT (fact_key) DO UPDATE SET
                    content = excluded.content,
                    version = canonical_facts.version + 1,
                    source_mem = excluded.source_mem,
                    updated_at = now()
                """,
                (key, content, source_mem),
            )

        self._txn(fn)

    # -------------------------------------------------------------- provenance
    def provenance_chain(self, mem_id: str) -> list[ProvenanceLink]:
        """Walk from the memory back to its root observation, nearest first."""

        def fn(conn: psycopg.Connection) -> list[ProvenanceLink]:
            self._require_memory(conn, mem_id)
            chain: list[ProvenanceLink] = []
            seen: set[str] = set()
            cursor: str | None = mem_id
            while cursor is not None and cursor not in seen:
                seen.add(cursor)
                row = conn.execute(
                    """
                    SELECT link_id, mem_id, parent_mem, agent_id, signature,
                           hop_count, created_at
                    FROM provenance WHERE mem_id = %s::UUID
                    """,
                    (cursor,),
                ).fetchone()
                if row is None:
                    break
                link_id, link_mem, parent_mem, link_agent, signature, hop_count, created_at = row
                chain.append(
                    ProvenanceLink(
                        mem_id=str(link_mem),
                        agent_id=link_agent,
                        signature=signature,
                        link_id=str(link_id) if link_id is not None else None,
                        parent_mem=str(parent_mem) if parent_mem is not None else None,
                        hop_count=hop_count,
                        created_at=created_at,
                    )
                )
                cursor = str(parent_mem) if parent_mem is not None else None
            return chain

        return self._txn(fn)

    def quarantine_log(self) -> list[QuarantineRecord]:
        """Full rejection feed, newest last — powers the immune panel. (Extension.)"""

        def fn(conn: psycopg.Connection) -> list[QuarantineRecord]:
            rows = conn.execute(
                """
                SELECT q_id, mem_id, reason, detector, payload, created_at
                FROM quarantine_log ORDER BY created_at, q_id
                """
            ).fetchall()
            records: list[QuarantineRecord] = []
            for q_id, mem_id, reason, detector, payload, created_at in rows:
                records.append(
                    QuarantineRecord(
                        reason=QuarantineReason(reason),
                        detector=detector,
                        mem_id=str(mem_id) if mem_id is not None else None,
                        payload=dict(payload or {}),
                        q_id=str(q_id) if q_id is not None else None,
                        created_at=created_at,
                    )
                )
            return records

        return self._txn(fn)

    # -------------------------------------------------------------------- misc
    def stats(self, agent_id: str | None = None) -> MemoryStats:
        def fn(conn: psycopg.Connection) -> MemoryStats:
            if agent_id is None:
                rows = conn.execute(
                    """
                    SELECT status, count(*), coalesce(sum(cost_tokens), 0),
                           min(created_at), max(created_at)
                    FROM memories GROUP BY status
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT status, count(*), coalesce(sum(cost_tokens), 0),
                           min(created_at), max(created_at)
                    FROM memories WHERE agent_id = %s GROUP BY status
                    """,
                    (agent_id,),
                ).fetchall()

            by_status: dict[MemoryStatus, int] = dict.fromkeys(MemoryStatus, 0)
            total = 0
            total_cost = 0
            oldest = None
            newest = None
            for status, count, cost_sum, min_ts, max_ts in rows:
                by_status[MemoryStatus(status)] += count
                total += count
                total_cost += cost_sum or 0
                if min_ts is not None:
                    oldest = min_ts if oldest is None else min(oldest, min_ts)
                if max_ts is not None:
                    newest = max_ts if newest is None else max(newest, max_ts)

            # canonical_facts is fleet-wide, unscoped — matches the oracle's
            # len(self._canonical) regardless of agent scope.
            canonical = conn.execute("SELECT count(*) FROM canonical_facts").fetchone()[0]

            return MemoryStats(
                agent_id=agent_id,
                total_memories=total,
                active=by_status[MemoryStatus.ACTIVE],
                superseded=by_status[MemoryStatus.SUPERSEDED],
                quarantined=by_status[MemoryStatus.QUARANTINED],
                archived=by_status[MemoryStatus.ARCHIVED],
                total_cost_tokens=total_cost,
                canonical_facts=canonical,
                oldest_created_at=oldest,
                newest_created_at=newest,
            )

        return self._txn(fn)

    def close(self) -> None:
        """Close every connection this adapter opened, across all threads."""
        with self._registry_lock:
            connections = list(self._connections)
            self._connections.clear()
        for conn in connections:
            with contextlib.suppress(Exception):  # pragma: no cover - best-effort teardown
                conn.close()
        # Drop this thread's cached handle so a reused adapter reopens cleanly.
        if getattr(self._local, "conn", None) is not None:
            self._local.conn = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CockroachDBAdapter(dim={self.embedding_dim}, conns={len(self._connections)})"

    def __enter__(self) -> CockroachDBAdapter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
