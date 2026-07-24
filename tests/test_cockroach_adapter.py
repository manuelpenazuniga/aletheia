"""Integration tests: the CockroachDB adapter against a live cluster.

The whole module skips unless ``ALETHEIA_CRDB_DSN`` is set, so the default (no
credentials) test run stays green and CI never touches a database. When a DSN is
present, every test runs the shared behavioural contract
(:mod:`tests.adapter_contract`) with a fresh :class:`InMemoryAdapter` oracle
alongside the live :class:`CockroachDBAdapter`, proving they are observably
identical.

Point it at the local dev cluster to run them:

    ALETHEIA_CRDB_DSN='postgresql://root@localhost:26257/aletheia?sslmode=disable' \\
        ./.venv/bin/python -m pytest -q tests/test_cockroach_adapter.py

Isolation: the memory tables are cleaned before each test and torn down after, so
fleet-wide queries (which are not agent-scoped by design) compare against an
oracle holding exactly the same rows. The "nothing is deleted" rule governs live
fleet memory, not the fixtures of an integration test.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("ALETHEIA_CRDB_DSN"),
        reason="ALETHEIA_CRDB_DSN not set — integration tests need a live cluster",
    ),
]

# Imported lazily-at-module-level: psycopg is a declared dependency, so importing
# it is safe; connecting is deferred to the fixtures, which the skip guards.
import psycopg  # noqa: E402

from adapters.cockroach import CockroachDBAdapter  # noqa: E402
from adapters.memory_inmem import InMemoryAdapter  # noqa: E402
from core.config import DEFAULT_EMBEDDING_DIM  # noqa: E402
from core.embeddings import DeterministicEmbedder  # noqa: E402
from tests import adapter_contract as contract  # noqa: E402

DIM = DEFAULT_EMBEDDING_DIM  # 1024 — the production VECTOR width
_AGENT_PREFIX = "crdb-test-"


def _clean(dsn: str) -> None:
    """Empty the mutable tables so fleet-wide queries have a known universe.

    Child tables first, then break the ``memories.superseded_by`` self-reference,
    then the memories themselves; test agents last. Autocommit: this is fixture
    plumbing, not a memory-layer transaction.
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("DELETE FROM quarantine_log")
        conn.execute("DELETE FROM canonical_facts")
        conn.execute("DELETE FROM provenance")
        conn.execute("UPDATE memories SET superseded_by = NULL")
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM agents WHERE agent_id LIKE %s", (_AGENT_PREFIX + "%",))


class _Live:
    """Bundle passed to each test: paired backend + oracle on the same agent."""

    def __init__(self, adapter: CockroachDBAdapter, oracle: InMemoryAdapter, dsn: str) -> None:
        self.adapter = adapter
        self.oracle = oracle
        self.dsn = dsn
        self.embedder = DeterministicEmbedder(dim=DIM, seed=0)
        self.agent_id = f"{_AGENT_PREFIX}{uuid.uuid4().hex[:12]}"
        self.register(self.agent_id)

    def register(self, agent_id: str) -> None:
        self.adapter.register_agent(agent_id, "sre", "hash")
        self.oracle.register_agent(agent_id, "sre", "hash")

    def make_adapter(self, **kwargs) -> CockroachDBAdapter:
        return CockroachDBAdapter(self.dsn, embedding_dim=DIM, **kwargs)


@pytest.fixture
def live():
    dsn = os.environ["ALETHEIA_CRDB_DSN"]
    _clean(dsn)
    adapter = CockroachDBAdapter(dsn, embedding_dim=DIM)
    oracle = InMemoryAdapter(embedding_dim=DIM)
    bundle = _Live(adapter, oracle, dsn)
    try:
        yield bundle
    finally:
        adapter.close()
        oracle.close()
        _clean(dsn)


@pytest.fixture
def sleep():
    """A tiny pause so sequential writes get strictly increasing timestamps."""
    return lambda: time.sleep(0.002)


# --------------------------------------------------------------------------- #
# Contract-driven tests: the oracle is the source of truth.
# --------------------------------------------------------------------------- #
def test_write_creates_root_provenance(live):
    contract.contract_write_creates_root_provenance(live.adapter, live.embedder, live.agent_id)


def test_write_rejections_leave_no_row(live):
    contract.contract_write_rejections_leave_no_row(live.adapter, live.embedder, live.agent_id)


def test_duplicate_mem_id(live):
    contract.contract_duplicate_mem_id(live.adapter, live.embedder, live.agent_id)


def test_query_semantic_matches_oracle(live):
    contract.contract_query_semantic_matches_oracle(
        live.adapter, live.oracle, live.embedder, live.agent_id
    )


def test_query_budget_truncation(live):
    contract.contract_query_budget_truncation(
        live.adapter, live.oracle, live.embedder, live.agent_id
    )


def test_read_recent_excludes_non_active(live, sleep):
    contract.contract_read_recent_excludes_non_active(
        live.adapter, live.oracle, live.embedder, live.agent_id, sleep
    )


def test_supersede_freshness(live):
    contract.contract_supersede_freshness(live.adapter, live.embedder, live.agent_id)


def test_quarantine(live):
    contract.contract_quarantine(live.adapter, live.embedder, live.agent_id)


def test_archive(live):
    contract.contract_archive(live.adapter, live.embedder, live.agent_id, live.make_adapter)


def test_set_canonical_versions(live):
    contract.contract_set_canonical_versions(
        live.adapter, live.oracle, live.embedder, live.agent_id
    )


def test_provenance_chain_multi_hop(live):
    contract.contract_provenance_chain_multi_hop(
        live.adapter, live.oracle, live.embedder, live.agent_id
    )


def test_stats_match_oracle(live):
    contract.contract_stats_match_oracle(live.adapter, live.oracle, live.embedder, live.agent_id)


def test_list_memories_match_oracle(live, sleep):
    contract.contract_list_memories_match_oracle(
        live.adapter, live.oracle, live.embedder, live.agent_id, sleep
    )


# --------------------------------------------------------------------------- #
# Concurrency: the 40001 retry path must survive real contention.
# --------------------------------------------------------------------------- #
def test_concurrent_writes_do_not_lose_or_duplicate(live):
    threads_n = 4
    per_thread = 10
    errors: list[BaseException] = []
    barrier = threading.Barrier(threads_n)

    def worker(t: int) -> None:
        barrier.wait()  # maximise overlap
        try:
            for i in range(per_thread):
                live.adapter.write_episode(
                    live.agent_id,
                    contract._event(live.embedder, live.agent_id, f"concurrent t{t} i{i}"),
                )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(threads_n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, errors
    assert live.adapter.stats(live.agent_id).active == threads_n * per_thread
