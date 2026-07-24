"""The read seam (MCP read path) — a :class:`MemoryReader` Protocol + impls.

An agent's read path is separated from its write path (the project plan §5.3): it
recalls shared institutional memory through a read-only seam. In production that
seam is the managed CockroachDB MCP server (:class:`MCPMemoryReader`, an OAuth
service-account, read-only); offline and in tests it is :class:`LocalMemoryReader`
over a :class:`~core.memory.MemoryService`, which is the oracle the fleet tests
run against.

Both honour the same budget rule: retrieval is truncated to
``config.retrieval_budget_tokens`` and excludes superseded/quarantined/archived
memory. The reader never re-implements budgeting — it delegates to
``MemoryService.recall`` so every arm inherits identical truncation (§8.7).

Portable: this module speaks only :mod:`core` types; the MCP transport is imported
lazily inside the edge implementation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from core.memory import MemoryService
from core.models import MemoryHit


@runtime_checkable
class MemoryReader(Protocol):
    """Recall shared fleet memory relevant to a query, within the token budget."""

    def recall(
        self,
        agent_id: str,
        query: str,
        *,
        k: int | None = None,
        budget_tokens: int | None = None,
    ) -> list[MemoryHit]:
        """Return the memories most relevant to ``query`` that fit the budget."""
        ...


class LocalMemoryReader:
    """Offline :class:`MemoryReader` over an in-process :class:`MemoryService`.

    Delegates straight to :meth:`MemoryService.recall`, so the budget/exclusion
    rules (superseded/quarantined/archived are never returned) are honoured
    without duplication. This is the read oracle the fleet tests use.
    """

    def __init__(self, memory_service: MemoryService) -> None:
        self._service = memory_service

    def recall(
        self,
        agent_id: str,
        query: str,
        *,
        k: int | None = None,
        budget_tokens: int | None = None,
    ) -> list[MemoryHit]:
        return self._service.recall(agent_id, query, k=k, budget_tokens=budget_tokens)


class MCPMemoryReader:
    """Edge :class:`MemoryReader` over the managed CockroachDB MCP server.

    Read-only by construction (the project plan §5.3): the agent authenticates
    with an OAuth service-account and can only query, never write. The MCP
    transport is imported lazily so importing this module pulls no HTTP client;
    this class is never exercised by the default test run.

    Left as a documented seam: the concrete transport call is wired in
    ``agents`` Phase 2 production wiring, against the live MCP endpoint.
    """

    def __init__(self, client: Any, *, embedder: Any = None) -> None:
        # ``client`` is an already-constructed MCP session; constructing one pulls
        # the transport, so the caller (production wiring) owns that lazy import.
        self._client = client
        self._embedder = embedder

    def recall(
        self,
        agent_id: str,
        query: str,
        *,
        k: int | None = None,
        budget_tokens: int | None = None,
    ) -> list[MemoryHit]:  # pragma: no cover - requires a live MCP server
        raise NotImplementedError(
            "MCPMemoryReader is wired against the live managed MCP server in "
            "production; tests use LocalMemoryReader."
        )
