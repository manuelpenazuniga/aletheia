"""C1 — fast episodic write + budgeted semantic retrieval.

`MemoryService` is the thin, portable seam between an agent and storage. It owns
two operations:

* **remember** — embed content and write it (memory + provenance) through a
  :class:`~core.adapter.StorageAdapter`.
* **recall** — embed a query and return the most relevant memories that fit in a
  fixed token budget.

It holds no infrastructure: the embedder and the storage adapter are injected, so
the same service runs against the in-memory adapter with a deterministic embedder
(tests) and against CockroachDB with Bedrock (production). The retrieval budget is
read from :class:`~core.config.AletheiaConfig` and is identical across experiment
arms — the control that keeps the comparison about design, not context volume
(project plan §8.7).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.adapter import StorageAdapter
from core.config import AletheiaConfig
from core.embeddings import Embedder
from core.models import MemoryEvent, MemoryHit, MemoryKind

#: Default number of nearest neighbours to consider before the token budget
#: truncates the result. The budget, not k, is the real limit; k just bounds the
#: vector search fan-out.
DEFAULT_RETRIEVAL_K = 20


class MemoryService:
    """Write and retrieve institutional memory over an injected adapter."""

    def __init__(
        self,
        adapter: StorageAdapter,
        embedder: Embedder,
        config: AletheiaConfig | None = None,
    ) -> None:
        self._adapter = adapter
        self._embedder = embedder
        self._config = config or AletheiaConfig()
        if embedder.dim != self._config.embedding_dim:
            raise ValueError(
                f"embedder dimension {embedder.dim} != config.embedding_dim "
                f"{self._config.embedding_dim}; they must agree (single source of truth)"
            )

    # ------------------------------------------------------------------ writes
    def remember(
        self,
        agent_id: str,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.EPISODIC,
        importance: float = 0.5,
        parent_mem: str | None = None,
        hop_count: int = 0,
        signature: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        """Embed ``content`` and persist it. Returns the new ``mem_id``.

        The embedding is computed here (the trusted side), never supplied by the
        caller, so an agent cannot poison the index with an arbitrary vector.
        """
        vector = self._embedder.embed(content)
        event = MemoryEvent(
            agent_id=agent_id,
            content=content,
            kind=kind,
            embedding=vector,
            importance=importance,
            parent_mem=parent_mem,
            hop_count=hop_count,
            signature=signature,
            meta=dict(meta) if meta else {},
        )
        return self._adapter.write_episode(agent_id, event)

    # ------------------------------------------------------------------- reads
    def recall(
        self,
        agent_id: str,
        query: str,
        *,
        k: int | None = None,
        budget_tokens: int | None = None,
    ) -> list[MemoryHit]:
        """Return the memories most relevant to ``query`` that fit the budget.

        ``budget_tokens`` defaults to ``config.retrieval_budget_tokens`` and must
        stay identical across experiment arms; pass an override only for probing.
        """
        vector = self._embedder.embed(query)
        return self._adapter.query_semantic(
            agent_id,
            vector,
            k if k is not None else DEFAULT_RETRIEVAL_K,
            budget_tokens if budget_tokens is not None else self._config.retrieval_budget_tokens,
        )

    def recent(self, agent_id: str, k: int) -> list[MemoryEvent]:
        """The agent's ``k`` most recent active memories, newest first."""
        return self._adapter.read_recent(agent_id, k)
