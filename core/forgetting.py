"""C3 — metabolic forgetting: bounded memory cost.

Every ACTIVE memory carries a retention cost (``cost_tokens``). When the fleet's
retained active footprint exceeds ``AletheiaConfig.memory_budget_tokens`` this
cycle archives the lowest-value active memories — one at a time, cheapest first —
until the footprint is back under budget. Archiving is a status transition that
runs the injected S3 offload callback before flipping the row to ``archived``:
nothing is ever destroyed (CLAUDE.md §6a).

Value decays with disuse — the "metabolic" part: a memory the fleet keeps
retrieving stays alive; stale, unused, low-importance memory is shed first.

Portable core: stdlib only (math, datetime, dataclasses, logging). All
infrastructure reaches this module through the adapter — never an import. Gated
by ``AletheiaConfig.enable_forgetting``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

from core.adapter import StorageAdapter
from core.config import AletheiaConfig
from core.models import MemoryEvent, MemoryStatus, utcnow

_log = logging.getLogger("aletheia.forgetting")

#: Recency decay time-constant, in **seconds**. A memory's recency value halves
#: every this-many seconds of disuse. 7 days is provisional (short demo windows
#: cluster all ``last_accessed``, so recency barely differentiates and
#: importance/usage dominate); kept a parameter rather than promoted to the
#: frozen config so tuning it touches nothing else.
DEFAULT_RECENCY_HALFLIFE_SECONDS: float = 7 * 24 * 3600


def memory_value(
    mem: MemoryEvent,
    *,
    now: datetime,
    halflife_seconds: float = DEFAULT_RECENCY_HALFLIFE_SECONDS,
) -> float:
    """Retention value of a memory: higher = keep, lower = prune first.

    Three qualitatively different signals modulate one base worth, multiplicatively:

    * ``importance`` (in ``[0, 1]``) is the base worth. ``importance == 0`` drives
      the whole product to 0 → forgotten first, which is correct.
    * ``access_count`` via ``log1p`` is a usage multiplier (``>= 1``): a memory the
      fleet keeps retrieving is proven useful and expensive to lose. Diminishing
      returns, so 0→5 accesses matters more than 100→105.
    * recency via exponential half-life decay on ``last_accessed`` is the metabolic
      part: value literally decays with disuse. Using ``last_accessed`` (which
      ``query_semantic`` bumps on every hit), not ``created_at``, means a
      frequently-retrieved old memory stays fresh.
    """
    # Clamp: last_accessed in the future (clock skew) => age 0 => no decay,
    # never a negative age blowing up the exponent.
    age_seconds = max(0.0, (now - mem.last_accessed).total_seconds())
    recency_factor = 0.5 ** (age_seconds / halflife_seconds)
    usage_factor = 1.0 + math.log1p(mem.access_count)
    return mem.importance * usage_factor * recency_factor


@dataclass(frozen=True, slots=True)
class ForgettingResult:
    """Outcome of one forgetting cycle."""

    ran: bool  # False iff enable_forgetting is False
    budget: int  # config.memory_budget_tokens
    total_before: int  # active cost_tokens before
    total_after: int  # active cost_tokens after (<= budget when ran and satisfiable)
    archived_ids: list[str]  # mem_ids archived this run, in archive order (lowest value first)
    reclaimed_tokens: int  # sum of cost_tokens of archived_ids

    @property
    def archived_count(self) -> int:
        return len(self.archived_ids)


def run_forgetting(
    adapter: StorageAdapter,
    config: AletheiaConfig,
    *,
    now: datetime | None = None,
    halflife_seconds: float = DEFAULT_RECENCY_HALFLIFE_SECONDS,
) -> ForgettingResult:
    """Archive the fewest lowest-value active memories to get under budget.

    Injecting ``now`` (default :func:`~core.models.utcnow`) is what makes recency
    deterministic and testable. When ``enable_forgetting`` is False this is a true
    no-op: it makes no adapter call at all.
    """
    budget = config.memory_budget_tokens
    if not config.enable_forgetting:
        return ForgettingResult(
            ran=False,
            budget=budget,
            total_before=0,
            total_after=0,
            archived_ids=[],
            reclaimed_tokens=0,
        )

    now = now or utcnow()
    # Only ACTIVE memories count toward the footprint and are archivable. NOT
    # stats().total_cost_tokens — that sums superseded/quarantined/archived too.
    active = adapter.list_memories(status=MemoryStatus.ACTIVE)
    total_before = sum(m.cost_tokens or 0 for m in active)

    if total_before <= budget:
        return ForgettingResult(
            ran=True,
            budget=budget,
            total_before=total_before,
            total_after=total_before,
            archived_ids=[],
            reclaimed_tokens=0,
        )

    # Prune cheapest-first with a fully deterministic total order: value asc, then
    # last_accessed asc, created_at asc, mem_id asc. The 4-tuple makes ties
    # reproducible across process restarts regardless of input order.
    active_sorted = sorted(
        active,
        key=lambda m: (
            memory_value(m, now=now, halflife_seconds=halflife_seconds),
            m.last_accessed,
            m.created_at,
            m.mem_id or "",
        ),
    )

    remaining = total_before
    archived_ids: list[str] = []
    reclaimed = 0
    for mem in active_sorted:
        if remaining <= budget:
            break
        mem_id = mem.mem_id or ""
        # archive() runs the S3 offload callback BEFORE flipping status; if the
        # offload raises, the memory stays active and the exception propagates.
        # The job is idempotent: a re-run recomputes the active set and resumes.
        adapter.archive(mem_id)
        archived_ids.append(mem_id)
        reclaimed += mem.cost_tokens or 0
        remaining -= mem.cost_tokens or 0
        _log.info(
            "forget",
            extra={
                "agent_id": mem.agent_id,
                "mem_id": mem_id,
                "op": "forget",
                "reclaimed": reclaimed,
            },
        )

    return ForgettingResult(
        ran=True,
        budget=budget,
        total_before=total_before,
        total_after=remaining,
        archived_ids=archived_ids,
        reclaimed_tokens=reclaimed,
    )
