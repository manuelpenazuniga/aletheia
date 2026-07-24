"""C2 — consolidation cycle: knowledge-update and canonical promotion.

Groups ACTIVE episodic memories by ``meta['fact_key']``, treats the newest member
of each group as the current truth, ``supersede()``s the older revisions, and
promotes the survivor into ``canonical_facts`` when its content is a genuine
knowledge-update. Runs as a stateless AWS Lambda on an EventBridge schedule.

The cycle is a pure data operation over stored memory: no embedder, no model
call, fully deterministic. Because superseded/quarantined/archived memories are
excluded from the candidate set, memories consolidated in a prior run do not
re-enter grouping — this is what makes the cycle idempotent and re-runnable.

Portable core: no infrastructure imports. Gated by
``AletheiaConfig.enable_consolidation``.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.adapter import StorageAdapter
from core.config import AletheiaConfig
from core.models import MemoryEvent, MemoryKind, MemoryStatus


@dataclass(frozen=True, slots=True)
class ConsolidationSummary:
    """Counts from one consolidation cycle."""

    groups: int  # fact_key groups examined (each has >=1 active episodic memory)
    supersedes: int  # memories transitioned ACTIVE -> SUPERSEDED this run
    canonical_updates: int  # canonical_facts created or version-bumped this run
    skipped: bool = False  # True iff disabled by config.enable_consolidation


def consolidate(adapter: StorageAdapter, config: AletheiaConfig) -> ConsolidationSummary:
    """Run one consolidation cycle; return the counts. See module docstring."""
    if not config.enable_consolidation:
        # No adapter mutation whatsoever when disabled.
        return ConsolidationSummary(groups=0, supersedes=0, canonical_updates=0, skipped=True)

    rows = adapter.list_memories(status=MemoryStatus.ACTIVE, kind=MemoryKind.EPISODIC)

    # Bucket by fact key. Only a non-empty string key groups a memory; anything
    # else (missing, None, empty, non-str) is ignored entirely.
    groups: dict[str, list[MemoryEvent]] = {}
    for mem in rows:
        key = mem.meta.get("fact_key")
        if isinstance(key, str) and key.strip() != "":
            groups.setdefault(key, []).append(mem)

    supersedes = 0
    canonical_updates = 0
    # Iterate keys in sorted order for deterministic processing/logging.
    for fact_key in sorted(groups):
        members = groups[fact_key]
        # Newest wins, deterministically: largest (created_at, mem_id). Matches the
        # oracle's own newest-first tie-break (read_recent/query_semantic).
        newest = max(members, key=lambda m: (m.created_at, m.mem_id or ""))

        # Supersede every obsolete member first, so canonical always sources from
        # the surviving ACTIVE memory. Exclude the newest strictly by mem_id: the
        # oracle raises ValueError on supersede(x, x).
        for mem in members:
            if mem.mem_id != newest.mem_id:
                adapter.supersede(mem.mem_id, newest.mem_id)
                supersedes += 1

        # Promote canonical only on a real knowledge-update. set_canonical bumps
        # version unconditionally, so guarding on content equality is what keeps
        # the cycle idempotent (no version inflation on unchanged content).
        current = adapter.get_canonical(fact_key)
        if current is None or current.content != newest.content:
            adapter.set_canonical(fact_key, newest.content, newest.mem_id)
            canonical_updates += 1

    return ConsolidationSummary(
        groups=len(groups),
        supersedes=supersedes,
        canonical_updates=canonical_updates,
        skipped=False,
    )
