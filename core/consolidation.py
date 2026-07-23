"""C2 — consolidation cycle: knowledge-update and canonical promotion.

Groups episodes, detects revised facts, ``supersede()``s what is obsolete and
promotes ``canonical_facts``. Runs as an AWS Lambda on an EventBridge schedule.

Placeholder: implemented in Phase 1 (CLAUDE.md §10). Gated by
``AletheiaConfig.enable_consolidation``.
"""
