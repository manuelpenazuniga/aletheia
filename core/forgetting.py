"""C3 — metabolic forgetting: bounded memory cost.

Every memory carries a retention cost; a global budget prunes low-value memory
while holding recall. Pruned memory is archived (to S3), never destroyed.

Placeholder: implemented in Phase 1 (CLAUDE.md §10). Gated by
``AletheiaConfig.enable_forgetting``.
"""
