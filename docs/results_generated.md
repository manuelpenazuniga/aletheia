<!-- GENERATED FILE — do not edit by hand.
Regenerate with: python -m experiments.make_tables
Source: experiment_runs
Reproducible by re-running against the same experiment_runs table state
(no generation timestamp is embedded, so identical DB state -> identical file).
This file carries NO predictions and no hardcoded numbers: every cell is
'pendiente' until a real run fills it (the project plan §3.1, §8.6). -->

# Results (generated)

Generated rendering of the `experiment_runs` table. The hand-written pre-registration with its up/down predictions lives in `docs/results.md`; this file reproduces none of them — a cell is `pendiente` until a real run fills it.

## R1 — Concurrency (E1)

Cells: mean ± sigma over repetitions.

| Backend | N agents | Inconsist./1000 wr | p95 write ms | p95 read ms |
|---|---:|---:|---:|---:|
| Naive (non-transactional) | 5 | pendiente | pendiente | pendiente |
| Naive | 20 | pendiente | pendiente | pendiente |
| Naive | 50 | pendiente | pendiente | pendiente |
| CockroachDB SERIALIZABLE | 5 | pendiente | pendiente | pendiente |
| CockroachDB SERIALIZABLE | 20 | pendiente | pendiente | pendiente |
| CockroachDB SERIALIZABLE | 50 | pendiente | pendiente | pendiente |

## R2 — Chaos (E2)

Integrity checksums computed before the event, verified after.

| Event | Writes in flight | Memories lost | Corruption | Recovery s | Fleet kept operating |
|---|---:|---:|---:|---:|---:|
| kill node | pendiente | pendiente | pendiente | pendiente | pendiente |
| kill region | pendiente | pendiente | pendiente | pendiente | pendiente |
| baseline single-node, kill | pendiente | pendiente | pendiente | pendiente | pendiente |

## R3 — Knowledge-update (E3), poisoning (E4), cost (E5)

Cells: mean ± sigma over seeds.

| Config | KU accuracy | % stale fact | Poison det. % | False pos. % | Tokens/query | Task success |
|---|---:|---:|---:|---:|---:|---:|
| A0 full | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| A1 -consolidation | pendiente | pendiente | — | — | — | pendiente |
| A4 -immune | — | — | pendiente | — | — | pendiente |
| A2 -forgetting | — | — | — | — | pendiente | pendiente |
| BL plain RAG | pendiente | pendiente | — | — | pendiente | pendiente |
| BL single-agent (E6) | — | — | — | — | — | pendiente |

## R3b — External validation on LongMemEval (E3b)

Cells: mean ± sigma over seeds, scored by the benchmark's own criteria.

| Config | LME knowledge-update acc. | LME temporal-reasoning acc. | Replicates E3 direction? |
|---|---:|---:|---:|
| A0 full | pendiente | pendiente | pendiente |
| A1 -consolidation | pendiente | pendiente | pendiente |
| BL plain RAG | pendiente | pendiente | pendiente |

> Audit: every run in experiment_runs mapped to a known results cell.
