# LongMemEval subset — sample fixture (test only)

This is a filled-in analogue of `scenarios/longmemeval/SUBSET.md`, used only by
`tests/test_scoring_lme.py` to exercise the parser. It contains invented question
IDs, not real benchmark content.

## Selected question IDs

| # | Question ID | Category | Notes |
|---|---|---|---|
| 1 | ku_0001 | knowledge-update | runbook revised twice |
| 2 | ku_0002 | knowledge_update | config value changed |
| 3 | tr_0001 | temporal-reasoning | ordering across sessions |

Selection parameters:

| Parameter | Value |
|---|---|
| Benchmark release / commit | v1.0-abc1234 |
| Sampling seed | 7 |
| n (knowledge-update) | 2 |
| n (temporal-reasoning) | 1 |
| Selection script | scripts/select_lme_subset.py |
