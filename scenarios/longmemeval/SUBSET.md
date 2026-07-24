# LongMemEval subset — E3b external validation

> **Status: placeholder. The subset has not been selected yet.** Selection happens
> in Phase 3 (the project plan §10), after E3 closes on the synthetic corpus. Until then
> every table below says `pendiente`, and no LongMemEval number appears anywhere
> in `docs/results.md`.

## Why this exists

E3 measures knowledge-update on our own synthetic SRE incident corpus. A jury has
no reason to trust an effect measured only on a dataset we wrote ourselves. E3b
replays the same arms (`A0_full`, `A1_no_consolidation`, `BL_rag`) on a
**public, recognised benchmark**, so the claim becomes falsifiable by someone
else. If the effect does not replicate, that is reported as-is (the project plan §8.6).

## Scope

* Benchmark: **LongMemEval** (public). <https://github.com/xiaowu0162/LongMemEval>
* Categories included: **knowledge-update** and **temporal-reasoning** only.
* Categories excluded: everything else. Running the full benchmark is out of
  budget, and those two categories are the ones our consolidation cycle makes a
  claim about. This is a stated scope limit, not a quiet cherry-pick — the
  excluded categories are named here precisely so the reduction is auditable.
* Sampling: stratified within the two categories, with a fixed seed.

## Selected question IDs

Reproducibility requires the exact list, not a sampling procedure. It is
committed here once selected and is not re-rolled between arms or seeds.

| # | Question ID | Category | Notes |
|---|---|---|---|
| — | pendiente | — | subset not yet selected |

Selection parameters, to be filled in when the subset is drawn:

| Parameter | Value |
|---|---|
| Benchmark release / commit | pendiente |
| Sampling seed | pendiente |
| n (knowledge-update) | pendiente |
| n (temporal-reasoning) | pendiente |
| Selection script | pendiente |

## Open question for Phase 3

LongMemEval is organised as conversational sessions between a user and an
assistant; Aletheia's memory is organised around **which agent observed what**,
with signed provenance. The mapping from benchmark sessions onto `MemoryEvent`
provenance is not obvious and must be decided and documented before any number is
produced — a bad mapping would make the comparison meaningless rather than
merely unfavourable.

## Licensing

Check LongMemEval's license before committing any benchmark content into this
repository. If redistribution is not permitted, only the IDs and a download
script are committed — never the data itself.
