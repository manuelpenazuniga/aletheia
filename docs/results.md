# Results

> **Integrity policy (CLAUDE.md §3.1, §8.6).** Every cell in this document is
> populated only from a real run. Cells that have not been run say `pendiente`.
> The ▲/▼ markers are **predictions registered before running**. If a prediction
> does not hold, the real value is reported with a note; re-running until the
> number looks good, changing the metric after the fact, or omitting a row are
> all forbidden. An honest null gains credibility with this jury; a suspiciously
> perfect table loses it.
>
> **Status: no experiment has been run yet.** Experiments are Phase 3
> (5–10 August 2026). Everything below is the pre-registration.

Runs of record are written to the `experiment_runs` table in CockroachDB
(`arm`, `seed`, `metrics` JSONB), not to loose CSV files. This document is the
human-readable rendering of that table.

---

## R1 — Concurrency (E1)

**Hypothesis.** A naive vector store without distributed transactions exhibits
inconsistencies (lost updates, reads of intermediate state, vector↔row divergence)
under concurrent fleet writes; CockroachDB under `SERIALIZABLE` exhibits none.

Cells: mean ± σ over 3 repetitions.

| Backend | N agents | Inconsist./1000 wr | p95 write ms | p95 read ms |
|---|---:|---:|---:|---:|
| Naive (non-transactional) | 5 | pendiente | pendiente | pendiente |
| Naive | 20 | ▲ pendiente | pendiente | pendiente |
| Naive | 50 | ▲ pendiente | pendiente | pendiente |
| CockroachDB SERIALIZABLE | 5 | 0 * | pendiente | pendiente |
| CockroachDB SERIALIZABLE | 20 | 0 * | pendiente | pendiente |
| CockroachDB SERIALIZABLE | 50 | 0 * | pendiente | pendiente |

\* pre-registered prediction. If it is not 0, the real value is reported.

---

## R2 — Chaos (E2)

**Hypothesis.** Killing a node during the consolidation cycle, with 20 agents
writing, loses no confirmed memory and corrupts nothing; the single-node baseline
stops entirely.

**Where this runs, stated up front.** CockroachDB Cloud does not expose node
termination, so E2 runs against a self-operated three-node cluster
(`docker-compose.chaos.yml`): real nodes, real Raft replication, `num_replicas = 3`,
and `docker kill` (SIGKILL, no drain). Every other experiment runs against
CockroachDB Cloud. Decided 2026-07-24 under CLAUDE.md §11, which authorises this
substitution on the condition that it is stated openly — here, in the README and
in the video.

Integrity checksums are computed **before** the chaos event and verified after.

| Event | Writes in flight | Memories lost | Corruption | Recovery s | Fleet kept operating |
|---|---:|---:|---:|---:|---|
| kill node | pendiente | pendiente | pendiente | pendiente | pendiente |
| kill region | pendiente | pendiente | pendiente | pendiente | pendiente |
| baseline single-node, kill | pendiente | ALL (expected) | — | does not recover | no |

Note: whether a region kill is possible depends on the cluster plan. If the plan
does not allow it, that is documented here and in the video rather than simulated.

---

## R3 — Knowledge-update (E3), poisoning (E4), cost (E5)

Cells: mean ± σ over 5 seeds.

| Config | KU accuracy | % stale fact | Poison det. % | False pos. % | Tokens/query | Task success |
|---|---:|---:|---:|---:|---:|---:|
| A0 full | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| A1 −consolidation | ▼ pendiente | ▲ pendiente | — | — | — | pendiente |
| A4 −immune | — | — | ▼ pendiente | — | — | pendiente |
| A2 −forgetting | — | — | — | — | ▲ pendiente | pendiente |
| BL plain RAG | ▼ pendiente | ▲ pendiente | — | — | pendiente | pendiente |
| BL single-agent (E6) | — | — | — | — | — | ▼ pendiente |

---

## R3b — External validation on LongMemEval (E3b)

**Hypothesis.** The knowledge-update effect measured in E3 on our own synthetic
corpus **replicates on a public benchmark**. If it does not, E3 was measuring a
property of our dataset rather than of consolidation, and that is reported as
such — this table exists precisely so the claim can fail in public.

Subset: stratified sample of LongMemEval, *knowledge-update* and
*temporal-reasoning* categories only. Exact question IDs are pinned in
[`scenarios/longmemeval/SUBSET.md`](../scenarios/longmemeval/SUBSET.md); the
scoring adapter is `experiments/scoring_lme.py`. Both are placeholders today —
**the subset has not been selected and nothing has been run.**

Cells: mean ± σ over 5 seeds, scored by the benchmark's own criteria.

| Config | LME knowledge-update acc. | LME temporal-reasoning acc. | Replicates E3 direction? |
|---|---:|---:|---|
| A0 full | pendiente | pendiente | pendiente |
| A1 −consolidation | ▼ pendiente | pendiente | pendiente |
| BL plain RAG | ▼ pendiente | pendiente | pendiente |

Scope limits stated up front, so the reduction is auditable rather than quiet:
the full benchmark is out of budget; the excluded categories are named in
`SUBSET.md`; the subset is drawn once with a fixed seed and is not re-rolled
between arms or seeds. Per §11, E3b is cut before seeds are reduced from 5 to 3.

---

## Controls in force for every arm

1. `retrieval_budget_tokens` is identical across arms — otherwise Aletheia would
   win on retrieved volume rather than on design.
2. Same model, same temperature, same prompts; only the ablated flag or backend
   changes.
3. Identical scenarios (the same committed JSON) across arms.
4. E2 integrity checksums are computed before the chaos event, not eyeballed after.

## Phase 0 measurements (not an experiment)

These are setup verifications, recorded for traceability. They are not results and
do not appear in the tables above.

| Check | Value | How |
|---|---|---|
| Local server version | CockroachDB CCL v25.4.13 | `smoke.py --local` |
| Default isolation | `serializable` | `SHOW transaction_isolation` |
| Vector round-trip | 1024 dims | `vector_dims(embedding)` after commit |
| Vector index serving *filtered* search | yes, `prefix spans: [/'active' - /'active']` | `EXPLAIN` in `smoke.py` |
| Superseded memory leaves search | immediately, no reindex | `smoke.py` re-queries after supersede |
| Chaos cluster forms | 3 live nodes, `num_replicas = 3` | `chaos/verify_cluster.sh` |
| Survives a real node kill | yes — writes + search + supersede continue | `docker kill` then `smoke.py` |
| Unit tests | 106 passed | `pytest` |
| Cloud cluster | pendiente | requires provisioning |
| Bedrock embeddings | pendiente | requires AWS model access |
