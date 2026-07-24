# Results

> **Integrity policy (the project plan §3.1, §8.6).** Every cell in this document is
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

**Result — prediction CONFIRMED.** Real run, 3 repetitions per cell, generated
from `experiment_runs` by `python -m experiments.make_tables` (never hand-typed).
Cells: mean ± σ.

| Backend | N agents | Inconsist./1000 wr | p95 write ms | p95 read ms |
|---|---:|---:|---:|---:|
| Naive (non-transactional) | 5 | 867 ± 37.7 | 1.44 ± 0.03 | 0.28 ± 0.04 |
| Naive | 20 | ▲ 987 ± 30.9 | 3.56 ± 2.63 | 0.33 ± 0.04 |
| Naive | 50 | ▲ 975 ± 4.99 | 3.47 ± 0.92 | 0.59 ± 0.09 |
| CockroachDB SERIALIZABLE | 5 | 0 ± 0 * | 13.4 ± 5.55 | 9.91 ± 4.53 |
| CockroachDB SERIALIZABLE | 20 | 0 ± 0 * | 39.3 ± 7.72 | 15.2 ± 1.58 |
| CockroachDB SERIALIZABLE | 50 | 0 ± 0 * | 107 ± 26.6 | 37.3 ± 19.8 |

\* pre-registered prediction of 0, confirmed. The naive baseline loses most
writes and diverges its vectors under contention; CockroachDB SERIALIZABLE is
inconsistency-free at a real latency cost (the correctness/latency trade-off is
the point). ▲ marks a pre-registered "naive grows with N" prediction — also held.

**Run provenance (honesty).** E1 is systemic (no LLM): it measures the storage
layer under concurrency, so writes are deterministic synthetic payloads (the
project plan §8.1) — Bedrock is not involved and would not change the result.
CockroachDB rows are from a real local `cockroachdb/cockroach:v25.4.13` cluster;
the naive figures are from the in-process non-transactional baseline. Re-running
against CockroachDB Cloud is a provisioning-gated repeat, not a new result.

---

## R2 — Chaos (E2)

**Hypothesis.** Killing a node during the consolidation cycle, with 20 agents
writing, loses no confirmed memory and corrupts nothing; the single-node baseline
stops entirely.

**Where this runs, stated up front.** CockroachDB Cloud does not expose node
termination, so E2 runs against a self-operated three-node cluster
(`docker-compose.chaos.yml`): real nodes, real Raft replication, `num_replicas = 3`,
and `docker kill` (SIGKILL, no drain). Every other experiment runs against
CockroachDB Cloud. Decided 2026-07-24 under the project plan §11, which authorises this
substitution on the condition that it is stated openly — here, in the README and
in the video.

**Result — prediction CONFIRMED.** Real run of `chaos/run_e2.py`. Integrity
checksums computed **before** the kill and verified after (§8.7 control 4).

| Event | Writes in flight | Memories lost | Corruption | Recovery s | Fleet kept operating |
|---|---:|---:|---:|---:|---|
| kill node | 3 | **0** | no | 0.026 | **yes** |
| kill region | pendiente | pendiente | pendiente | pendiente | pendiente |
| baseline single-node, kill | 0 | **0** * | no * | does not recover (12 s downtime) | **no** |

**What the numbers say.** With 20 agents writing and a consolidation cycle in
progress, a real `docker kill` (SIGKILL) of a CockroachDB node lost **zero**
acknowledged memories, corrupted nothing (the pre-kill committed set verified
byte-identical after, by checksum), and the fleet kept writing through the
failover with a **26 ms** blip — quorum (2 of 3) carried it. The single-node
baseline, given the same storm, **stopped entirely**: 1938 writes failed, no
failover, no auto-recovery, the fleet down for the whole outage.

\* Honest null (§8.6): the single-node baseline loses **availability**, not data.
Its rows return intact on restart (0 memories lost, checksum verified), so
"corruption: no". Claiming "all lost" would be false — what is lost is the fleet's
ability to write during the outage, which is the point.

**Caveats, stated openly.** (a) `kill region` stays `pendiente`: a single local
docker host cannot lose a region. (b) The killed node is not the one the writers
are connected to — CockroachDB survives losing a peer while clients stay on a
surviving node; client-side connection failover across hosts is a separate
feature. (c) E2 ran against the local 3-node cluster (`docker-compose.chaos.yml`),
not Cloud — CockroachDB Cloud does not expose node termination (the project plan
§11); re-running against Cloud's Disruption API is provisioning-gated.

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
| Unit tests | 380 passed (snapshot) | `pytest` |
| Cloud cluster | pendiente | requires provisioning |
| Bedrock embeddings | pendiente | requires AWS model access |
