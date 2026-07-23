# Aletheia — architecture

> Status: Phase 0 (foundations). Components marked *planned* land in Phases 1–2
> per CLAUDE.md §10.

## 1. The shape of the system

Aletheia is the shared memory layer of a fleet of agents. Everything is organised
around one asymmetry: **agents are unreliable, the memory must not be.**

```
                        ┌───────────────────────── AWS ─────────────────────────┐
                        │                                                        │
  ┌───────────┐  read   │  ┌──────────────┐        ┌────────────────────────┐   │
  │  AGENT     │─────────┼─▶│ MCP client    │───────▶│ CockroachDB Managed    │──┼──┐
  │  FLEET     │  (MCP)  │  │ (read-only)   │        │ MCP Server (RBAC,      │  │  │
  │  (Bedrock  │         │  └──────────────┘        │ audit log)             │  │  │
  │   Claude)  │         │                           └────────────────────────┘  │  │
  │            │  write  │  ┌──────────────────────────────────────────────┐    │  │
  │            │─────────┼─▶│ INGEST SERVICE (FastAPI)                      │    │  ▼
  └───────────┘         │  │  └─ IMMUNE GATE: provenance, anomaly,         │────┼─▶ CockroachDB Cloud
        ▲               │  │     injection patterns -> quarantine          │    │   · memories (VECTOR + C-SPANN)
        │ tasks          │  └──────────────────────────────────────────────┘    │   · canonical_facts
  ┌───────────┐         │  ┌───────────────┐   ┌───────────────┐                │   · provenance
  │ scenarios  │         │  │ Lambda:        │   │ Lambda:        │                │   · quarantine_log
  │ (SRE       │         │  │ consolidation  │   │ gossip tick    │                │   · experiment_runs
  │ incidents) │         │  │ (EventBridge)  │   │ (EventBridge)  │                │
  └───────────┘         │  └───────┬───────┘   └───────┬───────┘                │
                        │          └─────────┬─────────┘                         │
                        │                    ▼                                   │
                        │  ┌──────────────────────────────┐  ┌────────────────┐ │
                        │  │ DEMO APP (App Runner)         │  │ S3: transcripts│ │
                        │  │ fleet dashboard · kill-switch │  │ snapshots      │ │
                        │  │ chaos · immune · ablation wall│  │ archived memory│ │
                        │  └──────────────────────────────┘  └────────────────┘ │
                        └───────────────────────────────────────────────────────┘
  ccloud CLI (ops agent + chaos experiment) ──▶ CockroachDB Cloud control plane
```

## 2. Read path ≠ write path

This separation is the security spine of the design, not a detail.

| | Read | Write |
|---|---|---|
| Entry point | CockroachDB **Managed MCP Server** | **Ingest service** (FastAPI) |
| Auth | Service-account API key, read-only | Per-agent token, HMAC-signed payloads |
| Who holds the DSN | nobody but the ingest service | the ingest service |
| Enforcement | RBAC + platform audit log | Immune gate before the transaction |

The MCP server is read-only by default. Rather than treating that as a limitation
to work around, we treat it as least privilege: agents can read institutional
memory directly, but **no agent can write to the database without passing through
validation**. This is the concrete answer to "does the agent use the tools
correctly and safely?".

## 3. The five components

| | Component | Flag | Phase |
|---|---|---|---|
| C1 | Fast write + budgeted retrieval | — | 1 |
| C2 | Consolidation cycle (knowledge-update, canonical promotion) | `enable_consolidation` | 1 |
| C3 | Metabolic forgetting (token budget, archive to S3) | `enable_forgetting` | 1 |
| C4 | Gossip between agents (degradation by hop) | `enable_gossip` | 2 |
| C5 | Immune system (provenance, anomaly, injection → quarantine) | `enable_immune` | 2 |

Every component is behind its flag from its first commit. The experimental
ablations and the demo kill-switch are the same four booleans.

## 4. Life of a memory

1. An agent resolves a step of an incident and produces a `MemoryEvent`: content,
   embedding, and provenance signed with its token.
2. `POST` to the ingest service → **immune gate**: valid provenance? semantic
   anomaly against the agent's own history? injection pattern? → pass, or
   `quarantine`.
3. One serializable transaction writes the row in `memories` (embedding included)
   and the link in `provenance`. There is no window in which a vector exists
   without its operational row — that is the CockroachDB argument in one line.
4. Other agents find it through semantic retrieval over the MCP server, or
   receive it through a gossip tick.
5. The consolidation cycle detects that a fact was revised → `supersede(old, new)`
   → `canonical_facts` is bumped. The logical removal is reflected in the vector
   index immediately (C-SPANN freshness).
6. Metabolic forgetting moves low-value memory out of the budget: archived to S3,
   status `archived`, never destroyed.

## 5. Portability boundary

`core/` contains the memory logic and imports no infrastructure — no `boto3`, no
`psycopg`, no web framework. Storage enters through `core.adapter.StorageAdapter`;
embeddings through `core.embeddings.Embedder`; the S3 offload through an injected
callback. `tests/test_architecture.py` enforces this statically and at runtime.

```
core/  ──(Protocol)──▶  adapters/memory_inmem.py   (tests, reference oracle)
       ──(Protocol)──▶  adapters/cockroach.py      (production; Phase 1)
       ──(Protocol)──▶  core.embeddings.Embedder
                            ├── DeterministicEmbedder (offline, seeded)
                            └── adapters/bedrock_embedder.py (Titan V2)
```

## 6. Schema

See [`infra/ddl.sql`](../infra/ddl.sql). Two invariants the schema encodes:

* **Nothing is deleted.** `supersede`, `quarantine` and `archive` are status
  transitions. The history is the audit trail, and the audit trail is the product.
* **Multi-table writes are atomic.** `memories` and `provenance` are always
  written in the same transaction.

## 7. Verified in Phase 0

Against `cockroachdb/cockroach:v25.4.13` (local single node):

* Full schema applies, including `CREATE VECTOR INDEX idx_mem_embedding`.
* Default transaction isolation is `serializable`.
* 1024-dimension vectors round-trip through the `VECTOR` column.
* `EXPLAIN` confirms the vector index serves nearest-neighbour search
  (`• vector search  table: memories@idx_mem_embedding`).

## 8. Open decisions for Phase 1

1. **Filtered vector search.** CockroachDB routes a query through the vector index
   only when the query's filters are covered by the index. `idx_mem_embedding` is
   defined on `(embedding)` alone (CLAUDE.md §6), so

   ```sql
   SELECT ... FROM memories WHERE status = 'active' ORDER BY embedding <-> $1 LIMIT k
   ```

   plans as a **full scan**, while the same query without the `WHERE` clause uses
   the index. Two options, to be decided with the human before touching the frozen
   schema: (a) add prefix columns — `CREATE VECTOR INDEX ... ON memories (status,
   embedding)` — via a numbered migration; or (b) over-fetch from the index and
   filter afterwards. Option (a) is the production-grade answer; option (b) keeps
   §6 untouched at the cost of read amplification.

2. **Distance operator.** Retrieval currently uses `<->` (L2), which matches the
   default opclass of the vector index. `InMemoryAdapter` uses cosine distance.
   For unit-length embeddings (Titan normalises, and `DeterministicEmbedder` does
   too) the two induce the same ranking, but the adapters should agree explicitly
   rather than by coincidence.

3. **Bedrock model ids.** `core/config.py` carries best-guess defaults; the exact
   ids and their availability in the target region must be confirmed in the
   Bedrock console before any experimental run (CLAUDE.md §11).

4. **Cluster version alignment.** Local development pins
   `CRDB_IMAGE_TAG=v25.4.13`; align it with whatever version CockroachDB Cloud
   provisions.

## 9. Deferred decisions (Phases 3–4)

Recorded here so they are not rediscovered late. Neither is implemented yet.

1. **E3b — external validation on LongMemEval** (CLAUDE.md §8.1, §8.4). The
   knowledge-update arms (`A0_full`, `A1_no_consolidation`, `BL_rag`) are replayed
   on a stratified subset of the public LongMemEval benchmark, restricted to the
   *knowledge-update* and *temporal-reasoning* categories — the full benchmark is
   out of budget. Two artefacts are required and currently exist as placeholders:
   `scenarios/longmemeval/SUBSET.md` (the exact question IDs, for reproducibility)
   and `experiments/scoring_lme.py` (the scoring adapter). Open question for
   Phase 3: how the benchmark's conversational sessions map onto `MemoryEvent`
   provenance, since LongMemEval has no notion of which agent observed what.

2. **Ablation wall — live fleets or synchronised replays** (CLAUDE.md §9.1.6,
   §10 Phase 4). A four-panel grid — `full`, `−consolidation`, `−immune`,
   `−forgetting` — running the same scenario under the same seed and diverging
   under the same events. Four live fleets is the stronger demonstration; four
   synchronised replays of real recorded runs is the cheaper one. The decision is
   deferred to Phase 4 and must be taken on the token cost actually measured in
   Phase 3, then **documented in the README and stated in the video** — a replay
   presented as a live fleet would be exactly the kind of disguised mock
   CLAUDE.md §3.2 forbids. Either way the wall reads the four ablations from the
   same feature flags in `AletheiaConfig`; no new configuration surface.
