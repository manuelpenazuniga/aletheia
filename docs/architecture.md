# Aletheia — architecture

> Status: Phase 0 (foundations). Components marked *planned* land in Phases 1–2
> per the project plan §10.

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

This separation is the security spine of the design, not a detail. It is a
**design invariant realised in Phases 1–2**: the MCP read client and the ingest
write service are not part of the Phase 0 deliverable. Phase 0 ships the schema
and the portable core that make the separation enforceable — notably that `core/`
holds no DSN and the DSN lives only where the write service will.

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
* `EXPLAIN` confirms the vector index serves the **filtered** retrieval the fleet
  actually issues, pruned to the active partition:
  `• vector search  table: memories@idx_mem_embedding  prefix spans: [/'active' - /'active']`.
* A superseded memory leaves the search results **immediately**, with no reindex
  step — the C-SPANN freshness property, asserted in `smoke.py`.
* On the three-node chaos cluster: `docker kill` of one node (SIGKILL, no drain)
  leaves the node reported `is_live = false`, and writes, vector search and
  supersede all keep working through the surviving two. Smoke-level only — the
  measured version is E2 in Phase 3.

## 8. Resolved decisions

### 8.1 Filtered vector search — prefix columns *(decided 2026-07-24)*

CockroachDB routes a query through the vector index only when the query's filters
are covered by the index. Retrieval must exclude superseded, quarantined and
archived memories, so the index carries `status` as a **prefix column**:

```sql
CREATE VECTOR INDEX idx_mem_embedding ON memories (status, embedding vector_cosine_ops);
```

Measured against v25.4.13: with the index on `(embedding)` alone, `WHERE status =
'active' ORDER BY embedding <=> $1 LIMIT k` plans as a **full scan**; with the
prefix it plans as `vector search … prefix spans: [/'active' - /'active']` — the
index is used *and* only the active partition is searched. The alternative
considered and rejected was over-fetching unfiltered and post-filtering, which has
no lower bound on how many results survive the filter.

Decided during Phase 0, while `ddl.sql` was still editable under §6 rule (c); any
later change goes to `infra/migrations/`.

### 8.2 Distance operator — cosine, by contract *(decided 2026-07-24)*

`vector_cosine_ops` on the index, `<=>` in every query. `InMemoryAdapter` already
scores with cosine, so the two adapters now agree **by contract** rather than by
the coincidence that L2 and cosine rank unit-length vectors identically.

### 8.3 Where the chaos experiment runs *(decided 2026-07-24)*

CockroachDB Cloud manages nodes and does not expose node termination, so E2 runs
against a self-operated three-node cluster (`docker-compose.chaos.yml`) where a
kill is genuine. Authorised by the project plan §11, with its condition: **the video must
say so out loud**. See §11 below.

## 9. Still open

1. **Bedrock model ids.** `core/config.py` carries best-guess defaults; the exact
   ids and their availability in the target region must be confirmed in the
   Bedrock console before any experimental run (the project plan §11).

2. **Cluster version alignment.** Local development pins
   `CRDB_IMAGE_TAG=v25.4.13`; align it with whatever version CockroachDB Cloud
   provisions on the Basic plan.

3. **MCP Server availability on the Basic plan.** The read path depends on it. To
   be confirmed in the Cloud console during provisioning.

4. **Serialization-error retry (Phase 1).** The production `CockroachDBAdapter`
   must wrap writes in a retry loop on SQLSTATE `40001` — under concurrent writers
   some serializable conflicts require an application-level retry. The smoke
   fixture does not retry (single writer); the adapter will.

## 10. Deferred decisions (Phases 3–4)

Recorded here so they are not rediscovered late. Neither is implemented yet.

1. **E3b — external validation on LongMemEval** (the project plan §8.1, §8.4). The
   knowledge-update arms (`A0_full`, `A1_no_consolidation`, `BL_rag`) are replayed
   on a stratified subset of the public LongMemEval benchmark, restricted to the
   *knowledge-update* and *temporal-reasoning* categories — the full benchmark is
   out of budget. Two artefacts are required and currently exist as placeholders:
   `scenarios/longmemeval/SUBSET.md` (the exact question IDs, for reproducibility)
   and `experiments/scoring_lme.py` (the scoring adapter). Open question for
   Phase 3: how the benchmark's conversational sessions map onto `MemoryEvent`
   provenance, since LongMemEval has no notion of which agent observed what.

2. **Ablation wall — live fleets or synchronised replays** (the project plan §9.1.6,
   §10 Phase 4). A four-panel grid — `full`, `−consolidation`, `−immune`,
   `−forgetting` — running the same scenario under the same seed and diverging
   under the same events. Four live fleets is the stronger demonstration; four
   synchronised replays of real recorded runs is the cheaper one. The decision is
   deferred to Phase 4 and must be taken on the token cost actually measured in
   Phase 3, then **documented in the README and stated in the video** — a replay
   presented as a live fleet would be exactly the kind of disguised mock
   the project plan §3.2 forbids. Either way the wall reads the four ablations from the
   same feature flags in `AletheiaConfig`; no new configuration surface.

## 11. The chaos cluster (E2)

E2 is the thesis: kill a node mid-consolidation with 20 agents writing, and the
fleet keeps remembering. CockroachDB Cloud manages nodes and does not expose node
termination as a user operation, so the experiment runs against a cluster we
operate ourselves — `docker-compose.chaos.yml`, three real nodes, real Raft
replication, `num_replicas = 3` so quorum survives losing one.

**Nothing about the failure is simulated.** `docker kill` sends SIGKILL: no drain,
no graceful shutdown, the node simply stops. That is a harsher event than a
managed platform would ever give us.

```bash
docker compose -f docker-compose.chaos.yml up -d
./chaos/verify_cluster.sh                     # assert 3 live nodes before claiming anything
docker kill aletheia-chaos-2                  # the real event
```

Verified at smoke level on 2026-07-24: with node 2 killed and reported
`is_live = false`, writes, vector search, supersede and provenance all continued
to work through the surviving two nodes. The measured version — writes in flight,
memories lost, integrity checksums taken before the event and verified after,
recovery time — is E2 in Phase 3, and populates table R2.

**The condition attached to this choice is not optional** (the project plan §11): the
video and the README must state plainly that the kill happens on a self-operated
three-node cluster, and why. A viewer must never be left to assume it was the
managed cluster. Faking a kill would be disqualifying; explaining an honest
substitution costs nothing and, with this jury, is worth more than the illusion.
