# Aletheia — submission draft

> Draft of the Devpost submission text for the CockroachDB × AWS Hackathon
> ("Build with Agentic Memory"). Every claim here matches the repo's honest state:
> R1/R2 are real; R3/R3b, the public URL and the video are marked pending until a
> provisioned run fills them. Replace the `<…>` placeholders before submitting.
> Repo: https://github.com/manuelpenazuniga/aletheia · License: Apache-2.0.

---

## Elevator pitch (one line)

Institutional memory for AI agent fleets on CockroachDB — reliable memory despite
unreliable agents: it consolidates stale knowledge, quarantines poisoned writes,
and keeps the fleet remembering through a node kill.

## Inspiration

A fleet of AI agents in production does not fail because of the model. It fails
because of **memory** — memory that is stale, memory that is poisoned, memory that
goes down. A runbook is revised and the fleet keeps answering with last month's
fix (knowledge-update, the open problem in agentic memory). One compromised or
hallucinating agent writes a false fact and every other agent inherits it
(hallucination contagion). And shared memory is a single point of failure: if it
stops, every agent stops. We wanted the memory layer to be the *reliable* part of
an unreliable fleet.

## What it does

Aletheia is the institutional memory layer of a fleet. Agents write concurrently
what they learn; five components keep that memory trustworthy:

- **C1 — Fast write + budgeted retrieval.** Immediate episodic writes; semantic
  retrieval under a fixed token budget.
- **C2 — Consolidation.** A periodic cycle detects revised facts, `supersede()`s
  the obsolete version, and promotes a `canonical_fact` with a full version
  history (the "git of beliefs").
- **C3 — Metabolic forgetting.** Every memory carries a token cost; a global
  budget prunes low-value memory to S3 while holding recall.
- **C4 — Gossip.** Agents share findings; information degrades as it propagates,
  and consolidation repairs it.
- **C5 — Immune system.** Validates provenance at ingest, detects injection and
  poisoning, and `quarantine()`s instead of deleting — auditable, never destroyed.

All of it on CockroachDB, because concurrent writes to shared memory demand
distributed serializable transactions and vectors with no consistency gap — and
because memory that goes down stops the entire fleet. The public dashboard lets
you watch the fleet remember, search the vector memory, trace any memory's
provenance, flip each component off with a kill-switch, and launch a real attack
and watch the immune gate quarantine it.

## How we built it

Python 3.12 core with a strict portability boundary: `core/` imports no
infrastructure (enforced by a test), and storage/embeddings enter through
Protocols, so the same logic runs against an in-memory oracle or a live cluster.
Read path and write path are deliberately separated (least privilege): agents read
through CockroachDB's Managed MCP Server (read-only, RBAC + audit log) and never
hold a DSN; writes go through an ingest service (per-agent HMAC) and an immune gate
before a single serializable transaction. See `docs/architecture.md` for diagrams.

## CockroachDB tools used — and what the agent did with each

| Tool | What the agent does with it |
|---|---|
| **Managed MCP Server** | The fleet's read path — semantic + relational memory queries under a read-only service account, so every read is authenticated, authorized, and audited by the platform. |
| **Distributed Vector Indexing (C-SPANN)** | Semantic memory — `VECTOR(1024)` embeddings indexed for nearest-neighbour retrieval, fresh immediately after insert/supersede with no reindex step. The index is prefixed by `status` so filtered retrieval (active only) uses the index instead of full-scanning. |
| **ccloud CLI (agent-ready)** | Provisioning, backups, and the resilience experiment — the ops agent kills a node under load and measures continuity. |
| **Agent Skills repo** | Operational skills mounted on the ops agent for cluster diagnosis via SQL. |

The single serializable transaction is the core argument: `memories` (with its
embedding) and `provenance` are written together, so there is never a window in
which a vector exists without its operational row.

## AWS services used — and how

| Service | Role |
|---|---|
| **Amazon Bedrock** | Fleet models (Claude) drive the SRE agents; Titan v2 embeddings power the vector memory. |
| **AWS Lambda + EventBridge** | The consolidation cycle and gossip ticks run serverless on a schedule. |
| **Amazon S3** | Incident transcripts, experiment snapshots, and archived (forgotten) memories. |
| **AWS App Runner** | Hosts the public demo dashboard (containerized, one-command deploy). |

## Results (real, not promised)

Every number is generated from the `experiment_runs` table — never hand-typed —
and raw run rows are committed under `docs/experiment_data/` for traceability.

- **R1 — Concurrency.** Under 20 concurrent writers, the naive non-transactional
  baseline logs **~870–990 inconsistencies per 1000 writes** (lost updates,
  vector↔row divergence, dirty reads); CockroachDB `SERIALIZABLE` logs **0**, at a
  real latency cost. The correctness/latency trade-off is the point.
- **R2 — Chaos.** A real `docker kill` (SIGKILL) of a node, mid-write and
  mid-consolidation, lost **zero** acknowledged memories, corrupted nothing
  (checksum verified before and after), and the fleet kept writing through a
  **26 ms** blip — quorum carried it. The single-node baseline, given the same
  storm, stopped entirely.
- **R3 / R3b (knowledge-update, poisoning, cost, LongMemEval) — pending.** The
  harness is built and validated; these cells stay `pendiente` until a provisioned
  Bedrock run fills them. No number is invented.

**Where chaos runs, stated plainly:** CockroachDB Cloud does not expose node
termination, so R2 runs against a self-operated three-node cluster (real Raft
replication, `num_replicas = 3`, a real SIGKILL). We would rather explain an honest
substitution than stage a kill.

## Challenges we ran into

- **Filtered vector search.** A vector index on `(embedding)` alone full-scans a
  `WHERE status='active'` query; prefixing the index with `status` was needed to
  actually use the index for the retrieval the fleet issues.
- **Measuring inconsistency honestly.** Each phenomenon (lost updates, vector↔row
  divergence, dirty reads) had to be derived from its own source, not conflated —
  an external review caught an early double-count.
- **Deterministic demo.** Consolidation picks the newest fact by `(created_at,
  mem_id)`; microsecond-tied timestamps let a random UUID crown a stale version.
  Fixed with version-ordered timestamps so the demo is reproducible.

## Accomplishments we're proud of

Real R1/R2 numbers with an honest null policy; a portable core with 409 unit +
16 integration tests; a public dashboard that runs the *real* core cycles offline
(no mocks) including an interactive kill-switch and a launch-an-attack immune
panel; and a clean read/write privilege separation that answers "does the agent
use the tools safely?" literally.

## What we learned

Serializable isolation and vector freshness are not academic when a fleet writes
concurrently — the naive baseline visibly corrupts itself while CockroachDB does
not. And an honest null (R3 pending) reads as more credible than a suspiciously
perfect table.

## What's next

Provision CockroachDB Cloud + Bedrock to fill R3/R3b (knowledge-update, poisoning,
cost, and external LongMemEval validation), deploy the public URL, and run the
chaos experiment against CockroachDB Cloud's own disruption tooling.

## Links

- **Repository:** https://github.com/manuelpenazuniga/aletheia (Apache-2.0)
- **Demo app:** `<pending — App Runner URL once provisioned>`
- **Video (< 3 min):** `<pending — YouTube/Vimeo link>`
- **Architecture:** `docs/architecture.md` · **Results:** `docs/results.md`

## Requirements checklist (§2)

- [x] ≥2 CockroachDB tools — all four used (MCP Server, C-SPANN vector index, ccloud CLI, Agent Skills).
- [x] ≥1 AWS service — Bedrock, Lambda + EventBridge, S3, App Runner.
- [x] Public open-source repo with Apache-2.0 visible, README, setup/run instructions.
- [ ] Public demo URL — pending deployment (image + one-command deploy ready).
- [ ] Video < 3 min — pending.
- [x] Identify which CockroachDB tools were used and what the agent did with them (above).
- [x] Identify which AWS services and how (above).
- [x] Architecture diagram (`docs/architecture.md`).
