# Aletheia — Institutional memory for agent fleets

**Reliable memory despite unreliable agents.**

Built for the [CockroachDB × AWS Hackathon — *Build with Agentic Memory*](https://cockroachdb-ai.devpost.com/).

---

## The problem

A fleet of AI agents in production does not fail because of the model. It fails
because of **memory** — memory that is stale, memory that is poisoned, memory
that goes down.

- **Stale memory.** A runbook is revised; the fleet keeps answering with last
  month's fix. Knowledge-update is the open problem in agentic memory.
- **Poisoned memory.** One compromised or hallucinating agent writes a false
  fact, and every other agent inherits it. Hallucination contagion.
- **Memory that goes down.** Shared memory is a single point of failure for the
  whole fleet. If it stops, every agent stops.

## The solution

Aletheia is the institutional memory layer of a fleet. Agents write concurrently
what they learn; a **consolidation cycle** overwrites what is obsolete
(knowledge-update); **metabolic forgetting** keeps the token cost bounded; and an
**immune system** quarantines poisoned memory before it contaminates the fleet.

All of it on CockroachDB — because concurrent writes to shared memory demand
distributed serializable transactions and vectors with no consistency gap, and
because memory that goes down stops the entire fleet.

The demo will prove it with chaos engineering: we kill a node mid-consolidation
with 20 agents writing, and the fleet keeps remembering. *(The fleet, experiments
and demo app are under construction — see the project status table below.)*

## The five core components

| Component | What it does |
|---|---|
| **C1 Fast write + retrieval** | Immediate episodic writes; semantic retrieval under a fixed token budget |
| **C2 Consolidation** | Periodic job: groups episodes, detects revised facts, `supersede()`s the obsolete, promotes `canonical_facts` |
| **C3 Metabolic forgetting** | Every memory carries a cost; a global budget prunes low-value memory while holding recall |
| **C4 Gossip** | Agents share findings; information degrades as it propagates — and consolidation repairs it |
| **C5 Immune system** | Validates provenance at ingest, detects injection/poisoning, `quarantine()`s instead of deleting (auditable) |

## Architecture

Read path and write path are deliberately separated (least privilege). The read
and write services below are the Phase 1–2 deliverables; Phase 0 ships the
portable core, the schema, and the storage contract they build on.

- **Read:** agents → CockroachDB **Managed MCP Server** (service-account, read-only,
  RBAC + platform audit log). Agents never hold a database DSN.
- **Write:** agents → **Ingest service** (per-agent HMAC auth) → **immune gate** →
  a single serializable transaction in CockroachDB. The ingest service is the
  only holder of write credentials.

```
  FLEET (Bedrock Claude) ──read (MCP)──▶ CockroachDB Managed MCP Server ─┐
        │                                                                │
        └──write (HTTP+HMAC)──▶ Ingest service ──immune gate──▶──────────┴─▶ CockroachDB Cloud
                                                                             · memories (VECTOR + C-SPANN)
  Lambda: consolidation ─┐                                                   · canonical_facts
  Lambda: gossip tick ───┴─(EventBridge)──────────────────────────────────▶  · provenance / quarantine_log
  ccloud CLI (ops agent + chaos) ──▶ CockroachDB Cloud control plane         · experiment_runs
```

Full diagram: [`docs/architecture.md`](docs/architecture.md).

## CockroachDB tools used

| Tool | What the agent does with it |
|---|---|
| **Managed MCP Server** | The fleet's read path: semantic + relational memory queries under a read-only service account |
| **Distributed Vector Indexing (C-SPANN)** | Semantic memory: `VECTOR(1024)` embeddings indexed for retrieval, fresh immediately after insert/supersede |
| **ccloud CLI (agent-ready)** | Provisioning, backups, and the chaos experiment (kill a node under load) |
| **Agent Skills repo** | Operational skills mounted on the SRE/ops agent for cluster diagnosis via SQL |

## AWS services used

| Service | Role |
|---|---|
| **Amazon Bedrock** | Fleet models (Claude) + Titan embeddings for the vector memory |
| **AWS Lambda + EventBridge** | Consolidation cycle and gossip ticks, on a schedule |
| **Amazon S3** | Incident transcripts, experiment snapshots, archived (forgotten) memories |
| **AWS App Runner** | The public demo app |

## Results

Experiments are pre-registered in `CLAUDE.md` §8 (E1 concurrency, E2 chaos,
E3 knowledge-update, E4 poisoning, E5 cost). Tables R1–R3 are populated **only**
with real numbers from real runs; unrun cells say `pendiente`.

**E3b — external validation.** The knowledge-update result is replicated on a
stratified subset of [LongMemEval](https://github.com/xiaowu0162/LongMemEval),
restricted to the *knowledge-update* and *temporal-reasoning* categories. The
point is falsifiability: a recognised external benchmark shows the consolidation
effect is not an artefact of our own synthetic corpus. The exact question IDs are
committed in [`scenarios/longmemeval/SUBSET.md`](scenarios/longmemeval/SUBSET.md)
so the run is reproducible.

**Where the chaos experiment runs — stated plainly.** CockroachDB Cloud manages
nodes and does not expose node termination as a user operation, so E2 runs against
a **three-node CockroachDB cluster we operate ourselves**
([`docker-compose.chaos.yml`](docker-compose.chaos.yml)) rather than against the
managed cluster. Nothing about the failure is simulated: three real nodes, real
Raft replication with `num_replicas = 3`, and `docker kill` sends SIGKILL with no
drain — a harsher event than a managed platform would ever hand us. Every other
experiment runs against CockroachDB Cloud. We would rather explain an honest
substitution than stage a kill.

**Status: `pendiente` — experiments run in Phase 3.** See [`docs/results.md`](docs/results.md).

## Quickstart (local development)

Requires Python 3.12+ and Docker.

```bash
git clone <this repo> && cd aletheia
cp .env.example .env                  # fill in as needed; local defaults work offline

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

./infra/apply_ddl.sh                  # starts the container, creates db + schema

pytest                                # unit tests + architecture tests
python smoke.py --local               # end-to-end write/read smoke against local CRDB
```

The DB Console is at <http://localhost:8081> (8081 rather than CockroachDB's usual
8080, which is commonly already taken; override with `CRDB_UI_PORT`).

Against the cloud cluster with real Bedrock embeddings:

```bash
python smoke.py --cloud               # needs ALETHEIA_CRDB_DSN + AWS credentials
```

The three-node cluster used by the resilience experiment (E2):

```bash
docker compose -f docker-compose.chaos.yml up -d
./chaos/verify_cluster.sh             # assert 3 live nodes before claiming anything
docker kill aletheia-chaos-2          # a real node loss: SIGKILL, no drain
```

`smoke.py` has two real modes and no fake one. `--local` uses a seeded, documented
offline embedder so the loop can run without spending cloud credit; `--cloud` uses
Amazon Bedrock Titan. If a credential or service is missing, it fails loudly with
the remediation step rather than degrading to synthetic vectors. See `CLAUDE.md` §3.

What the local smoke run proves end to end: serializable isolation, `memories` and
`provenance` written in one transaction, 1024-dimension vectors round-tripping, the
C-SPANN index actually serving nearest-neighbour search (verified via `EXPLAIN`),
and `supersede` preserving the row instead of deleting it.

## Repository layout

```
core/         portable core — never imports boto3 or psycopg (enforced by a test)
adapters/     InMemoryAdapter (tests) + CockroachDBAdapter (psycopg + VECTOR)
ingest/       write service (FastAPI) with the immune gate
agents/       SRE fleet: loop, prompts, MCP client
scenarios/    simulated incidents, distributed clues, poison suite, LongMemEval subset
experiments/  runner, arms, seeds, scoring
chaos/        ccloud scripts for kill-node + integrity measurement
lambdas/      consolidation and gossip-tick handlers
demoapp/      FastAPI + fleet dashboard (public URL)
infra/        provisioning scripts and infra/ddl.sql
docs/         architecture and results
```

## Project status

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations: repo, CI, schema, frozen core contracts, smoke test | in progress |
| 1 | Memory core: CockroachDB adapter, consolidation, forgetting, ingest | pendiente |
| 2 | Fleet, gossip, immune system, scenarios | pendiente |
| 3 | Experiments E1–E6 + E3b external validation, results tables | pendiente |
| 4 | Public demo app, incl. the ablation wall | pendiente |
| 5 | Packaging and submission | pendiente |

## License

[Apache-2.0](LICENSE).
