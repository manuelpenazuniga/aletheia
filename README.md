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

We prove it with chaos engineering — and the numbers are real, not a promise:
with 20 agents writing and a consolidation cycle running, a real `docker kill` of
a CockroachDB node lost **zero** acknowledged memories and the fleet kept writing
through a 26 ms blip; a single-node baseline given the same storm **stopped
entirely**. Under concurrent load, the naive store loses ~90% of writes while
CockroachDB SERIALIZABLE loses **none**. See [Results](#results-real-numbers).

## The five core components

| Component | What it does |
|---|---|
| **C1 Fast write + retrieval** | Immediate episodic writes; semantic retrieval under a fixed token budget |
| **C2 Consolidation** | Periodic job: groups episodes, detects revised facts, `supersede()`s the obsolete, promotes `canonical_facts` |
| **C3 Metabolic forgetting** | Every memory carries a cost; a global budget prunes low-value memory while holding recall |
| **C4 Gossip** | Agents share findings; information degrades as it propagates — and consolidation repairs it |
| **C5 Immune system** | Validates provenance at ingest, detects injection/poisoning, `quarantine()`s instead of deleting (auditable) |

## Architecture

Read path and write path are deliberately separated (least privilege):

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

## Results (real numbers)

Experiments are **pre-registered** in [`docs/results.md`](docs/results.md) with
▲/▼ predictions declared before running. Every cell is populated **only** from a
real run, generated from the `experiment_runs` table by
`python -m experiments.make_tables` — never hand-typed. Unrun cells say
`pendiente`. Raw run rows are committed under
[`docs/experiment_data/`](docs/experiment_data/) so every number is traceable.

### R1 — Concurrency (E1) · prediction CONFIRMED

20 agents write shared memory under real contention; we count inconsistencies
(lost updates, vector↔row divergence, dirty reads) and latency. 3 reps per cell.

| Backend | N | Inconsist./1000 wr | p95 write ms |
|---|--:|--:|--:|
| Naive (non-transactional) | 5 / 20 / 50 | **867 / 987 / 975** | 1–3 |
| CockroachDB SERIALIZABLE | 5 / 20 / 50 | **0 / 0 / 0** | 13–107 |

The naive store loses most writes and diverges its vectors; CockroachDB is
inconsistency-free at a real latency cost — the correctness/latency trade-off is
the point.

### R2 — Chaos (E2) · prediction CONFIRMED

Real `docker kill` (SIGKILL) of a node, mid-write, mid-consolidation. A checksum
taken **before** the kill is verified after.

| Event | Memories lost | Corruption | Recovery | Fleet kept operating |
|---|--:|--:|--:|--:|
| **kill node (3-node CockroachDB)** | **0** | no | **26 ms** | **yes** |
| baseline single-node, kill | 0 \* | no | does not recover (12 s down) | **no** |

\* Honest null: the single node loses **availability**, not data — its rows return
intact on restart. Claiming "all lost" would be false. See
[`docs/results.md`](docs/results.md) for the full tables and run provenance.

### R3 / R3b — `pendiente`

E3 (knowledge-update), E4 (poisoning), E5 (cost) and the external
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) validation (E3b) need
real LLM inference (Amazon Bedrock). The harness is built and validated; those
cells stay `pendiente` until a provisioned run fills them — no number is invented.

**Where chaos runs — stated plainly.** CockroachDB Cloud does not expose node
termination, so E2 runs against a **three-node cluster we operate ourselves**
([`docker-compose.chaos.yml`](docker-compose.chaos.yml)): real nodes, real Raft
replication (`num_replicas = 3`), a real SIGKILL. We would rather explain an
honest substitution than stage a kill.

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
the remediation step rather than degrading to synthetic vectors. See the project plan §3.

What the local smoke run proves end to end: serializable isolation, `memories` and
`provenance` written in one transaction, 1024-dimension vectors round-tripping, the
C-SPANN index actually serving nearest-neighbour search (verified via `EXPLAIN`),
and `supersede` preserving the row instead of deleting it.

## The demo dashboard

The public dashboard (fleet view, live semantic search + provenance, the canonical
"git of beliefs", an interactive kill-switch/ablation wall, and a launch-an-attack
immune panel) is a container that runs standalone — no cloud credentials needed to
start. It serves a seeded offline world and says so (`mode: offline-demo`); the same
image serves live fleet memory once a CockroachDB + Bedrock adapter is injected.

```bash
docker build -t aletheia-demo .
docker run -p 8080:8080 aletheia-demo          # then open http://localhost:8080
```

Deploy to the public URL (AWS App Runner, image-based) once the account is ready:

```bash
AWS_REGION=us-east-1 ACCESS_ROLE_ARN=arn:aws:iam::<acct>:role/AppRunnerECRAccessRole \
  ALETHEIA_DEMO_TOKEN=<demo-secret> ./infra/deploy_demoapp.sh
```

The script builds for `linux/amd64`, pushes to ECR, and creates or updates the App
Runner service (health check on `/healthz`, port 8080). `ALETHEIA_DEMO_TOKEN`, when
set, gates the attack/destructive demo buttons.

## Repository layout

```
core/         portable core — never imports boto3 or psycopg (enforced by a test)
adapters/     InMemoryAdapter (oracle) + CockroachDBAdapter (psycopg + VECTOR) + Bedrock embedder
ingest/       write service (FastAPI): per-agent HMAC auth + immune gate
agents/       SRE fleet: loop, model/read/write seams, adversary, fleet runner
scenarios/    seeded incident corpus + labelled poison suite + loader
experiments/  runner, arms, baselines, scoring, make_tables (R1/R2/R3)
chaos/        3-node cluster + the E2 chaos measurement harness (run_e2.py)
lambdas/      consolidation and gossip-tick handlers
demoapp/      FastAPI + fleet dashboard (public URL)
infra/        provisioning scripts and infra/ddl.sql
docs/         architecture, results, experiment_data (raw run rows)
```

## Project status

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations: repo, CI, schema, frozen core contracts, smoke test | ✅ done |
| 1 | Memory core: CockroachDB adapter, consolidation, forgetting, ingest | ✅ done |
| 2 | Fleet, gossip, immune system, scenarios | ✅ done |
| 3 | Experiments — **R1 & R2 real**; R3/R3b harness ready, Bedrock-gated | ◑ R1/R2 done |
| 4 | Public demo app | ◑ in progress |
| 5 | Packaging and submission | pendiente |

380 unit tests + 16 integration tests (against a live CockroachDB) pass; `core/`
is import-clean (no boto3/psycopg), enforced by a test. The remaining work is
gated on provisioning CockroachDB Cloud + AWS Bedrock (R3, the live fleet, and
the public deployment).

## License

[Apache-2.0](LICENSE).
