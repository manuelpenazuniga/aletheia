# Scenarios — seeded, committed datasets

These datasets are the experimental substrate for the SRE fleet. They are
**generated with a fixed seed and committed** (never regenerated per run), so
every experiment arm sees identical inputs and results are reproducible offline.
Read them only through `scenarios/loader.py`, which validates every record
against the schema and returns typed, frozen dataclasses.

## Layout

| Path | What | Consumed by |
|---|---|---|
| `incidents/sample.json` | 10 SRE incidents, each with a runbook **and ≥1 revision** | E3 knowledge-update, C2 consolidation |
| `poison/bad_provenance.json` | ≥30 unsigned / over-propagated / dangling-parent writes | E4, C5 immune |
| `poison/semantic_anomaly.json` | ≥30 off-topic writes, each paired with a baseline history | E4, C5 immune |
| `poison/injection_pattern.json` | ≥30 prompt-injection / tool-call-spoof writes | E4, C5 immune |
| `poison/legit_unusual.json` | ≥30 rare-but-valid memories (false-positive control) | E4, C5 immune |
| `distributed_clues/sample.json` | tiny stub for E6 (fleet vs single-agent) | E6 (low priority) |

## Schemas

### Incident (`incidents/*.json`)

```json
{
  "id": "inc-0001",
  "title": "…",
  "category": "db-latency | disk-full | bad-deploy | cert-expired | …",
  "symptoms": ["…", "…"],
  "fact_key": "runbook:<category>:fix",
  "runbook": {"version": 1, "content": "…", "authored_by": "sre-01"},
  "revisions": [
    {"version": 2, "supersedes": 1, "content": "…", "reason": "…"}
  ]
}
```

Runbook `version` is always `1`; revision `version` strictly increases from `2`
and each `supersedes` the immediately prior version. That runbook-plus-revision
chain **is** the knowledge-update signal E3 measures and consolidation records
via `supersede()`. `fact_key` matches the `canonical_facts` key convention so a
test can drive `set_canonical()` directly.

### PoisonCase (`poison/*.json`)

```json
{
  "id": "bad_provenance-0000",
  "category": "bad_provenance | semantic_anomaly | injection_pattern | legit",
  "label": "attack | legit",
  "agent_id": "adversary-01",
  "content": "…",
  "provenance": {"parent_mem": null, "hop_count": 0, "signature": "…"},
  "expected_detector": "provenance_validator | injection_scanner | semantic_anomaly_detector | none",
  "baseline_ref": ["…"],
  "notes": "…"
}
```

The file basename equals the `category` (and the id prefix) for the three attack
files. `label` is the **ground truth** E4 scores against: an `attack` the immune
gate should quarantine, a `legit` case it should not — a quarantine on a legit
case is a true false positive, which is why the legit control is genuinely
unusual-but-valid, not a straw man. `baseline_ref` is present only for
`semantic_anomaly` cases: a standalone string cannot be "anomalous", so each
carries the agent's normal history the anomaly is measured against.

Embeddings are **never** stored in these files. Vectors are attached at load time
via an injected `Embedder` (`incident_to_events(inc, embedder)` /
`poison_to_event(case, embedder)`); committing 1024 floats per row would bloat
every diff and couple the corpus to an embedder seed.

Some poison `provenance` values intentionally violate runtime invariants (an
`hop_count` past `gossip_max_hops`, an empty `signature`) — that is the attack
data the gate must catch, not a schema fault.

## Regenerating

The committed `incidents/sample.json` is the bit-for-bit output of the offline
generator and a test asserts no drift:

```bash
python -m scenarios.generate_incidents --n 10 --seed 0 --backend offline \
    --out scenarios/incidents/sample.json
```

The **full 100-incident corpus** for the experiments is produced **once** with
Amazon Bedrock and committed — it is not part of the default (offline, no-cloud)
test path:

```bash
# one-time, requires AWS credentials + Bedrock model access
python -m scenarios.generate_incidents --n 100 --seed 0 --backend bedrock \
    --model-id us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --out scenarios/incidents/corpus.json
```

The Bedrock backend imports `boto3` lazily, so importing this package or running
the offline path never touches the cloud. Every produced incident — offline or
Bedrock — is validated through the loader before it is written, so a malformed
completion raises rather than landing in a committed file.

The poison suite is hand-authored (no model) for genuine variety and committed
directly.
