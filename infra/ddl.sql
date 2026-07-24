-- =============================================================================
-- Aletheia — full schema (CLAUDE.md §6). Source of truth for the memory layer.
--
-- Apply with:  ./infra/apply_ddl.sh          (local docker or cloud, via DSN)
--
-- Rules that this schema encodes (do not break them):
--   (a) `quarantine` and `supersede` are STATUS TRANSITIONS, never DELETEs —
--       the audit trail is the product.
--   (b) every multi-table write (memories + provenance) happens in ONE
--       serializable transaction: no window where a vector exists without its
--       operational row, or vice versa. This is the CockroachDB argument.
--   (c) after Phase 1 this file is frozen: schema changes go to numbered files
--       in infra/migrations/, never as retroactive edits here.
--
-- Embedding dimension is 1024 (Amazon Titan Text Embeddings V2) and MUST match
-- AletheiaConfig.embedding_dim in core/config.py. Changing it rebuilds the index.
-- =============================================================================

CREATE TABLE IF NOT EXISTS agents (
    agent_id     STRING PRIMARY KEY,
    role         STRING NOT NULL,              -- 'sre' | 'ops' | 'adversary'
    token_hash   STRING NOT NULL,              -- ingest auth: hash of the per-agent token
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memories (
    mem_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id     STRING NOT NULL REFERENCES agents(agent_id),
    kind         STRING NOT NULL,              -- 'episodic' | 'semantic' | 'canonical_ref'
    content      STRING NOT NULL,
    embedding    VECTOR(1024),
    importance   FLOAT NOT NULL DEFAULT 0.5,   -- drives metabolic forgetting
    cost_tokens  INT NOT NULL,                 -- estimated retention cost
    status       STRING NOT NULL DEFAULT 'active',  -- 'active'|'superseded'|'quarantined'|'archived'
    superseded_by UUID,                        -- knowledge-update chain
    created_at   TIMESTAMPTZ DEFAULT now(),
    last_accessed TIMESTAMPTZ DEFAULT now(),
    access_count INT NOT NULL DEFAULT 0
);

-- Distributed vector index (C-SPANN). Freshness is immediate after
-- insert/supersede — no separate vector store to fall out of sync with.
--
-- `status` is a PREFIX COLUMN, not decoration. Retrieval must exclude superseded,
-- quarantined and archived memories, and CockroachDB only routes a query through
-- the vector index when the query's filters are covered by the index: measured
-- against v25.4.13, `WHERE status = 'active' ORDER BY embedding <=> $1` plans as
-- a FULL SCAN on an index defined over (embedding) alone, and as
-- `vector search ... prefix spans: [/'active' - /'active']` on this one. The
-- prefix also means a supersede moves the entry out of the searched partition
-- immediately — the C-SPANN freshness property, made visible.
--
-- `vector_cosine_ops` pins cosine distance so the CockroachDB adapter and the
-- InMemoryAdapter (which scores with cosine) agree by contract rather than by the
-- coincidence that L2 and cosine rank unit-length vectors identically.
-- Retrieval must therefore use the `<=>` operator.
--
-- Decided 2026-07-24 with the human, during Phase 0, while ddl.sql was still
-- editable under §6 rule (c). Any later change goes to infra/migrations/.
CREATE VECTOR INDEX IF NOT EXISTS idx_mem_embedding
    ON memories (status, embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_mem_agent_status ON memories (agent_id, status);

CREATE TABLE IF NOT EXISTS canonical_facts (
    fact_key     STRING PRIMARY KEY,           -- e.g. 'runbook:db-latency:fix'
    content      STRING NOT NULL,
    version      INT NOT NULL DEFAULT 1,
    source_mem   UUID REFERENCES memories(mem_id),
    updated_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS provenance (
    link_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mem_id       UUID NOT NULL REFERENCES memories(mem_id),
    parent_mem   UUID,                         -- NULL = direct observation
    agent_id     STRING NOT NULL,
    signature    STRING NOT NULL,              -- HMAC(agent token, content)
    hop_count    INT NOT NULL DEFAULT 0,       -- gossip hops (degradation distance)
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS quarantine_log (
    q_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mem_id       UUID,
    reason       STRING NOT NULL,              -- 'bad_provenance'|'semantic_anomaly'|'injection_pattern'
    detector     STRING NOT NULL,
    payload      JSONB,
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- Experimental results live in the database, not in loose CSV files.
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    arm          STRING NOT NULL,              -- 'A0_full', 'A1_no_consolidation', ...
    seed         INT NOT NULL,
    metrics      JSONB NOT NULL,
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ
);
