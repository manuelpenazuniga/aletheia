"""The generation seam (C-scenarios) — a ModelClient Protocol + an offline fake.

Corpus generation goes through :class:`ModelClient` so the incident generator is
testable and reproducible offline. The real path (``--backend bedrock``) lazily
imports boto3 inside :mod:`scenarios.generate_incidents`; the default path uses
:class:`OfflineTemplateModel`, a deterministic, seed-driven template expander
that touches no network.

This is a *minimal* seam defined here because ``agents/`` does not yet ship a
ModelClient. When the agent loop lands one, reconcile the two Protocols (align on
this signature or re-export) rather than letting two diverge — flagged in the
project plan Phase 2 notes.

Determinism rule (the project plan §3.5): the offline model never uses the
builtin ``hash`` (salted per process). All entropy comes from ``random.Random``
seeded through a BLAKE2b digest of ``(seed, prompt)``, so the same inputs yield a
byte-identical completion in every process and on every machine.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelClient(Protocol):
    """Turns a prompt into a completion string.

    The generator asks for a JSON document and validates whatever comes back
    through the loader schema, so an implementation is free to be a real LLM or a
    deterministic template — the contract is only "return a string".
    """

    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> str:
        """Complete ``prompt`` deterministically under ``seed``."""
        ...


# -----------------------------------------------------------------------------
# Offline template catalogs — static, English, SRE-flavoured. Rendered into the
# Incident JSON shape the loader validates. Kept deliberately varied so that the
# knowledge-update signal (a runbook plus a real revision) is legible and the
# generated corpus does not read as trivial repetition.
# -----------------------------------------------------------------------------

#: One entry per SRE incident category. Each supplies enough distinct phrasings
#: that a 10-incident draw produces genuinely different text, and every revision
#: encodes a real reason a runbook fix would change over time.
_CATALOG: dict[str, dict[str, object]] = {
    "db-latency": {
        "titles": [
            "Primary database p99 latency spike",
            "Query latency regression on the orders DB",
            "Connection-pool saturation on the primary",
        ],
        "symptoms": [
            "p99 read latency above 800ms",
            "connection pool exhausted, requests queueing",
            "replica lag climbing past 30s",
            "slow-query log filling with full table scans",
        ],
        "runbooks": [
            "Fail reads over to the warm replica, then raise the pool size on the "
            "primary from 50 to 120 and restart the pgbouncer sidecar.",
            "Throttle the batch reconciliation job and add the missing index on "
            "orders(created_at) before re-enabling writes.",
        ],
        "revisions": [
            (
                "Do NOT raise the pool size blindly — that masked a runaway "
                "analytics query last time. First identify the offending query in "
                "pg_stat_activity and cancel it, then fail reads to the replica.",
                "raising the pool size hid the real cause and delayed recovery",
            ),
            (
                "The warm replica is now read-only by default; promote it with "
                "cockroach node decommission only after checking replication factor.",
                "replica topology changed after the multi-region migration",
            ),
        ],
    },
    "disk-full": {
        "titles": [
            "Disk usage at 98% on the log volume",
            "Write failures from a full data partition",
            "Ingest halted by an out-of-space error",
        ],
        "symptoms": [
            "df reports 98% on /var/log",
            "writes failing with ENOSPC",
            "log rotation stalled for six hours",
            "core dumps accumulating under /var/crash",
        ],
        "runbooks": [
            "Truncate rotated logs older than 3 days, then extend the LVM volume by "
            "20GB and restart the ingest workers.",
            "Move core dumps to the archive bucket and re-enable logrotate with a "
            "hard 7-day retention.",
        ],
        "revisions": [
            (
                "Never truncate the active log file while the process holds it — the "
                "space is not reclaimed until restart. Rotate first, then delete the "
                "rotated files, then extend the volume.",
                "truncating an open file did not free space and caused a second page",
            ),
            (
                "The archive bucket now enforces object-lock; stage dumps in the "
                "quarantine prefix instead so retention policy applies.",
                "archive bucket policy changed to WORM storage",
            ),
        ],
    },
    "bad-deploy": {
        "titles": [
            "Error rate spike after the 14:02 deploy",
            "Bad rollout of the checkout service",
            "Post-deploy 500s on the payments path",
        ],
        "symptoms": [
            "5xx rate jumped from 0.1% to 12% at 14:02",
            "new pods crash-looping on a missing env var",
            "canary health check red across two zones",
            "feature flag defaulting to on in production",
        ],
        "runbooks": [
            "Roll back to the previous revision with the deploy tool, confirm the "
            "canary goes green, then open an incident for the missing env var.",
            "Disable the offending feature flag, drain the crash-looping pods, and "
            "hold the pipeline until the fix merges.",
        ],
        "revisions": [
            (
                "Rollback alone is not enough when a schema migration shipped with "
                "the deploy — check for applied migrations first and run the down "
                "migration before rolling back the code.",
                "a forward-only migration made a naive rollback corrupt state",
            ),
            (
                "Use the progressive rollback (25% steps) now that the deploy tool "
                "supports it; an instant rollback overwhelmed the cold cache.",
                "instant rollback caused a thundering-herd cache miss",
            ),
        ],
    },
    "cert-expired": {
        "titles": [
            "TLS certificate expired on the API gateway",
            "Handshake failures after a cert rotation",
            "Expired intermediate cert breaking mTLS",
        ],
        "symptoms": [
            "clients failing with CERTIFICATE_EXPIRED",
            "gateway returning 526 to all TLS traffic",
            "internal mTLS handshakes rejected fleet-wide",
            "cert expiry alert fired 2 hours before outage",
        ],
        "runbooks": [
            "Issue a new leaf cert from the internal CA, deploy it to the gateway "
            "secret, and roll the gateway pods to pick it up.",
            "Rotate the intermediate, rebuild the full chain, and reload envoy "
            "without a restart using the hot-reload endpoint.",
        ],
        "revisions": [
            (
                "Rolling the pods drops in-flight connections — use the envoy "
                "hot-reload endpoint to swap the cert with zero downtime instead of "
                "a pod restart.",
                "the pod restart caused an avoidable connection reset storm",
            ),
            (
                "The internal CA moved to ACME; request certs via the cert-manager "
                "Issuer rather than the old manual signing script.",
                "manual signing was deprecated after the ACME migration",
            ),
        ],
    },
    "memory-leak": {
        "titles": [
            "OOM kills on the recommendation service",
            "Slow memory leak in the worker fleet",
            "RSS climbing until the pod is evicted",
        ],
        "symptoms": [
            "RSS growing 200MB/hour with flat traffic",
            "kernel OOM-killer terminating the main process",
            "GC pause time doubling every few hours",
            "pod evicted then crash-looping under memory pressure",
        ],
        "runbooks": [
            "Restart the leaking workers on a rolling schedule, raise the memory "
            "limit by 30%, and capture a heap dump for the owning team.",
            "Roll back to the last known-good build and pin the memory limit until "
            "the leak is root-caused.",
        ],
        "revisions": [
            (
                "Raising the limit only delays the OOM — set a memory-based "
                "liveness probe so the pod recycles before the kernel kills it, and "
                "always capture the heap dump BEFORE the restart.",
                "a bare limit bump just moved the OOM a few hours later",
            ),
            (
                "The heap dump endpoint is now auth-gated; fetch it through the "
                "debug sidecar token, not the old unauthenticated port.",
                "the debug port was closed after a security review",
            ),
        ],
    },
    "network-partition": {
        "titles": [
            "Cross-zone network partition on the mesh",
            "Split-brain risk from a flapping link",
            "Service discovery timeouts across regions",
        ],
        "symptoms": [
            "zone-a cannot reach zone-b for 40s intervals",
            "service discovery returning stale endpoints",
            "quorum warnings from the coordination layer",
            "retries storming as timeouts cascade",
        ],
        "runbooks": [
            "Drain traffic from the flapping zone, confirm quorum holds in the "
            "remaining zones, and page the network team with the link ID.",
            "Increase the discovery TTL temporarily and enable request hedging "
            "until the link stabilises.",
        ],
        "revisions": [
            (
                "Draining a zone during a partition can tip the remaining zones out "
                "of quorum — verify the replication factor allows losing a zone "
                "FIRST, otherwise hedge requests instead of draining.",
                "an eager drain nearly caused a quorum loss",
            ),
            (
                "Request hedging now has a global budget; enable it per-route, not "
                "fleet-wide, to avoid doubling backend load.",
                "fleet-wide hedging doubled backend load in the last incident",
            ),
        ],
    },
}

_CATEGORIES: tuple[str, ...] = tuple(_CATALOG)

#: A small pool of plausible authoring agents, so ``runbook.authored_by`` varies.
_AUTHORS: tuple[str, ...] = ("sre-01", "sre-02", "sre-03", "ops-01")


class OfflineTemplateModel:
    """A deterministic, network-free :class:`ModelClient` for corpus generation.

    Every completion is a self-contained JSON document (an Incident *without* an
    id — the generator assigns ids) rendered from the static catalog above. The
    only entropy is a :class:`random.Random` seeded from a BLAKE2b digest of
    ``(seed, prompt)``, so two instances with the same seed return byte-identical
    strings for the same prompt, in any process.
    """

    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> str:
        rng = self._rng(prompt, seed)

        category = rng.choice(_CATEGORIES)
        entry = _CATALOG[category]

        titles: list[str] = entry["titles"]  # type: ignore[assignment]
        symptom_pool: list[str] = entry["symptoms"]  # type: ignore[assignment]
        runbooks: list[str] = entry["runbooks"]  # type: ignore[assignment]
        revision_pool: list[tuple[str, str]] = entry["revisions"]  # type: ignore[assignment]

        title = rng.choice(titles)
        n_symptoms = rng.randint(2, min(3, len(symptom_pool)))
        symptoms = rng.sample(symptom_pool, n_symptoms)
        runbook_content = rng.choice(runbooks)
        author = rng.choice(_AUTHORS)

        # One or two revisions — always at least one, because the knowledge-update
        # signal is the whole point of the corpus.
        n_revisions = rng.randint(1, len(revision_pool))
        chosen = rng.sample(revision_pool, n_revisions)
        revisions = [
            {
                "version": 2 + idx,
                "supersedes": 1 + idx,
                "content": content,
                "reason": reason,
            }
            for idx, (content, reason) in enumerate(chosen)
        ]

        incident = {
            "title": title,
            "category": category,
            "symptoms": symptoms,
            "fact_key": f"runbook:{category}:fix",
            "runbook": {"version": 1, "content": runbook_content, "authored_by": author},
            "revisions": revisions,
        }
        return json.dumps(incident, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _rng(prompt: str, seed: int) -> random.Random:
        digest = hashlib.blake2b(
            prompt.encode("utf-8"), key=str(seed).encode("utf-8"), digest_size=16
        ).digest()
        return random.Random(int.from_bytes(digest, "big"))
