"""The SRE agent fleet: the perceive->recall->model->remember loop over injected
read/write/model seams, an adversary poisoning variant, and a deterministic,
fully-offline fleet runner with an audit trail.

Import surface is the seams (Protocols + offline/edge impls), the agents, and the
runner. ``agents`` never imports ``scenarios`` (the generation seam reuses
``agents.model_client`` in the other direction), and the only cloud/HTTP SDKs are
imported lazily inside the edge implementations, so importing this package pulls
no boto3 or httpx.
"""

from agents.adversary import (
    ATTACK_BAD_PROVENANCE,
    ATTACK_CANONICAL_REWRITE,
    ATTACK_CONTENT_INJECTION,
    ATTACKS,
    AdversaryAgent,
)
from agents.agent import SREAgent
from agents.fleet import (
    make_local_adversary,
    make_local_sre_agent,
    run_fleet,
)
from agents.model_client import (
    BedrockModelClient,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ScriptedModel,
)
from agents.read_client import LocalMemoryReader, MCPMemoryReader, MemoryReader
from agents.signing import HmacSigner, Signer, derive_agent_token, sign_message
from agents.types import (
    AgentStep,
    AuditRecord,
    Diagnosis,
    FleetRunResult,
    Incident,
)
from agents.write_client import (
    ImmuneGate,
    IngestHTTPWriter,
    LocalMemoryWriter,
    MemoryWriter,
    PoisonRejected,
)

__all__ = [
    "ATTACKS",
    "ATTACK_BAD_PROVENANCE",
    "ATTACK_CANONICAL_REWRITE",
    "ATTACK_CONTENT_INJECTION",
    "AdversaryAgent",
    "AgentStep",
    "AuditRecord",
    "BedrockModelClient",
    "Diagnosis",
    "FleetRunResult",
    "HmacSigner",
    "ImmuneGate",
    "Incident",
    "IngestHTTPWriter",
    "LocalMemoryReader",
    "LocalMemoryWriter",
    "MCPMemoryReader",
    "MemoryReader",
    "MemoryWriter",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "PoisonRejected",
    "SREAgent",
    "ScriptedModel",
    "Signer",
    "derive_agent_token",
    "make_local_adversary",
    "make_local_sre_agent",
    "run_fleet",
    "sign_message",
]
