"""Request/response models for the ingest write-path (pydantic v2).

These sit at the HTTP boundary only. They validate and shape what crosses the
wire; the moment a request is authenticated it is turned into a
:class:`core.models.MemoryEvent` and everything downstream speaks core types.

Two boundary rules are enforced here, before any auth or write runs:

* ``extra="forbid"`` on the request — an agent POSTs *content*, never a raw
  vector. A stray ``embedding`` field (or any unknown key) is a 422, not a
  silently ignored value that could poison the index.
* Field constraints mirror :class:`~core.models.MemoryEvent`'s own invariants
  (non-empty content, importance in [0, 1], hop_count >= 0) so a malformed body
  is rejected by pydantic with a 422 before it reaches the core.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.models import MemoryKind, MemoryStatus


class Provenance(BaseModel):
    """The signed provenance an agent attaches to a write.

    ``signature`` is ``HMAC(agent_token, canonical_message)`` — see
    :mod:`ingest.auth`. ``parent_mem``/``hop_count`` describe a gossip edge and
    are bound into that signature, so they cannot be altered after signing.
    """

    model_config = ConfigDict(extra="forbid")

    parent_mem: str | None = None
    hop_count: int = Field(0, ge=0)
    signature: str = Field(min_length=1)


class RememberRequest(BaseModel):
    """A request to persist one memory. The embedding is computed server-side."""

    # Reject anything we did not ask for — most importantly a client-supplied
    # `embedding`: the vector is derived from content by the trusted service, so
    # an agent can never inject an arbitrary point into the index.
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    kind: MemoryKind = MemoryKind.EPISODIC
    importance: float = Field(0.5, ge=0.0, le=1.0)
    provenance: Provenance
    meta: dict[str, Any] = Field(default_factory=dict)


class RememberResponse(BaseModel):
    """What the caller gets back after a successful write."""

    mem_id: str
    agent_id: str
    kind: MemoryKind
    status: MemoryStatus
    cost_tokens: int
    created_at: datetime


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: str = "ok"
