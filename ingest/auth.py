"""Write-path authentication: per-agent HMAC tokens and signed provenance.

All cryptography lives here so it is unit-testable without HTTP. The scheme has
two layers over one server secret (``ALETHEIA_INGEST_HMAC_SECRET``):

1. **Per-agent token.** ``derive_agent_token(secret, agent_id)`` =
   ``HMAC(secret, agent_id)``. Each agent is handed only its own token; the
   server needs no per-agent secret store — it re-derives the expected token
   from the id in the body. A token for agent X therefore cannot authenticate a
   write claiming ``agent_id = Y``.
2. **Body signature.** ``sign(token, agent_id, content, parent_mem, hop_count)``
   binds the fields that must not change in transit into a single HMAC. Tampering
   with the content, the parent, or the hop count after signing invalidates it.

Threat-model note (demo scope): the token and the signing key are the same
derived value, so a holder of an agent's token can also forge that agent's
signature. The signature adds transit-integrity and populates the provenance
row; it is not a second independent factor. Real per-agent secrets and rotation
are out of scope for this unit.

This module may import fastapi (it is part of the write path, not the core).
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from fastapi import Header, Request

from ingest.schemas import RememberRequest


class Unauthorized(Exception):
    """Any authentication failure. Translated to HTTP 401 by the app handler.

    The message is for server logs only — the handler returns a generic body so
    a probing client learns nothing about which check failed.
    """


@dataclass(frozen=True)
class AuthContext:
    """The result of a successful authentication, passed to the endpoint."""

    agent_id: str
    agent_token: str


def derive_agent_token(secret: str, agent_id: str) -> str:
    """Per-agent token derived from the single server secret: ``HMAC(secret, id)``."""
    return hmac.new(secret.encode("utf-8"), agent_id.encode("utf-8"), hashlib.sha256).hexdigest()


def canonical_message(agent_id: str, content: str, parent_mem: str | None, hop_count: int) -> bytes:
    """The exact bytes an agent signs. Order and separators are part of the contract.

    A newline-joined, fixed-order encoding so the agent and the server compute
    byte-identical input; changing any field changes these bytes and breaks the
    signature.
    """
    return "\n".join([agent_id, content, parent_mem or "", str(hop_count)]).encode("utf-8")


def sign(
    agent_token: str,
    agent_id: str,
    content: str,
    parent_mem: str | None,
    hop_count: int,
) -> str:
    """Produce ``Provenance.signature``. Used by agents and by tests to sign a write."""
    message = canonical_message(agent_id, content, parent_mem, hop_count)
    return hmac.new(agent_token.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify(secret: str, req: RememberRequest, bearer: str) -> AuthContext:
    """Authenticate a request. Raise :class:`Unauthorized` on any mismatch.

    Both comparisons use :func:`hmac.compare_digest` (constant time) so a caller
    cannot use response timing as an oracle to guess a token or signature.
    """
    expected_token = derive_agent_token(secret, req.agent_id)
    # Proves the caller holds THIS agent's token. Because the token is derived
    # from the body's agent_id, a token for one agent cannot sign for another.
    # Compare as UTF-8 bytes: compare_digest raises TypeError on a non-ASCII str,
    # and both operands here are attacker-influenced — a non-ASCII bearer must be
    # a plain 401 mismatch, never a 500.
    if not hmac.compare_digest(bearer.encode("utf-8"), expected_token.encode("utf-8")):
        raise Unauthorized("bearer token does not match the agent")

    expected_sig = sign(
        expected_token,
        req.agent_id,
        req.content,
        req.provenance.parent_mem,
        req.provenance.hop_count,
    )
    # Body integrity: content / parent / hop cannot be altered after signing.
    if not hmac.compare_digest(
        req.provenance.signature.encode("utf-8"), expected_sig.encode("utf-8")
    ):
        raise Unauthorized("provenance signature does not match the body")

    return AuthContext(agent_id=req.agent_id, agent_token=expected_token)


def get_hmac_secret(request: Request) -> str:
    """Provider: the server signing secret, stashed on ``app.state`` by create_app.

    Defined alongside the crypto (and re-exported from :mod:`ingest.app`) so
    dependencies here can read it without importing the app module and creating a
    circular import.
    """
    return request.app.state.hmac_secret


async def require_bearer(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    """FastAPI dependency: extract the ``Bearer <token>`` credential, or 401.

    Deliberately does *not* declare the request body. FastAPI keys a body
    parameter by name, so declaring ``RememberRequest`` both here and on the path
    operation would make it demand two separate JSON objects. The endpoint owns
    the single body param and calls :func:`verify` with it explicitly — same
    crypto, one parse of the stream.
    """
    if not authorization:
        raise Unauthorized("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Unauthorized("Authorization scheme must be 'Bearer <token>'")
    return token
