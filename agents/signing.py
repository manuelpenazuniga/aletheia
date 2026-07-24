"""Provenance signing for the write path — a pure, fastapi-free :class:`Signer`.

An agent signs every write so the memory it produces is attributable (the project
plan §5.3 write path). The canonical encoding here is **byte-identical** to
:func:`ingest.auth.sign` / :func:`ingest.auth.canonical_message`, so a memory
signed offline by :class:`HmacSigner` also authenticates against the real ingest
server — a parity guaranteed by a test.

It is duplicated here, rather than imported, on purpose: ``ingest.auth`` imports
FastAPI at module top, and dragging a web framework into the agent path would
break the loop's offline testability. The canonical encoding is three lines; the
right long-term fix is to lift ``canonical_message``/``sign`` into a
framework-free module both sides share (flagged for a later refactor).

Portable: stdlib only.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Protocol, runtime_checkable


def derive_agent_token(secret: str, agent_id: str) -> str:
    """Per-agent token from the single server secret: ``HMAC(secret, agent_id)``.

    Byte-identical to :func:`ingest.auth.derive_agent_token`. Each agent holds
    only its own token; a token for agent X cannot sign for agent Y because the id
    is bound into the derivation.
    """
    return hmac.new(secret.encode("utf-8"), agent_id.encode("utf-8"), hashlib.sha256).hexdigest()


def canonical_message(agent_id: str, content: str, parent_mem: str | None, hop_count: int) -> bytes:
    """The exact bytes an agent signs — order and separators are the contract.

    Byte-identical to :func:`ingest.auth.canonical_message`.
    """
    return "\n".join([agent_id, content, parent_mem or "", str(hop_count)]).encode("utf-8")


def sign_message(
    agent_token: str, agent_id: str, content: str, parent_mem: str | None, hop_count: int
) -> str:
    """Produce the provenance signature. Matches :func:`ingest.auth.sign` exactly."""
    message = canonical_message(agent_id, content, parent_mem, hop_count)
    return hmac.new(agent_token.encode("utf-8"), message, hashlib.sha256).hexdigest()


@runtime_checkable
class Signer(Protocol):
    """Signs the provenance of one write."""

    def sign(self, agent_id: str, content: str, parent_mem: str | None, hop_count: int) -> str:
        """Return the HMAC signature bound to these provenance fields."""
        ...


class HmacSigner:
    """A :class:`Signer` holding one agent's token.

    Construct directly from a token, or via :meth:`for_agent` from the shared
    server secret and an agent id (the same derivation the ingest server uses).
    """

    def __init__(self, agent_token: str) -> None:
        self._token = agent_token

    @classmethod
    def for_agent(cls, secret: str, agent_id: str) -> HmacSigner:
        return cls(derive_agent_token(secret, agent_id))

    @property
    def agent_token(self) -> str:
        """The agent's bearer token — used only by the HTTP edge writer."""
        return self._token

    @property
    def token_hash(self) -> str:
        """A non-reversible fingerprint of the token, for ``agents.token_hash``."""
        return hashlib.sha256(self._token.encode("utf-8")).hexdigest()

    def sign(self, agent_id: str, content: str, parent_mem: str | None, hop_count: int) -> str:
        return sign_message(self._token, agent_id, content, parent_mem, hop_count)
