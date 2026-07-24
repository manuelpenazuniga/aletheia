"""The write seam — a :class:`MemoryWriter` Protocol + an offline in-process mirror.

The write path is the guarded one (the project plan §5.3): an agent signs the
provenance of what it learned, the immune gate inspects the provisional memory,
and only an accepted write is persisted. :class:`LocalMemoryWriter` is the
in-process mirror of the ingest endpoint used offline and in tests — it signs,
runs the gate *before* any write (reject-before-persist), and writes through a
:class:`~core.memory.MemoryService`. :class:`IngestHTTPWriter` is the production
edge that POSTs to the real ingest service; it imports ``httpx`` lazily and is
never run by the default test suite.

The immune dependency is kept **structural**: :class:`LocalMemoryWriter` takes any
object with ``inspect(event) -> verdict`` and treats a verdict as a rejection when
its ``accepted`` (core shape) or ``allowed`` (ingest shape) attribute is false.
That decouples the writer from where the immune types live and lets a test inject
a tiny gate double while production wires the real
:class:`core.immune.ImmuneSystem`.

Portable seam: this module speaks only :mod:`core` types + :mod:`agents.signing`;
``httpx`` is function-local to the edge writer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from agents.signing import Signer
from core.config import AletheiaConfig
from core.memory import MemoryService
from core.models import MemoryEvent, MemoryKind


class PoisonRejected(Exception):
    """Raised when the immune gate refuses a write before it is persisted.

    Carries the forensic fields of the immune verdict so the loop can surface a
    reject as an :class:`~agents.types.AgentStep` without any memory being written.
    """

    def __init__(
        self,
        reason: str | None,
        detector: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason = reason
        self.detector = detector
        self.payload: dict[str, Any] = dict(payload or {})
        super().__init__(f"write rejected by immune gate: {reason or 'unspecified'}")


@runtime_checkable
class MemoryWriter(Protocol):
    """Persist one memory the agent learned, or raise :class:`PoisonRejected`."""

    def remember(
        self,
        agent_id: str,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.EPISODIC,
        importance: float = 0.5,
        parent_mem: str | None = None,
        hop_count: int = 0,
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        """Return the new ``mem_id`` on success; raise on an immune rejection."""
        ...


@runtime_checkable
class ImmuneGate(Protocol):
    """Structural gate: inspect a provisional memory and return a verdict.

    The verdict must expose ``accepted`` (core shape) or ``allowed`` (ingest
    shape); a false value on whichever is present is a rejection.
    """

    def inspect(self, event: MemoryEvent) -> Any: ...


def _verdict_allowed(verdict: Any) -> bool:
    accepted = getattr(verdict, "accepted", None)
    if accepted is not None:
        return bool(accepted)
    return bool(getattr(verdict, "allowed", True))


def _verdict_reason(verdict: Any) -> str | None:
    reason = getattr(verdict, "reason", None)
    # A core QuarantineReason is a StrEnum; surface its string value.
    return getattr(reason, "value", reason)


class LocalMemoryWriter:
    """In-process mirror of the ingest endpoint: sign, gate, then write.

    When ``config.enable_immune`` is set and a gate is present, a provisional
    :class:`~core.models.MemoryEvent` is inspected before any write and a
    not-allowed verdict raises :class:`PoisonRejected` — nothing is persisted, and
    (matching the ingest endpoint's reject-before-persist for a memory that never
    got an id) no ``quarantine_log`` row is created. With the flag off, or no gate,
    the write goes straight through: a disabled immune system is a clean no-op
    (the A4 ablation), which is what E4 measures.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        signer: Signer,
        config: AletheiaConfig,
        immune_gate: ImmuneGate | None = None,
    ) -> None:
        self._service = memory_service
        self._signer = signer
        self._config = config
        self._gate = immune_gate
        # Registration reaches the shared adapter behind the service. The ingest
        # server registers agents out-of-band (infra); offline the fleet runner
        # registers through here so a write never hits AgentNotRegistered.
        self._adapter = getattr(memory_service, "_adapter", None)

    def register_agent(self, agent_id: str, role: str) -> None:
        """Register the agent on the shared adapter (offline infra mirror)."""
        if self._adapter is None:  # pragma: no cover - MemoryService always has one
            return
        token_hash = getattr(self._signer, "token_hash", "offline")
        self._adapter.register_agent(agent_id, role, token_hash)

    def remember(
        self,
        agent_id: str,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.EPISODIC,
        importance: float = 0.5,
        parent_mem: str | None = None,
        hop_count: int = 0,
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        signature = self._signer.sign(agent_id, content, parent_mem, hop_count)

        if self._config.enable_immune and self._gate is not None:
            provisional = MemoryEvent(
                agent_id=agent_id,
                content=content,
                kind=kind,
                importance=importance,
                parent_mem=parent_mem,
                hop_count=hop_count,
                signature=signature,
                meta=dict(meta) if meta else {},
            )
            verdict = self._gate.inspect(provisional)
            if not _verdict_allowed(verdict):
                raise PoisonRejected(
                    reason=_verdict_reason(verdict),
                    detector=getattr(verdict, "detector", None),
                    payload=dict(getattr(verdict, "payload", {}) or {}),
                )

        return self._service.remember(
            agent_id,
            content,
            kind=kind,
            importance=importance,
            parent_mem=parent_mem,
            hop_count=hop_count,
            signature=signature,
            meta=meta,
        )


class IngestHTTPWriter:
    """Edge :class:`MemoryWriter` that POSTs to the real ingest service.

    Signs via the :class:`~agents.signing.Signer`, sends the agent's bearer token,
    and maps a 422 immune-reject to :class:`PoisonRejected`. ``httpx`` is imported
    lazily inside :meth:`remember`, so importing this module pulls no HTTP client;
    it is never run by the default test suite. Requires an :class:`HmacSigner`
    (it exposes the bearer ``agent_token``).
    """

    def __init__(self, base_url: str, signer: Signer, *, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._signer = signer
        self._timeout = timeout

    def register_agent(self, agent_id: str, role: str) -> None:
        # Ingest registers agents out-of-band (infra provisioning); nothing to do
        # from the client side.
        return

    def remember(
        self,
        agent_id: str,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.EPISODIC,
        importance: float = 0.5,
        parent_mem: str | None = None,
        hop_count: int = 0,
        meta: Mapping[str, Any] | None = None,
    ) -> str:  # pragma: no cover - requires a running ingest server
        import httpx  # lazy on purpose: keep the zero-network default import-clean

        token = getattr(self._signer, "agent_token", None)
        if token is None:
            raise ValueError("IngestHTTPWriter requires a signer exposing agent_token")
        signature = self._signer.sign(agent_id, content, parent_mem, hop_count)
        body = {
            "agent_id": agent_id,
            "content": content,
            "kind": str(MemoryKind(kind)),
            "importance": importance,
            "provenance": {
                "parent_mem": parent_mem,
                "hop_count": hop_count,
                "signature": signature,
            },
            "meta": dict(meta) if meta else {},
        }
        resp = httpx.post(
            f"{self._base_url}/v1/memories",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=self._timeout,
        )
        if resp.status_code == 422:
            raise PoisonRejected(reason="write_rejected", detector="ingest")
        resp.raise_for_status()
        return resp.json()["mem_id"]
