"""The SRE agent loop: perceive -> recall -> model -> remember.

One :class:`SREAgent` handles one incident per :meth:`handle_incident` call. The
loop is four steps over injected seams (a :class:`~agents.read_client.MemoryReader`,
a :class:`~agents.write_client.MemoryWriter`, a
:class:`~agents.model_client.ModelClient`), so the whole thing runs offline and
deterministically in tests, and against Bedrock + the managed MCP server in
production, with no code change.

The write step is isolated behind :meth:`_build_write` so the adversary variant
(:mod:`agents.adversary`) can substitute a malicious write while inheriting the
identical perceive/recall/model machinery — which is exactly why a caught attack
surfaces as an ordinary :class:`~agents.types.AgentStep` with ``rejected=True``.

Portable: :mod:`core` + sibling ``agents`` modules only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from agents.model_client import ModelClient, ModelRequest
from agents.prompts import SRE_SYSTEM, build_messages, parse_diagnosis
from agents.read_client import MemoryReader
from agents.types import AgentStep, Diagnosis, Incident
from agents.write_client import MemoryWriter, PoisonRejected
from core.config import AletheiaConfig
from core.models import MemoryKind


@dataclass(frozen=True, slots=True)
class _WriteSpec:
    """The write one incident produces — the seam the adversary overrides."""

    content: str
    kind: MemoryKind
    importance: float
    parent_mem: str | None
    hop_count: int
    meta: dict[str, Any] = field(default_factory=dict)


class SREAgent:
    """A single SRE fleet agent over injected read/write/model seams."""

    def __init__(
        self,
        agent_id: str,
        role: str,
        reader: MemoryReader,
        writer: MemoryWriter,
        model: ModelClient,
        config: AletheiaConfig,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self._reader = reader
        self._writer = writer
        self._model = model
        self._config = config

    # -- lifecycle ------------------------------------------------------------
    def ensure_registered(self) -> None:
        """Register this agent through the writer, if the writer supports it.

        A no-op for the HTTP edge (agents are registered server-side); the offline
        writer registers on the shared adapter so the first write does not hit
        ``AgentNotRegistered``.
        """
        register = getattr(self._writer, "register_agent", None)
        if callable(register):
            register(self.agent_id, self.role)

    # -- the loop -------------------------------------------------------------
    def handle_incident(self, incident: Incident) -> AgentStep:
        """Run the perceive -> recall -> model -> remember loop for one incident."""
        started = perf_counter()

        # 1. PERCEIVE + 2. RECALL (read path).
        query = self._build_query(incident)
        hits = self._reader.recall(self.agent_id, query)
        recalled_ids = [hit.mem_id for hit in hits]

        # 3. MODEL STEP.
        request = ModelRequest(
            system=SRE_SYSTEM,
            messages=build_messages(incident, hits),
            tag=incident.incident_id,
            temperature=self._config.temperature,
        )
        response = self._model.complete(request)
        diagnosis = parse_diagnosis(response.text)

        # 4. REMEMBER (write path) — reject-before-persist on a poisoned write.
        spec = self._build_write(incident, diagnosis)
        try:
            mem_id = self._writer.remember(
                self.agent_id,
                spec.content,
                kind=spec.kind,
                importance=spec.importance,
                parent_mem=spec.parent_mem,
                hop_count=spec.hop_count,
                meta=spec.meta,
            )
        except PoisonRejected as exc:
            return AgentStep(
                agent_id=self.agent_id,
                role=self.role,
                incident_id=incident.incident_id,
                recalled_ids=recalled_ids,
                diagnosis=diagnosis,
                mem_id=None,
                rejected=True,
                reason=str(exc.reason) if exc.reason is not None else "rejected",
                latency_ms=_elapsed_ms(started),
            )

        return AgentStep(
            agent_id=self.agent_id,
            role=self.role,
            incident_id=incident.incident_id,
            recalled_ids=recalled_ids,
            diagnosis=diagnosis,
            mem_id=mem_id,
            latency_ms=_elapsed_ms(started),
        )

    # -- steps a subclass can override ---------------------------------------
    def _build_query(self, incident: Incident) -> str:
        """The recall query: title plus signals, falling back to the description.

        Never empty — an empty query would embed to the anchored fallback vector
        and recall unrelated memory; the title always anchors it.
        """
        signals = [s for s in incident.signals if s and s.strip()]
        body = signals or [incident.description or incident.title]
        return incident.title + "\n" + "\n".join(body)

    def _build_write(self, incident: Incident, diagnosis: Diagnosis) -> _WriteSpec:
        """A first-hand observation: no parent, zero hops. The honest write path."""
        return _WriteSpec(
            content=diagnosis.learned_fact,
            kind=diagnosis.kind,
            importance=diagnosis.importance,
            parent_mem=None,
            hop_count=0,
            meta={
                "incident_id": incident.incident_id,
                "scenario": incident.meta.get("scenario"),
                "arm": self._config.ablation_name,
            },
        )


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)
