"""System prompt, message assembly, and tolerant diagnosis parsing.

The SRE agent asks the model for a JSON diagnosis and parses it here. Parsing is
deliberately tolerant: a model that returns malformed output, extra prose around
the JSON, or nothing usable must never crash the loop — it falls back to using the
raw text as the learned fact, so the fleet keeps running (the project plan §3.2:
everything shown is real; a bad model turn degrades gracefully rather than failing
the run).

Portable: stdlib + :mod:`core.models` + :mod:`agents.types` only.
"""

from __future__ import annotations

import json
from typing import Any

from agents.types import Diagnosis, Incident
from core.models import MemoryHit, MemoryKind

#: The fleet's operating instructions. Framed as production institutional memory
#: (the project plan §0): reliable memory despite unreliable agents.
SRE_SYSTEM = (
    "You are an SRE agent in a production incident-response fleet. You share one "
    "institutional memory with your peers. Diagnose the incident using the recalled "
    "institutional memory when it is relevant, prefer the most recent runbook over an "
    "older one, and record the durable lesson so the rest of the fleet can reuse it. "
    "Respond with a single JSON object with keys: summary, action, learned_fact, "
    "importance (0.0-1.0), kind (episodic|semantic). Output only the JSON object."
)


def build_messages(incident: Incident, hits: list[MemoryHit]) -> list[dict[str, str]]:
    """Assemble the chat messages for one incident, grounding on recalled memory.

    Deterministic: recalled memories are rendered in the order retrieval returned
    them (descending relevance, already budget-truncated), so the same inputs
    produce the same prompt in every process.
    """
    lines: list[str] = [f"INCIDENT {incident.incident_id}: {incident.title}"]
    if incident.description:
        lines.append(incident.description)
    signals = [s for s in incident.signals if s and s.strip()]
    if signals:
        lines.append("Observed signals:")
        lines.extend(f"- {s}" for s in signals)

    if hits:
        lines.append("")
        lines.append("Recalled institutional memory (most relevant first):")
        for hit in hits:
            lines.append(f"- [{hit.mem_id}] {hit.content}")
    else:
        lines.append("")
        lines.append("Recalled institutional memory: none yet.")

    return [{"role": "user", "content": "\n".join(lines)}]


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object out of ``text``. None if unusable."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    # Prose around the JSON: try the span from the first '{' to the last '}'.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(stripped[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _coerce_importance(value: Any) -> float:
    try:
        importance = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, importance))


def _coerce_kind(value: Any) -> MemoryKind:
    try:
        return MemoryKind(value)
    except (ValueError, KeyError):
        return MemoryKind.EPISODIC


def parse_diagnosis(text: str) -> Diagnosis:
    """Parse a model completion into a :class:`Diagnosis`, tolerant of garbage.

    A well-formed JSON object maps directly; anything else falls back to using the
    raw text as the ``learned_fact`` (truncated for the summary), guaranteeing a
    non-empty learned fact so the write never violates ``MemoryEvent``'s contract.
    """
    data = _extract_json(text)
    if data is not None:
        summary = str(data.get("summary", "")).strip()
        action = str(data.get("action", "")).strip()
        learned = str(data.get("learned_fact", "")).strip()
        if not learned:
            learned = summary or text.strip()
        if learned:
            return Diagnosis(
                summary=summary or learned[:120],
                action=action,
                learned_fact=learned,
                importance=_coerce_importance(data.get("importance", 0.5)),
                kind=_coerce_kind(data.get("kind", MemoryKind.EPISODIC)),
            )

    raw = text.strip() or "no diagnosis produced"
    return Diagnosis(
        summary=raw[:120],
        action="",
        learned_fact=raw,
        importance=0.5,
        kind=MemoryKind.EPISODIC,
    )
