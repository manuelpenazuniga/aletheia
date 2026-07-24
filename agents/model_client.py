"""The model seam — a :class:`ModelClient` Protocol + a deterministic offline fake.

An SRE agent never calls Bedrock directly. It calls :class:`ModelClient.complete`,
so the loop is testable and reproducible offline. Two implementations ship here:

* :class:`ScriptedModel` — deterministic, network-free, routes a request to a
  canned completion by ``tag`` (falling back to a callable, a default, or a stable
  BLAKE2b hash of the prompt). Same request, same response, in every process. This
  is what the tests and offline demos use.
* :class:`BedrockModelClient` — the edge implementation. It imports ``boto3``
  *lazily* inside ``__init__``/``complete`` so that ``import agents.model_client``
  pulls no cloud SDK, and it is never exercised by the default test run.

This is the single definition of the agent model seam; ``scenarios`` reuses it
via ``from agents.model_client import ModelClient, ScriptedModel`` rather than
keeping a second, drifting copy. ``agents`` must never import ``scenarios`` in
return, to keep that reuse cycle-free.

Determinism rule (the project plan §3.5): the fallback never uses the builtin
``hash`` (salted per process); all entropy comes from BLAKE2b over the prompt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One completion request. ``tag`` is the deterministic routing key.

    The loop sets ``tag`` to the incident id so a scripted model can return a
    fixed completion per incident; ``system``/``messages`` follow the usual chat
    shape and are what the hash fallback keys on when no ``tag`` route matches.
    """

    system: str
    messages: list[dict[str, str]]
    tag: str | None = None
    temperature: float = 0.2
    max_tokens: int = 512


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One completion. ``text`` is the raw model output the loop parses."""

    text: str
    stop_reason: str = "stop"
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelClient(Protocol):
    """Turns a :class:`ModelRequest` into a :class:`ModelResponse`."""

    def complete(self, req: ModelRequest) -> ModelResponse:
        """Complete ``req``. Must be a pure function of its inputs for tests."""
        ...


class ScriptedModel:
    """Deterministic, offline :class:`ModelClient` for tests and local dev.

    Routing order for a request:

    1. If ``responses`` is a callable, it is invoked with the request and its
       return value is the completion text (the caller owns determinism).
    2. Else if the request's ``tag`` is a key in the ``responses`` mapping, that
       string is returned.
    3. Else if a ``default`` was configured, it is returned.
    4. Else a stable BLAKE2b digest of ``system`` + ``messages`` is returned — a
       deterministic, non-JSON string, which also exercises the loop's tolerant
       parse fallback.
    """

    def __init__(
        self,
        responses: Mapping[str, str] | Callable[[ModelRequest], str] | None = None,
        default: str | None = None,
    ) -> None:
        self._responses = responses
        self._default = default

    def complete(self, req: ModelRequest) -> ModelResponse:
        return ModelResponse(text=self._resolve(req), meta={"model": "scripted"})

    def _resolve(self, req: ModelRequest) -> str:
        if callable(self._responses):
            return self._responses(req)
        if (
            isinstance(self._responses, Mapping)
            and req.tag is not None
            and req.tag in self._responses
        ):
            return self._responses[req.tag]
        if self._default is not None:
            return self._default
        return self._hash_fallback(req)

    @staticmethod
    def _hash_fallback(req: ModelRequest) -> str:
        # Stable across processes: BLAKE2b over a canonical JSON encoding of the
        # routing-relevant fields. Never the builtin salted hash.
        payload = json.dumps(
            {"system": req.system, "messages": req.messages}, sort_keys=True, ensure_ascii=False
        )
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
        return f"UNSCRIPTED:{digest}"


class BedrockModelClient:
    """Edge :class:`ModelClient` over Amazon Bedrock. Not run by the default tests.

    ``boto3`` is imported lazily inside ``__init__`` and ``complete`` so that
    importing this module never pulls the cloud SDK into the interpreter. Model id
    and temperature come from :class:`~core.config.AletheiaConfig` — the single
    source of truth (the project plan §4).
    """

    def __init__(self, config: Any, *, region_name: str | None = None, client: Any = None) -> None:
        self._config = config
        if client is not None:
            self._client = client
        else:  # pragma: no cover - requires boto3 + AWS credentials
            import boto3  # lazy on purpose: keep the zero-network default import-clean

            self._client = boto3.client("bedrock-runtime", region_name=region_name)

    def complete(self, req: ModelRequest) -> ModelResponse:  # pragma: no cover - needs Bedrock
        import json as _json  # kept local so the edge stays self-contained

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "system": req.system,
            "messages": [
                {"role": m.get("role", "user"), "content": m["content"]} for m in req.messages
            ],
        }
        resp = self._client.invoke_model(modelId=self._config.model_id, body=_json.dumps(body))
        payload = _json.loads(resp["body"].read())
        parts = payload.get("content", [])
        text = "".join(p.get("text", "") for p in parts)
        return ModelResponse(
            text=text,
            stop_reason=payload.get("stop_reason", "stop"),
            meta={"model": self._config.model_id},
        )
