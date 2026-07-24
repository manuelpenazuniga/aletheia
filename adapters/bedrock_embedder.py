"""Amazon Bedrock embedder (Titan Text Embeddings V2) — the production Embedder.

Lives outside ``core/`` on purpose: this is the only place boto3 is allowed to
meet the embedding contract. The core receives it as a
:class:`core.embeddings.Embedder`.

Fails loudly. If credentials, region or model access are missing it raises with
the exact remediation step — it never falls back to a synthetic vector, because a
silently fake embedding would corrupt every number the project reports
(the project plan §3.1, §3.2).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from core.config import DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL_ID


class BedrockUnavailable(RuntimeError):
    """Raised when Bedrock cannot be reached or the model is not enabled."""


class BedrockEmbedder:
    """Titan Text Embeddings V2 via ``bedrock-runtime:InvokeModel``."""

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
        dim: int = DEFAULT_EMBEDDING_DIM,
        *,
        region_name: str | None = None,
        client: object | None = None,
    ) -> None:
        self.model_id = model_id
        self._dim = dim
        self._region = region_name
        self._client = client
        self.total_input_tokens = 0  # observability: real token spend, not estimated

    @property
    def dim(self) -> int:
        return self._dim

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - boto3 is a declared dep
                raise BedrockUnavailable("boto3 is not installed: pip install -e '.[dev]'") from exc
            kwargs = {"region_name": self._region} if self._region else {}
            try:
                self._client = boto3.client("bedrock-runtime", **kwargs)
            except Exception as exc:
                raise BedrockUnavailable(
                    f"could not create a bedrock-runtime client: {exc}\n"
                    "Check AWS_REGION and, if set, that AWS_PROFILE names a profile that "
                    "exists in ~/.aws/config (an empty AWS_PROFILE= line in .env is a "
                    "common cause)."
                ) from exc
        return self._client

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Titan embeds one input per call; batching here is sequential by design."""
        client = self._get_client()
        vectors: list[list[float]] = []
        for text in texts:
            body = json.dumps({"inputText": text, "dimensions": self._dim, "normalize": True})
            try:
                response = client.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )
            except Exception as exc:  # boto3 error types are created at runtime
                raise BedrockUnavailable(
                    f"Bedrock InvokeModel failed for model {self.model_id!r} "
                    f"in region {self._region or '(default from environment)'}: {exc}\n"
                    "Check: (1) AWS credentials are configured, (2) the region is correct, "
                    "(3) model access is granted in the Bedrock console under Model access."
                ) from exc

            payload = json.loads(response["body"].read())
            vector = payload["embedding"]
            if len(vector) != self._dim:
                raise BedrockUnavailable(
                    f"embedding dimension mismatch: model returned {len(vector)}, "
                    f"configuration expects {self._dim} (AletheiaConfig.embedding_dim "
                    "and VECTOR(...) in infra/ddl.sql must agree)"
                )
            self.total_input_tokens += int(payload.get("inputTextTokenCount", 0))
            vectors.append([float(v) for v in vector])
        return vectors

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BedrockEmbedder(model_id={self.model_id!r}, dim={self._dim})"
