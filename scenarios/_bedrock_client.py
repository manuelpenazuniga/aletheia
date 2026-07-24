"""Bedrock-backed :class:`~scenarios.model_client.ModelClient` (one-time corpus).

Isolated in its own module and imported lazily by
:func:`scenarios.generate_incidents._make_bedrock_client`, so ``boto3`` never
enters the import graph of the default (offline) generation path or any test —
the no-cloud-in-the-default-test-run rule. This module is exercised only when a
human runs ``--backend bedrock`` to produce the full 100-incident corpus once.
"""

from __future__ import annotations

import json


class BedrockModelClient:  # pragma: no cover - runs only against real Bedrock
    """Calls Amazon Bedrock and returns the model's JSON completion string.

    The generator validates whatever comes back through the loader schema, so a
    drifting or malformed completion raises rather than being committed.
    """

    def __init__(self, model_id: str, region: str | None = None) -> None:
        import boto3  # lazy: keeps boto3 out of the default import graph

        self._model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self._client.invoke_model(modelId=self._model_id, body=json.dumps(body))
        payload = json.loads(response["body"].read())
        # Claude-on-Bedrock returns content blocks; concatenate the text blocks.
        return "".join(
            block.get("text", "") for block in payload.get("content", []) if isinstance(block, dict)
        )
