"""Seeded incident-corpus generator + CLI.

The committed ``scenarios/incidents/sample.json`` is the frozen output of this
generator with ``--backend offline --n 10 --seed 0`` — bit-for-bit regenerable,
which a test asserts, so the corpus can never silently drift from its generator.

The full 100-incident corpus for the experiments (the project plan §8.4) is
produced ONCE with ``--backend bedrock`` and committed; it is not regenerated per
run and is out of scope for the default test path. The Bedrock backend imports
boto3 *lazily inside the backend function*, so importing this module and running
the default offline path never touch the cloud (the project plan §3.4 / no-cloud-
in-tests rule).

Every produced object — offline or Bedrock — is validated through
:func:`scenarios.loader` before it is returned, so a malformed completion raises
:class:`~scenarios.loader.ScenarioSchemaError` rather than being written to a
committed file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scenarios.loader import Incident, ScenarioSchemaError, _parse_incident
from scenarios.model_client import ModelClient, OfflineTemplateModel


def _incident_prompt(index: int, seed: int) -> str:
    """A deterministic per-incident prompt.

    Natural language so the Bedrock backend receives a usable instruction; the
    offline model uses only its bytes (plus the seed) as entropy. Encoding the
    index keeps each of the N prompts distinct, so the drawn incidents vary.
    """
    return (
        "You are generating one synthetic SRE incident for a training corpus. "
        "Return a single JSON object with keys: title, category, symptoms (array), "
        "fact_key ('runbook:<category>:fix'), runbook ({version:1, content, "
        "authored_by}), and revisions (array of {version, supersedes, content, "
        "reason}) with at least one revision that changes the runbook fix. "
        f"Incident #{index} of this batch (batch seed {seed})."
    )


def generate_incidents(n: int, seed: int, client: ModelClient) -> list[Incident]:
    """Generate ``n`` incidents deterministically under ``seed`` via ``client``.

    Each completion is parsed and validated through the loader schema (with the
    id assigned here, ``inc-{i:04d}``) before being collected, so nothing invalid
    is ever returned.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    incidents: list[Incident] = []
    for i in range(n):
        prompt = _incident_prompt(i, seed)
        raw = client.complete(prompt, seed=seed, max_tokens=1024)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ScenarioSchemaError(
                f"incident #{i}: model returned invalid JSON ({exc})"
            ) from exc
        if not isinstance(obj, dict):
            raise ScenarioSchemaError(f"incident #{i}: model returned a non-object")
        obj["id"] = f"inc-{i:04d}"
        incidents.append(_parse_incident(obj, f"generated[{i}]"))
    return incidents


def _incident_to_json(inc: Incident) -> dict[str, object]:
    """Serialise back to the committed JSON shape (embeddings never included)."""
    return {
        "id": inc.id,
        "title": inc.title,
        "category": inc.category,
        "symptoms": list(inc.symptoms),
        "fact_key": inc.fact_key,
        "runbook": {
            "version": inc.runbook.version,
            "content": inc.runbook.content,
            "authored_by": inc.runbook.authored_by,
        },
        "revisions": [
            {
                "version": r.version,
                "supersedes": r.supersedes,
                "content": r.content,
                "reason": r.reason,
            }
            for r in inc.revisions
        ],
    }


def dumps_incidents(incidents: list[Incident]) -> str:
    """Stable serialisation: sorted keys + trailing newline, for clean diffs."""
    payload = [_incident_to_json(inc) for inc in incidents]
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _make_bedrock_client(model_id: str, region: str | None) -> ModelClient:
    """Construct the Bedrock-backed client. boto3 is imported HERE, lazily, so the

    default offline path and every test never import it.
    """
    from scenarios._bedrock_client import BedrockModelClient  # local import on purpose

    return BedrockModelClient(model_id=model_id, region=region)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the SRE incident corpus.")
    parser.add_argument("--n", type=int, default=100, help="number of incidents")
    parser.add_argument("--seed", type=int, default=0, help="generation seed")
    parser.add_argument("--backend", choices=("offline", "bedrock"), default="offline")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scenarios/incidents/corpus.json"),
        help="output path (use scenarios/incidents/sample.json to regenerate the sample)",
    )
    parser.add_argument("--model-id", default=None, help="Bedrock model id (bedrock backend)")
    parser.add_argument("--region", default=None, help="AWS region (bedrock backend)")
    args = parser.parse_args(argv)

    if args.backend == "offline":
        client: ModelClient = OfflineTemplateModel()
    else:  # pragma: no cover - exercised only against real Bedrock, never in tests
        model_id = args.model_id or "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        client = _make_bedrock_client(model_id, args.region)

    incidents = generate_incidents(args.n, args.seed, client)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps_incidents(incidents), encoding="utf-8")
    print(f"wrote {len(incidents)} incidents to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
