"""Run the demo dashboard: ``python -m demoapp`` (offline seeded world).

Binds to ``DEMOAPP_HOST``/``DEMOAPP_PORT`` (defaults 127.0.0.1:8000). In
production (App Runner) the container command is the same; the CockroachDB +
Bedrock wiring is injected in ``create_app`` there rather than here, so this
entrypoint stays credential-free and always runnable.
"""

from __future__ import annotations

import os

import uvicorn

from demoapp.app import create_app


def main() -> None:
    host = os.environ.get("DEMOAPP_HOST", "127.0.0.1")
    port = int(os.environ.get("DEMOAPP_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
