"""Write path: FastAPI ingest service fronted by the immune gate.

``create_app`` is re-exported for convenience; ``ingest.main`` holds the
production wiring (the only place psycopg/boto3 enter the write path).
"""

from ingest.app import create_app

__all__ = ["create_app"]
