"""Apply infra/ddl.sql to a CockroachDB instance (local docker or Cloud).

Usage:
    python infra/apply_ddl.py                  # uses ALETHEIA_CRDB_DSN from .env
    python infra/apply_ddl.py --dsn <dsn>
    python infra/apply_ddl.py --verify-only    # only report what exists

This is infrastructure tooling, not core: importing psycopg here is fine.
It is idempotent — every statement in ddl.sql is IF NOT EXISTS.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
import urllib.parse

DDL_PATH = pathlib.Path(__file__).with_name("ddl.sql")

EXPECTED_TABLES = [
    "agents",
    "memories",
    "canonical_facts",
    "provenance",
    "quarantine_log",
    "experiment_runs",
]


def _redact(dsn: str) -> str:
    """Never print a password. CLAUDE.md §3.4."""
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", dsn)


def _split_statements(sql: str) -> list[str]:
    """Split a DDL file into statements. Our DDL contains no semicolons inside
    string literals, so a top-level split is sufficient and gives per-statement
    error messages."""
    without_comments = "\n".join(
        line for line in sql.splitlines() if not line.strip().startswith("--")
    )
    return [s.strip() for s in without_comments.split(";") if s.strip()]


def _swap_database(dsn: str, dbname: str) -> str:
    parts = urllib.parse.urlsplit(dsn)
    return urllib.parse.urlunsplit(parts._replace(path=f"/{dbname}"))


def _target_database(dsn: str) -> str:
    path = urllib.parse.urlsplit(dsn).path.lstrip("/")
    return path or "aletheia"


def ensure_database(dsn: str) -> str:
    """Create the target database if missing, connecting via `defaultdb`."""
    import psycopg

    dbname = _target_database(dsn)
    bootstrap_dsn = _swap_database(dsn, "defaultdb")
    with psycopg.connect(bootstrap_dsn, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE IF NOT EXISTS "{dbname}"')
    print(f"[apply_ddl] database ready: {dbname}")
    return dbname


def apply_ddl(dsn: str) -> None:
    import psycopg

    statements = _split_statements(DDL_PATH.read_text())
    with psycopg.connect(dsn, autocommit=True) as conn:
        for stmt in statements:
            label = " ".join(stmt.split())[:70]
            try:
                conn.execute(stmt)
            except psycopg.errors.FeatureNotSupported as exc:
                # Vector indexing was behind a cluster setting before it went GA.
                # Enable it explicitly rather than silently skipping the index.
                if "vector" not in stmt.lower():
                    raise
                print(f"[apply_ddl] vector index rejected ({exc}); enabling feature flag")
                conn.execute("SET CLUSTER SETTING feature.vector_index.enabled = true")
                conn.execute(stmt)
            print(f"[apply_ddl] ok: {label}")


def verify(dsn: str) -> bool:
    import psycopg

    ok = True
    with psycopg.connect(dsn, autocommit=True) as conn:
        rows = conn.execute("SELECT table_name FROM [SHOW TABLES] ORDER BY table_name").fetchall()
        present = {r[0] for r in rows}
        for table in EXPECTED_TABLES:
            mark = "ok " if table in present else "MISSING"
            print(f"[verify] table {table:<16} {mark}")
            ok &= table in present

        idx = conn.execute(
            "SELECT index_name FROM [SHOW INDEXES FROM memories] WHERE index_name = 'idx_mem_embedding'"
        ).fetchall()
        print(f"[verify] vector index idx_mem_embedding {'ok ' if idx else 'MISSING'}")
        ok &= bool(idx)

        version = conn.execute("SELECT version()").fetchone()[0]
        print(f"[verify] server: {version.split(' (')[0]}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=None, help="defaults to $ALETHEIA_CRDB_DSN")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        pass

    dsn = args.dsn or os.environ.get("ALETHEIA_CRDB_DSN")
    if not dsn:
        print(
            "error: no DSN. Set ALETHEIA_CRDB_DSN in .env (copy .env.example) or pass --dsn.",
            file=sys.stderr,
        )
        return 2

    print(f"[apply_ddl] target: {_redact(dsn)}")
    if not args.verify_only:
        ensure_database(dsn)
        apply_ddl(dsn)
    return 0 if verify(dsn) else 1


if __name__ == "__main__":
    raise SystemExit(main())
