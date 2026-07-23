"""Architectural invariants — the rules that CLAUDE.md says must never break.

These are cheap tests guarding expensive mistakes. The portability of ``core/``
is a stated strategic asset (CLAUDE.md §7): once an infrastructure import leaks
in, it leaks in everywhere and is painful to reverse.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "core"

#: Infrastructure that must never be imported from the portable core.
FORBIDDEN_IN_CORE = {"boto3", "botocore", "psycopg", "psycopg2", "fastapi", "uvicorn", "httpx"}


def _core_modules() -> list[pathlib.Path]:
    return sorted(CORE_DIR.rglob("*.py"))


def _imported_roots(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_core_has_modules():
    """Guard against the scan silently passing because it found nothing."""
    assert len(_core_modules()) >= 5


def test_core_never_imports_infrastructure():
    """core/ must not import boto3, psycopg or web frameworks — statically."""
    offenders: list[str] = []
    for module in _core_modules():
        forbidden = _imported_roots(module) & FORBIDDEN_IN_CORE
        if forbidden:
            offenders.append(f"{module.relative_to(REPO_ROOT)}: {sorted(forbidden)}")
    assert not offenders, (
        "core/ must stay portable (CLAUDE.md §7). Move infrastructure behind a "
        "StorageAdapter or an injected callback:\n  " + "\n  ".join(offenders)
    )


def test_core_imports_cleanly_without_infrastructure_modules():
    """Importing every core module must not pull infrastructure in at runtime.

    A subprocess is used so the check is unaffected by modules other tests have
    already imported into this interpreter.
    """
    program = f"""
import importlib, pathlib, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
forbidden = {sorted(FORBIDDEN_IN_CORE)!r}
for path in pathlib.Path({str(CORE_DIR)!r}).glob("*.py"):
    if path.stem != "__init__":
        importlib.import_module("core." + path.stem)
print(",".join(sorted(m for m in forbidden if m in sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip()
    assert leaked == "", f"importing core/ pulled in infrastructure modules: {leaked}"


def test_adapters_implement_the_frozen_protocol():
    """InMemoryAdapter must satisfy StorageAdapter structurally."""
    from adapters.memory_inmem import InMemoryAdapter
    from core.adapter import StorageAdapter

    assert isinstance(InMemoryAdapter(embedding_dim=4), StorageAdapter)


def test_embedding_dimension_matches_the_schema():
    """AletheiaConfig.embedding_dim and VECTOR(n) in infra/ddl.sql must agree.

    They are two halves of the same decision; drift between them silently breaks
    every vector write.
    """
    import re

    from core.config import AletheiaConfig

    ddl = (REPO_ROOT / "infra" / "ddl.sql").read_text()
    match = re.search(r"embedding\s+VECTOR\((\d+)\)", ddl)
    assert match, "no VECTOR(n) embedding column found in infra/ddl.sql"
    assert int(match.group(1)) == AletheiaConfig().embedding_dim


def test_ddl_never_deletes():
    """Nothing in the schema or its tooling may DELETE memory rows (CLAUDE.md §6a)."""
    ddl = (REPO_ROOT / "infra" / "ddl.sql").read_text().lower()
    assert "on delete cascade" not in ddl
    assert "delete from" not in ddl
