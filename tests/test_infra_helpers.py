"""Unit tests for the pure helpers in smoke.py and infra/apply_ddl.py.

These need no database — they cover the string handling that is easy to get
subtly wrong and security-relevant: DSN redaction and the VECTOR literal.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load("smoke_mod", REPO_ROOT / "smoke.py")
apply_ddl = _load("apply_ddl_mod", REPO_ROOT / "infra" / "apply_ddl.py")


# --------------------------------------------------------------- DSN redaction


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://user:s3cret@host:26257/aletheia?sslmode=verify-full",
        "postgresql://user@host/db?password=s3cret",
        "postgresql://host/db?sslmode=require&password=s3cret",
    ],
)
def test_redaction_hides_the_password(dsn):
    for redact in (smoke.redact, apply_ddl._redact):
        out = redact(dsn)
        assert "s3cret" not in out, f"{redact.__module__}.{redact.__name__} leaked the password"


def test_redaction_keeps_the_useful_parts():
    out = apply_ddl._redact("postgresql://user:pw@host:26257/aletheia?sslmode=verify-full")
    assert "user" in out and "host" in out and "aletheia" in out and "verify-full" in out


# ------------------------------------------------------------- VECTOR literal


def test_vector_literal_is_bracketed_and_comma_joined():
    assert smoke.to_vector_literal([1.0, -2.5, 0.0]) == "[1.0,-2.5,0.0]"


def test_vector_literal_coerces_to_float_and_has_no_injection_surface():
    # Every element goes through float(); a non-numeric element raises rather
    # than reaching the SQL string.
    out = smoke.to_vector_literal([1, 2, 3])
    assert out == "[1.0,2.0,3.0]"
    with pytest.raises((ValueError, TypeError)):
        smoke.to_vector_literal(["1); DROP TABLE memories;--"])


# --------------------------------------------------------- DSN db-name parsing


def test_target_database_defaults_to_aletheia():
    assert apply_ddl._target_database("postgresql://root@localhost:26257") == "aletheia"
    assert apply_ddl._target_database("postgresql://root@localhost:26257/mydb") == "mydb"


def test_swap_database_preserves_host_and_query():
    swapped = apply_ddl._swap_database(
        "postgresql://root@localhost:26257/aletheia?sslmode=disable", "defaultdb"
    )
    assert "/defaultdb" in swapped
    assert "localhost:26257" in swapped
    assert "sslmode=disable" in swapped
