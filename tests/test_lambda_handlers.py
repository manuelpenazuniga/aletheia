"""Lambda skeletons: the flag must be honoured before anything else happens.

The kill-switch is only credible if disabling a component actually stops its
scheduled job, so that behaviour is tested from the first commit.
"""

from __future__ import annotations

import pytest

from lambdas import consolidation_handler, gossip_handler


def test_consolidation_skips_when_the_flag_is_off(monkeypatch):
    monkeypatch.setenv("ALETHEIA_ENABLE_CONSOLIDATION", "false")
    result = consolidation_handler.handler({}, None)
    assert result == {"status": "skipped", "reason": "enable_consolidation=false"}


def test_gossip_skips_when_the_flag_is_off(monkeypatch):
    monkeypatch.setenv("ALETHEIA_ENABLE_GOSSIP", "0")
    result = gossip_handler.handler({}, None)
    assert result == {"status": "skipped", "reason": "enable_gossip=false"}


def test_consolidation_fails_clearly_without_a_dsn(monkeypatch):
    """Flag on but no database configured must fail loudly, not silently no-op."""
    monkeypatch.setenv("ALETHEIA_ENABLE_CONSOLIDATION", "true")
    monkeypatch.delenv("ALETHEIA_CRDB_DSN", raising=False)
    with pytest.raises(RuntimeError, match="ALETHEIA_CRDB_DSN"):
        consolidation_handler.handler({}, None)


def test_gossip_fails_clearly_without_a_dsn(monkeypatch):
    """Flag on but no database configured must fail loudly, not silently no-op."""
    monkeypatch.setenv("ALETHEIA_ENABLE_GOSSIP", "true")
    monkeypatch.delenv("ALETHEIA_CRDB_DSN", raising=False)
    with pytest.raises(RuntimeError, match="ALETHEIA_CRDB_DSN"):
        gossip_handler.handler({}, None)


def test_gossip_handler_imports_without_infrastructure():
    """The module must import with no psycopg/boto3/DSN present (lazy adapter)."""
    assert callable(gossip_handler.handler)
