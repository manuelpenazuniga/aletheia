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


def test_handlers_do_not_fake_work_while_unimplemented(monkeypatch):
    """Better an explicit NotImplementedError than an invented metric."""
    monkeypatch.setenv("ALETHEIA_ENABLE_CONSOLIDATION", "true")
    monkeypatch.setenv("ALETHEIA_ENABLE_GOSSIP", "true")
    with pytest.raises(NotImplementedError, match="Phase 1"):
        consolidation_handler.handler({}, None)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        gossip_handler.handler({}, None)
