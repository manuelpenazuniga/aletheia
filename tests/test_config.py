"""AletheiaConfig: the frozen contract of CLAUDE.md §7."""

from __future__ import annotations

import pytest

from core.config import (
    DEFAULT_EMBEDDING_DIM,
    AletheiaConfig,
    ConfigError,
)

FLAGS = (
    "enable_consolidation",
    "enable_forgetting",
    "enable_gossip",
    "enable_immune",
)


def test_defaults_match_the_frozen_contract():
    cfg = AletheiaConfig()
    assert all(getattr(cfg, flag) is True for flag in FLAGS)
    assert cfg.retrieval_budget_tokens == 4000
    assert cfg.embedding_dim == DEFAULT_EMBEDDING_DIM == 1024
    assert cfg.temperature == 0.2
    assert cfg.seed == 0


def test_config_is_immutable():
    """An experiment arm is a value; a run must not mutate its own config."""
    cfg = AletheiaConfig()
    with pytest.raises(AttributeError):
        cfg.seed = 7  # type: ignore[misc]


def test_with_overrides_derives_an_arm_without_touching_the_original():
    base = AletheiaConfig()
    ablated = base.with_overrides(enable_consolidation=False)
    assert ablated.enable_consolidation is False
    assert base.enable_consolidation is True
    assert ablated.retrieval_budget_tokens == base.retrieval_budget_tokens


def test_with_overrides_rejects_unknown_fields():
    """A typo in an arm definition must fail loudly, not silently do nothing."""
    with pytest.raises(ConfigError, match="unknown config fields"):
        AletheiaConfig().with_overrides(enable_consolidaton=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_dim": 0},
        {"retrieval_budget_tokens": 0},
        {"memory_budget_tokens": -1},
        {"temperature": 1.5},
        {"temperature": -0.1},
        {"gossip_max_hops": -1},
        {"model_id": ""},
        {"embedding_model_id": ""},
    ],
)
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(ConfigError):
        AletheiaConfig(**kwargs)


def test_ablation_name_identifies_the_arm():
    assert AletheiaConfig().ablation_name == "A0_full"
    assert AletheiaConfig(enable_consolidation=False).ablation_name == "no_consolidation"
    assert (
        AletheiaConfig(enable_gossip=False, enable_immune=False).ablation_name == "no_gossip_immune"
    )


def test_from_env_reads_flags_and_budget(monkeypatch):
    monkeypatch.setenv("ALETHEIA_ENABLE_IMMUNE", "false")
    monkeypatch.setenv("ALETHEIA_ENABLE_GOSSIP", "0")
    monkeypatch.setenv("ALETHEIA_RETRIEVAL_BUDGET_TOKENS", "1234")
    monkeypatch.setenv("ALETHEIA_SEED", "42")
    monkeypatch.setenv("ALETHEIA_MODEL_ID", "some.model:1")

    cfg = AletheiaConfig.from_env()

    assert cfg.enable_immune is False
    assert cfg.enable_gossip is False
    assert cfg.enable_consolidation is True
    assert cfg.retrieval_budget_tokens == 1234
    assert cfg.seed == 42
    assert cfg.model_id == "some.model:1"


def test_from_env_rejects_a_malformed_boolean(monkeypatch):
    """A typo must fail loudly, not silently disable a security flag."""
    monkeypatch.setenv("ALETHEIA_ENABLE_IMMUNE", "treu")
    with pytest.raises(ConfigError, match="not a boolean"):
        AletheiaConfig.from_env()


def test_from_env_accepts_common_false_spellings(monkeypatch):
    for spelling in ("off", "no", "FALSE", "0"):
        monkeypatch.setenv("ALETHEIA_ENABLE_GOSSIP", spelling)
        assert AletheiaConfig.from_env().enable_gossip is False


def test_labels_are_read_only(monkeypatch):
    cfg = AletheiaConfig(labels={"arm": "A0"})
    assert cfg.labels["arm"] == "A0"
    with pytest.raises(TypeError):
        cfg.labels["arm"] = "tampered"  # type: ignore[index]


def test_to_dict_labels_is_a_detached_copy():
    cfg = AletheiaConfig(labels={"arm": "A0"})
    snapshot = cfg.to_dict()
    snapshot["labels"]["arm"] = "tampered"
    assert cfg.labels["arm"] == "A0"


def test_two_configs_do_not_share_a_labels_dict():
    shared = {"arm": "A0"}
    a = AletheiaConfig(labels=shared)
    b = AletheiaConfig(labels=shared)
    shared["arm"] = "mutated-after"
    assert a.labels["arm"] == "A0"
    assert b.labels["arm"] == "A0"


def test_from_env_ignores_empty_values(monkeypatch):
    """An unfilled line in .env must not override a default with garbage."""
    monkeypatch.setenv("ALETHEIA_MODEL_ID", "")
    monkeypatch.setenv("ALETHEIA_ENABLE_FORGETTING", "")
    cfg = AletheiaConfig.from_env()
    assert cfg.model_id == AletheiaConfig().model_id
    assert cfg.enable_forgetting is True


def test_explicit_overrides_beat_the_environment(monkeypatch):
    monkeypatch.setenv("ALETHEIA_SEED", "42")
    assert AletheiaConfig.from_env(seed=7).seed == 7


def test_to_dict_snapshot_is_complete():
    """Every run records its full configuration alongside its metrics."""
    snapshot = AletheiaConfig().to_dict()
    for flag in FLAGS:
        assert flag in snapshot
    assert snapshot["seed"] == 0
    assert snapshot["retrieval_budget_tokens"] == 4000
