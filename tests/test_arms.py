"""The arm registry is pure data — these tests assert flags, membership, the
budget-invariance control (§8.7), label stamping and lookup errors. No I/O."""

from __future__ import annotations

import pytest

from core.config import AletheiaConfig
from experiments.arms import ARMS, EXPERIMENTS, arms_for, get_arm


def test_arms_flags():
    """Each named ablation flips exactly the flags it claims, and no others."""
    a0 = get_arm("A0_full").config()
    assert (a0.enable_consolidation, a0.enable_forgetting, a0.enable_gossip, a0.enable_immune) == (
        True,
        True,
        True,
        True,
    )

    a1 = get_arm("A1_no_consolidation").config()
    assert a1.enable_consolidation is False
    assert (a1.enable_forgetting, a1.enable_gossip, a1.enable_immune) == (True, True, True)

    a2 = get_arm("A2_no_forgetting").config()
    assert a2.enable_forgetting is False
    assert (a2.enable_consolidation, a2.enable_gossip, a2.enable_immune) == (True, True, True)

    a4 = get_arm("A4_no_immune").config()
    assert a4.enable_immune is False
    assert (a4.enable_consolidation, a4.enable_forgetting, a4.enable_gossip) == (True, True, True)

    rag = get_arm("BL_rag").config()
    assert (
        rag.enable_consolidation,
        rag.enable_forgetting,
        rag.enable_gossip,
        rag.enable_immune,
    ) == (
        False,
        False,
        False,
        False,
    )


def test_bl_rag_name_is_not_the_flag_derived_ablation_name():
    """BL_rag keeps its registry identity even though its flags render a long name."""
    arm = get_arm("BL_rag")
    assert arm.name == "BL_rag"
    # The flag-derived reporting string is deliberately NOT the arm key.
    assert arm.config().ablation_name == "no_consolidation_forgetting_gossip_immune"


def test_arms_budget_invariant():
    """§8.7 control 1: no arm may override the retrieval budget."""
    base = AletheiaConfig().retrieval_budget_tokens
    for arm in ARMS.values():
        assert arm.config().retrieval_budget_tokens == base
        assert "retrieval_budget_tokens" not in arm.overrides


def test_e1_arms_leave_flags_default():
    """E1 is systemic: the only variables are backend and N, never the flags."""
    e1 = arms_for("E1")
    assert len(e1) == 6
    assert {a.backend for a in e1} == {"crdb_serializable", "naive_baseline"}
    assert {a.fleet_size for a in e1} == {5, 20, 50}
    for arm in e1:
        cfg = arm.config()
        assert (
            cfg.enable_consolidation
            and cfg.enable_forgetting
            and cfg.enable_gossip
            and cfg.enable_immune
        )


def test_arms_for_membership():
    assert {a.name for a in arms_for("E3")} == {"A0_full", "A1_no_consolidation", "BL_rag"}
    assert {a.name for a in arms_for("E4")} == {"A0_full", "A4_no_immune"}
    assert {a.name for a in arms_for("E5")} == {"A0_full", "A2_no_forgetting"}
    assert {a.name for a in arms_for("E6")} == {"E6_fleet_full", "E6_bl_single"}

    e2 = arms_for("E2")
    assert len(e2) == 1
    assert e2[0].name == "E2_crdb_3node_kill_node"
    assert e2[0].event == "kill_node"
    assert e2[0].backend == "crdb_3node"


def test_arms_for_declared_order():
    """arms_for preserves the registry's declared order."""
    names = [a.name for a in arms_for("E3")]
    assert names == ["A0_full", "A1_no_consolidation", "BL_rag"]


def test_arm_labels_stamped():
    for arm in ARMS.values():
        cfg = arm.config()
        assert cfg.labels["arm"] == arm.name
        assert cfg.labels["backend"] == arm.backend


def test_config_does_not_mutate_shared_overrides():
    """Materialising a config must not reach back into the frozen arm."""
    arm = get_arm("A1_no_consolidation")
    before = dict(arm.overrides)
    arm.config()
    arm.config(AletheiaConfig(seed=7))
    assert dict(arm.overrides) == before


def test_config_layers_on_base():
    """Overrides layer onto a caller-supplied base config (e.g. a seeded run)."""
    cfg = get_arm("A1_no_consolidation").config(AletheiaConfig(seed=3))
    assert cfg.seed == 3
    assert cfg.enable_consolidation is False


def test_get_arm_unknown_raises():
    with pytest.raises(ValueError, match="unknown arm"):
        get_arm("nope")


def test_arms_for_unknown_raises_valueerror_listing_experiments():
    with pytest.raises(ValueError, match="unknown experiment") as exc:
        arms_for("E9")
    for exp in EXPERIMENTS:
        assert exp in str(exc.value)


def test_every_arm_belongs_to_a_known_experiment():
    for arm in ARMS.values():
        assert arm.experiments
        assert set(arm.experiments) <= set(EXPERIMENTS)
