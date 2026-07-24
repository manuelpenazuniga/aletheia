"""Versioned experiment-arm registry (the project plan §8.2).

An *arm* is a named, frozen value: everything that distinguishes one run of an
experiment from another (which feature flags are on, which backend, how many
agents, which chaos event). The runner turns an arm into an
:class:`~core.config.AletheiaConfig` and a backend, runs it once per seed, and
records the result under ``experiment_runs.arm = arm.name``.

Two rules keep this module honest:

* **Pure data, no I/O and no numbers.** Nothing here reads a file, opens a
  connection or produces a metric. Resolving a ``backend`` string to a concrete
  storage factory is the *runner's* job, not this module's — that keeps the arm
  table trivially testable and diff-able.
* **The retrieval budget is never an arm variable (§8.7 control 1).** No arm may
  override ``retrieval_budget_tokens``; every arm inherits the base 4000 so a
  comparison measures design, not context volume. ``tests/test_arms.py`` fails
  loudly if this is ever broken.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.config import AletheiaConfig

#: Every experiment id the harness knows about (the project plan §8.1).
EXPERIMENTS: tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5", "E6")


@dataclass(frozen=True, slots=True)
class Arm:
    """One experimental condition — a frozen bundle of config overrides + wiring.

    ``name`` is the registry key and the value that lands in
    ``experiment_runs.arm``. It is deliberately kept distinct from
    :attr:`AletheiaConfig.ablation_name`: ``BL_rag`` disables all four mechanisms,
    which ``ablation_name`` would render as the long
    ``no_consolidation_forgetting_gossip_immune`` — useful in a flag report, but
    not the identity we want in the results table. The registry key is the
    identity; do not "fix" it to match the flag-derived string.
    """

    name: str
    experiments: tuple[str, ...]
    overrides: Mapping[str, Any] = field(default_factory=dict)
    backend: str = "aletheia"
    fleet_size: int | None = None
    event: str | None = None
    description: str = ""

    def config(self, base: AletheiaConfig | None = None) -> AletheiaConfig:
        """Materialise this arm's :class:`AletheiaConfig` on top of ``base``.

        The overrides are layered onto ``base`` (or a fresh default config) and
        the arm/backend are stamped into ``labels`` for the run record. ``labels``
        is replaced wholesale by ``with_overrides`` (intended); the arm's own
        ``overrides`` mapping is read, never mutated.
        """
        start = base or AletheiaConfig()
        return start.with_overrides(
            **dict(self.overrides),
            labels={"arm": self.name, "backend": self.backend},
        )


# The four flags that BL_rag switches off — the plain retrieve-and-answer baseline.
_ALL_MECHANISMS_OFF: Mapping[str, Any] = {
    "enable_consolidation": False,
    "enable_forgetting": False,
    "enable_gossip": False,
    "enable_immune": False,
}


def _e1_arms() -> list[Arm]:
    """E1 is systemic: the only variables are the backend and N. Flags stay at
    their defaults (all True) so the comparison isolates transactional storage."""
    arms: list[Arm] = []
    for backend, tag in (("crdb_serializable", "crdb_serializable"), ("naive_baseline", "naive")):
        for n in (5, 20, 50):
            arms.append(
                Arm(
                    name=f"E1_{tag}_n{n}",
                    experiments=("E1",),
                    backend=backend,
                    fleet_size=n,
                    description=f"E1 concurrency: {n} agents writing shared memory on {backend}.",
                )
            )
    return arms


#: The single source of truth, declared in stable reporting order. Ablation arms
#: first (E3-E5), then the systemic E1 grid, then E2 chaos, then E6.
ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in (
        Arm(
            name="A0_full",
            experiments=("E3", "E4", "E5"),
            overrides={},
            description="All four mechanisms on — the reference arm for every ablation.",
        ),
        Arm(
            name="A1_no_consolidation",
            experiments=("E3",),
            overrides={"enable_consolidation": False},
            description="Knowledge-update ablation: consolidation off, stale facts persist.",
        ),
        Arm(
            name="A2_no_forgetting",
            experiments=("E5",),
            overrides={"enable_forgetting": False},
            description="Cost ablation: metabolic forgetting off, footprint grows unbounded.",
        ),
        Arm(
            name="A4_no_immune",
            experiments=("E4",),
            overrides={"enable_immune": False},
            description="Poisoning ablation: immune gate off, the fleet adopts false facts.",
        ),
        Arm(
            name="BL_rag",
            experiments=("E3",),
            overrides=dict(_ALL_MECHANISMS_OFF),
            description=(
                "Plain retrieve-and-answer baseline (all four mechanisms off). NOT a "
                "single-flag ablation; the registry name is the identity, not the "
                "flag-derived ablation_name."
            ),
        ),
        *_e1_arms(),
        Arm(
            name="E2_crdb_3node_kill_node",
            experiments=("E2",),
            backend="crdb_3node",
            event="kill_node",
            fleet_size=20,
            description="E2 chaos: kill a node mid-consolidation with 20 agents writing.",
        ),
        Arm(
            name="E6_fleet_full",
            experiments=("E6",),
            fleet_size=None,
            description="E6: the full fleet with shared memory + gossip.",
        ),
        Arm(
            name="E6_bl_single",
            experiments=("E6",),
            fleet_size=1,
            description="E6 baseline: a single isolated agent, no shared memory.",
        ),
    )
}


def get_arm(name: str) -> Arm:
    """Look up an arm by name, or raise :class:`ValueError` listing the valid ones."""
    try:
        return ARMS[name]
    except KeyError:
        raise ValueError(f"unknown arm {name!r}; available arms: {list(ARMS)}") from None


def arms_for(exp: str) -> list[Arm]:
    """Every arm registered for experiment ``exp``, in declared order.

    Raises :class:`ValueError` (not ``KeyError``) on an unknown experiment id, so
    callers get the same failure mode as :func:`get_arm`.
    """
    if exp not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {exp!r}; known experiments: {list(EXPERIMENTS)}")
    return [arm for arm in ARMS.values() if exp in arm.experiments]
