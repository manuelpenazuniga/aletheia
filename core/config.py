"""Aletheia configuration and feature flags.

The contract in CLAUDE.md §7 is frozen here. Two rules govern this module:

1. **Feature flags first.** Every major component (consolidation, forgetting,
   gossip, immune) ships behind its flag from the first commit. The experimental
   ablations (§8.2) and the demo kill-switch (§9.1) are nothing more than these
   four booleans.
2. **One place for every knob.** Embedding dimension, model id, temperature and
   seed are read from here and nowhere else. `embedding_dim` in particular must
   match `VECTOR(1024)` in infra/ddl.sql — changing it rebuilds the index.

This module is part of the portable core: no boto3, no psycopg, no network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from typing import Any

# Bedrock model ids differ per region and inference-profile ids carry a `us.`
# prefix. CLAUDE.md §7 carries a placeholder ("anthropic.claude-sonnet-4-6") with
# an explicit instruction to verify the exact id in the Bedrock console; these
# defaults are the current best guess and MUST be confirmed for the target region
# before any experimental run (CLAUDE.md §11).
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

#: Embedding dimension of Amazon Titan Text Embeddings V2. Mirrored by
#: ``VECTOR(1024)`` in infra/ddl.sql. Single source of truth.
DEFAULT_EMBEDDING_DIM = 1024


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else raw


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class AletheiaConfig:
    """Immutable run configuration.

    Frozen on purpose: an experiment arm is a value, and a run must not be able
    to mutate its own configuration halfway through. Derive variants with
    :meth:`with_overrides`.
    """

    # --- feature flags (CLAUDE.md §7; the four ablation switches) -------------
    enable_consolidation: bool = True
    enable_forgetting: bool = True
    enable_gossip: bool = True
    enable_immune: bool = True

    # --- retrieval ------------------------------------------------------------
    #: Identical across every experimental arm (§8.7 control 1): if Aletheia
    #: retrieved more context than the baseline it would win on volume, not design.
    retrieval_budget_tokens: int = 4000

    # --- models ---------------------------------------------------------------
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    model_id: str = DEFAULT_MODEL_ID
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    temperature: float = 0.2

    # --- determinism ----------------------------------------------------------
    seed: int = 0

    # --- component knobs (extends §7; defaults are provisional until the
    #     components land in Phase 1-2) ----------------------------------------
    #: Global retention budget for metabolic forgetting (C3). Memories beyond it
    #: are archived to S3, never destroyed.
    memory_budget_tokens: int = 200_000
    #: Maximum gossip hops before a claim must be re-grounded (C4).
    gossip_max_hops: int = 3
    #: Cosine distance from an agent's own history beyond which a write is
    #: flagged as a semantic anomaly by the immune gate (C5).
    immune_anomaly_threshold: float = 0.85

    #: Free-form, non-behavioural labels (experiment arm name, run tags).
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_env(cls, **overrides: Any) -> AletheiaConfig:
        """Build from ``ALETHEIA_*`` environment variables (see .env.example).

        Explicit ``overrides`` win over the environment, which wins over defaults.
        """
        cfg = cls(
            enable_consolidation=_env_bool("ALETHEIA_ENABLE_CONSOLIDATION", True),
            enable_forgetting=_env_bool("ALETHEIA_ENABLE_FORGETTING", True),
            enable_gossip=_env_bool("ALETHEIA_ENABLE_GOSSIP", True),
            enable_immune=_env_bool("ALETHEIA_ENABLE_IMMUNE", True),
            retrieval_budget_tokens=_env_int("ALETHEIA_RETRIEVAL_BUDGET_TOKENS", 4000),
            embedding_dim=_env_int("ALETHEIA_EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM),
            model_id=_env_str("ALETHEIA_MODEL_ID", DEFAULT_MODEL_ID),
            embedding_model_id=_env_str(
                "ALETHEIA_EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID
            ),
            temperature=_env_float("ALETHEIA_TEMPERATURE", 0.2),
            seed=_env_int("ALETHEIA_SEED", 0),
        )
        return cfg.with_overrides(**overrides) if overrides else cfg

    def with_overrides(self, **overrides: Any) -> AletheiaConfig:
        """Return a copy with fields replaced — how experiment arms are derived."""
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ConfigError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **overrides)

    # -- validation -----------------------------------------------------------
    def validate(self) -> None:
        if self.embedding_dim <= 0:
            raise ConfigError("embedding_dim must be positive")
        if self.retrieval_budget_tokens <= 0:
            raise ConfigError("retrieval_budget_tokens must be positive")
        if self.memory_budget_tokens <= 0:
            raise ConfigError("memory_budget_tokens must be positive")
        if not 0.0 <= self.temperature <= 1.0:
            raise ConfigError("temperature must be in [0.0, 1.0]")
        if self.gossip_max_hops < 0:
            raise ConfigError("gossip_max_hops must be non-negative")
        if not self.model_id:
            raise ConfigError("model_id must not be empty")
        if not self.embedding_model_id:
            raise ConfigError("embedding_model_id must not be empty")

    # -- reporting ------------------------------------------------------------
    @property
    def ablation_name(self) -> str:
        """Canonical arm name derived from the flags — used in ``experiment_runs.arm``."""
        disabled = [
            name
            for name, on in (
                ("consolidation", self.enable_consolidation),
                ("forgetting", self.enable_forgetting),
                ("gossip", self.enable_gossip),
                ("immune", self.enable_immune),
            )
            if not on
        ]
        return "A0_full" if not disabled else "no_" + "_".join(disabled)

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot, recorded with every experiment run."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
