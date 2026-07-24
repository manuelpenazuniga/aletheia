"""Seeded, committed datasets: incident corpus, distributed clues, poison suite.

Public API is the typed loader — read the datasets only through it, so every
record is validated against the schema before use.
"""

from scenarios.loader import (
    COMMITTED_INCIDENTS_PATH,
    DISTRIBUTED_CLUES_PATH,
    POISON_DIR,
    Incident,
    PoisonCase,
    PoisonSuite,
    Provenance,
    Revision,
    RunbookVersion,
    ScenarioSchemaError,
    incident_to_events,
    load_incidents,
    load_poison_cases,
    load_poison_suite,
    poison_to_event,
)

__all__ = [
    "COMMITTED_INCIDENTS_PATH",
    "DISTRIBUTED_CLUES_PATH",
    "POISON_DIR",
    "Incident",
    "PoisonCase",
    "PoisonSuite",
    "Provenance",
    "Revision",
    "RunbookVersion",
    "ScenarioSchemaError",
    "incident_to_events",
    "load_incidents",
    "load_poison_cases",
    "load_poison_suite",
    "poison_to_event",
]
