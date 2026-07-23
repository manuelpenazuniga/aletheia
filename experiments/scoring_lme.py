"""Scoring adapter for the LongMemEval subset (experiment E3b).

Placeholder: implemented in Phase 3 (CLAUDE.md §10), after E3 closes on the
synthetic incident corpus. The module exists from now so the layout matches
CLAUDE.md §8.4 and so the open questions are recorded where they will be needed.

What this module will do
------------------------
Translate between LongMemEval's evaluation format and Aletheia's, for the
*knowledge-update* and *temporal-reasoning* categories only:

* load the subset pinned in ``scenarios/longmemeval/SUBSET.md`` (IDs, not a
  sampling procedure — the subset must not be re-rolled between arms or seeds);
* map benchmark sessions onto ``MemoryEvent`` writes with provenance. This is the
  unresolved design question: LongMemEval has no notion of which agent observed
  what, and Aletheia's whole write path is built on signed provenance. The
  mapping has to be decided and documented before any number is produced;
* score answers per the benchmark's own criteria, so the figure is comparable to
  published results rather than to a metric of our own invention;
* emit metrics into the same ``experiment_runs`` row shape as every other
  experiment, tagged with the arm and seed.

TODO(phase-3):
    - pin the benchmark release/commit in SUBSET.md and record it in the metrics
    - decide and document the session -> provenance mapping
    - implement load_subset(), run_arm() and score()
    - verify the license permits committing any benchmark content (see SUBSET.md)

Deliberately unimplemented rather than stubbed with plausible-looking numbers:
an invented external-validation figure would be the single most damaging thing
this project could publish (CLAUDE.md §3.1).
"""

from __future__ import annotations

__all__: list[str] = []
