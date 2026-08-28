# Unified4 v1.9 MAVEN relation patch

## What changes
- Layer 1 remains the frozen/reused v1.4 43-mention event inventory.
- Relation-time soft coreference is conservative, source-only, and non-destructive.
- Deterministic radius-4 group-pair coverage is retained in both directions.
- Pair-independent classification is replaced by source-centric multi-target relation existence selection.
- Relation existence and relation subtype are separated: selected pairs are labeled PRECONDITION vs CAUSE in a second label-only pass.
- Only one canonical source mention and target mention are emitted per selected soft-group relation, reducing duplicate relation explosion.
- No gold annotations are available to the pipeline; member-aware gold diagnostics run only post-L12.

## Execution guard
The notebook forces MAVEN only for `RUN_MODE="one_doc"` so this explicit refinement can run despite v1.8 having marked READY_5. It never force-bypasses smoke5/full safeguards.


## Hotfix
Restores `offline_maven_rescue_candidate_preview`, which the notebook preflight calls. The missing re-export caused an AttributeError before any paid run.
