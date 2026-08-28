# Unified4 v1.6 correction

MAVEN-ERE only. v1.5 preserved 100% Layer-1 event/endpoint-cluster recall on the fixed dev document but over-pruned relations to 0 TP / 3 FP / 14 FN. v1.6 keeps the v1.4 event inventory frozen and changes only relation reasoning.

- No relation-time coreference merge.
- Three complementary high-recall global graph proposals.
- Bounded source-cue rescue for candidate recall only.
- One PRECONDITION-aware calibrated adjudicator; no skeptic cascade.
- Strong implicit enabling/plan/process dependencies can count when grounded.
- Post-L12 only stage diagnostics measure gold pair/labeled recall at candidate, verifier-survivor, and final stages.
- No gold fields enter the pipeline.
- No `src/neoolaf` changes.
- Safe defaults: RUN_PAID=False, RUN_MODE=one_doc, RUN_DATASETS=[maven_ere].
