# Unified4 v1.10 Patch Notes

MAVEN-ERE only. This version intentionally returns to the v1.8 relation architecture.

- Frozen/reused v1.4 Layer 1.
- v1.8 all ordered mention pairs within sentence distance <= 4; no pre-classification cap or textual direction filter.
- v1.8 pairwise PRECONDITION/CAUSE/REJECT adjudication is preserved.
- NEW: one conservative document-level graph review may only REMOVE obvious false positives.
- The graph reviewer cannot add, relabel, reverse, or replace edges; missing/uncertain decisions default to KEEP.
- Removal requires confidence >= 0.86 and one of four explicit false-positive categories.
- A deterministic safety cap prevents removal of more than 40% of pre-review edges.
- Gold remains unavailable until post-L12 evaluation.
- No files under src/neoolaf are changed.
