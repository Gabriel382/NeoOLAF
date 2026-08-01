# EventStoryLine native NeoOLAF v1.5 patch

This patch changes only experiment files under `examples/RAGTreeDatasets`.
Nothing under `src/neoolaf` is modified.

## Main corrections

- Atomic minimal EventStoryLine event-trigger prompts.
- Final source-only KEEP/DROP/REPLACE/SPLIT atomic span audit.
- Deterministic auxiliary/negation stripping before the atomic audit.
- Pair pool rebuilt after atomic refinement.
- Pair-local five-way relation classification.
- Explicit direct-link and no-transitive-closure rules.
- Annotation-aligned examples for headline/body elaboration, setup, aftermath,
  reversed direction, NONE, and transitive rejection.
- Targeted review of structurally plausible primary `NONE` decisions.
- Conservative positive verifier that may keep, reverse, reclassify, or reject.
- Positive evidence must cover both endpoint sentences.
- Candidate closure disabled to avoid transitive false positives.
- Existing exact-span projection, strict native metrics, candidate-pool analysis,
  confusion matrix, and full native Layers 0--12 are preserved.

## Run

Open:

`examples/RAGTreeDatasets/EventStoryLine_NeoOLAF_Native_Layer_Ablation_OneDoc_v1_5.ipynb`

The fresh run directory is:

`examples/RAGTreeDatasets/runs/eventstoryline_native_layer_ablation/document_1_10ecbplus_v1_5_owltime_atomic_verified`
