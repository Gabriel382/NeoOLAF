# EventStoryLine native one-document v1.4 patch

This patch is experiment-only. It does not modify any file under `src/neoolaf`.

## Relation-focused changes

- Keeps the RAGTree OWL-Time seed ontology and the controlled relations `PRECONDITION` and `FALLING_ACTION`.
- Adds a third whole-document event-mention coverage review.
- Replaces Layer-1 free relation generation with a deterministic unordered pair pool:
  - exhaustive for inventories with at most 25 events;
  - broad hybrid coverage for larger inventories.
- Uses zero relation-generation LLM calls in Layer 1.
- Classifies relation pairs in bounded Layer-2 batches with five direction-aware outcomes:
  `A_PRECONDITION_B`, `A_FALLING_ACTION_B`, `B_PRECONDITION_A`,
  `B_FALLING_ACTION_A`, or `NONE`.
- Adds exact local context, schema-exact positive/negative/reversed examples,
  evidence sentence IDs, grounded evidence text, recovery/adjudication, caching,
  optional OWL-Time evidence as secondary support, and deterministic conflict filtering.
- Removes `NONE`, invalid, missing-evidence and conflicting outputs before Layer 3.
- Supports two-hop candidate closure for large hybrid pair pools without automatically predicting closure relations.
- Disables global-trigger projection and reports both projected benchmark metrics and strict native span metrics.
- Adds candidate-pool recall, class/direction confusion, and detailed relation failure tracing.

## Main notebook

`examples/RAGTreeDatasets/EventStoryLine_NeoOLAF_Native_Layer_Ablation_OneDoc_v1_4.ipynb`

## Fresh run directory

`examples/RAGTreeDatasets/runs/eventstoryline_native_layer_ablation/document_1_10ecbplus_v1_4_owltime_batched_five_way`
