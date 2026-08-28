# EventStoryLine native NeoOLAF v1.6.2 — portable examples bundle

This bundle contains every experiment-side Python helper required by the v1.6 source-centric EventStoryLine notebook. It is intended for a NeoOLAF checkout on another PC and does **not** modify `src/neoolaf`.

## Install

Extract this ZIP at the NeoOLAF repository root so the files land under:

`examples/RAGTreeDatasets/`

Open:

`examples/RAGTreeDatasets/EventStoryLine_NeoOLAF_Native_Layer_Ablation_OneDoc_v1_6_2.ipynb`

## External data expected

The notebook automatically checks the Windows dataset directory:

`C:\Users\galencarmedeiro\Documents\git\postdoc\RAGTree\preprocessed`

and the portable sibling-repository form `../RAGTree/preprocessed`.

It expects the five files:

- `causalbank.jsonl`
- `docred_causal.jsonl`
- `eventstoryline.jsonl`
- `fincausal.jsonl`
- `maven_ere.jsonl`

The EventStoryLine seed ontology remains the same OWL-Time file used by RAGTree and is expected under the sibling RAGTree repository, normally:

`C:\Users\galencarmedeiro\Documents\git\postdoc\RAGTree\data\ontology\OWLTime\time.ttl`

You can override the locations with `RAGTREE_PREPROCESSED_DIR` and `EVENTSTORYLINE_ONTOLOGY_PATH`.

## Included local dependency chain

The notebook imports `eventstoryline_native_ablation_v1_6`, which depends on:

`v1_6 -> v1_5 -> v1_3 -> docred_native_ablation + docred_native_ablation_v3 + docred_native_ablation_v4`

All of those helper modules are included in `examples/RAGTreeDatasets/tools/` in this ZIP.

The controlled EventStoryLine relation catalog/aliases and all v1.6 profile/guidance JSON files are also included.
