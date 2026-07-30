# EventStoryLine native NeoOLAF one-document experiment v1.2

This experiment keeps the complete native NeoOLAF Layer 0--12 pipeline and the
same OWL-Time seed ontology used by RAGTree. Gold annotations remain unavailable
until post-run evaluation.

## Main correction

Layer 1 is split into two experiment-level phases without modifying `src/neoolaf`:

1. **Layer 1A event inventory**: one exhaustive structured event extraction call.
   Every candidate is validated against the source sentence/token arrays. A
   bounded deterministic repair aligns complete trigger text, completes common
   phrasal verbs, and completes tokenized hyphen compounds. Function-word,
   connective-only, and auxiliary-only spans are rejected.
2. **Layer 1B relation discovery**: a second call receives only the validated
   closed event inventory and may only reference its local event IDs. It cannot
   create or alter event spans.

Layer 2 still performs the controlled contrastive choice among `PRECONDITION`,
`FALLING_ACTION`, and `found=false`. Layers 3--12 are unchanged.

## Notebook

`examples/RAGTreeDatasets/EventStoryLine_NeoOLAF_Native_Layer_Ablation_OneDoc_v1_2.ipynb`

## New logs

- `run_logs/layer01_event_inventory.json`
- `run_logs/layer01_relation_generation.json`
- `run_logs/layer01_event_relation_instances.json`

The first two logs expose all span repairs/rejections and all closed-inventory
pair decisions for scientific auditing.
