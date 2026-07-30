# EventStoryLine native NeoOLAF v1.1 patch

This patch corrects the one-document EventStoryLine experiment without changing
anything under `src/neoolaf`.

## Corrections

1. Uses the same external OWL-Time seed ontology as RAGTree:
   `../ragtree/data/ontology/OWLTime/time.ttl`.
2. Resolves the ontology through robust sibling-repository candidates or the
   `EVENTSTORYLINE_ONTOLOGY_PATH` environment variable.
3. Keeps `PRECONDITION` and `FALLING_ACTION` as the controlled normalized task
   relation schema. They are mapping/evaluation targets and may be induced as
   NeoOLAF ontology deltas; they are not falsely presented as OWL-Time seed
   properties.
4. Removes the invalid assertion that the seed ontology must contain exactly two
   properties. The notebook now reports the actual OWL-Time class/property count.
5. Fixes analysis for upstream `LAYER_NAMES` represented as a list instead of a
   dictionary. The former `LAYER_NAMES.get(...)` call caused:
   `AttributeError: 'list' object has no attribute 'get'`.
6. Uses a new run directory so the OWL-Time experiment does not reuse artifacts
   produced with the former task-only seed ontology.

## Notebook

`examples/RAGTreeDatasets/EventStoryLine_NeoOLAF_Native_Layer_Ablation_OneDoc_v1_1.ipynb`
