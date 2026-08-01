# EventStoryLine native NeoOLAF v1.6 patch

Extract this archive at the NeoOLAF repository root and open:

`examples/RAGTreeDatasets/EventStoryLine_NeoOLAF_Native_Layer_Ablation_OneDoc_v1_6.ipynb`

## Scientific change

The v1.5 atomic event inventory and deterministic unordered relation-pair pool are retained. Only Layer 2 experiment orchestration changes:

- one fixed source event per primary request;
- every candidate target receives `PRECONDITION`, `FALLING_ACTION`, or `NONE`;
- one compact retry only for target IDs omitted from a source response;
- evidence sentence/text formatting is repaired without deleting a valid relation decision;
- only opposite-direction positive conflicts receive compact adjudication;
- no broad false-`NONE` review;
- no destructive positive verifier;
- no candidate closure;
- `NONE` remains filtered before native Layer 3.

The prompt contains small synthetic source-centric examples for PRECONDITION, FALLING_ACTION, NONE, narrative annotation direction, headline/body development, and rejection of transitive chains. Gold events and gold relations are not available until post-Layer-12 evaluation.

## Fresh run directory

`examples/RAGTreeDatasets/runs/eventstoryline_native_layer_ablation/document_1_10ecbplus_v1_6_owltime_source_centric`

No file under `src/neoolaf` is included or modified by this patch.
