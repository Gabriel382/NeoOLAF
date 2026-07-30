# EventStoryLine native NeoOLAF experiment v1

This experiment runs the full native NeoOLAF Layers 0--12 without changing
`src/neoolaf`.

Files:

- `EventStoryLine_NeoOLAF_Native_Layer_Ablation_OneDoc_v1.ipynb`: one-document test;
- `data/eventstoryline_one_input_v1.jsonl`: first document without gold;
- `data/eventstoryline_one_gold_v1.jsonl`: first document gold, evaluation only;
- `data/eventstoryline_smoke5_input_v1.jsonl`: five documents without gold;
- `data/eventstoryline_smoke5_gold_v1.jsonl`: the corresponding five-document gold;
- `configs/eventstoryline_profile_native_ablation_v1.json`: document profile;
- `configs/guidance_eventstoryline_native_ablation_v1.json`: NeoOLAF UserGuidance;
- `configs/eventstoryline_task_guidance_v1.json`: relation definitions, direction rules, and synthetic examples;
- `ontology/eventstoryline_neoolaf_seed.ttl`: schema-only ontology with Event, PRECONDITION, and FALLING_ACTION;
- `tools/eventstoryline_native_ablation_v1.py`: runner and strict evaluator.

The dataset `null` relation key is ignored. Event endpoints use the deterministic
key `S{sentence}[start:end]::trigger`, derived only from the public sentence/token
structure. Gold event IDs and pairs are never passed to NeoOLAF.
