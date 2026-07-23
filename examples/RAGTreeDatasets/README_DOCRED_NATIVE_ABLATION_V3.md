# Native NeoOLAF DocRED layer ablation v3

Open:

```text
examples/RAGTreeDatasets/DocRED_NeoOLAF_Native_Layer_Ablation_SkaiTV_v3.ipynb
```

The experiment runs one complete DocRED document (`Skai TV`) through native NeoOLAF Layers 0–12.

## What changed in v3

- The JSONL input contains a `task_guidance` metadata object describing the target relation schema, relation direction, controlled inference policy, and synthetic examples.
- The metadata contains no gold entity clusters or gold relation pairs and is not prepended to the article text.
- User guidance explicitly requires `controlled_relation:Pxxx : label` and `promote_to_ontology:true` hints for relation-bearing expressions.
- The profile supplies 21 priority DocRED relation schemas and promotes relation schemas, while named entities remain instances.
- Layer 1 uses one whole-document chunk.
- Layer 2 uses 12 workers, a 768-token output limit, a 90-second request timeout, and one retry.
- Layer 4 performs the same native NeoOLAF relation task with 8 concurrent workers, a 512-token output limit, a 90-second timeout, and at most two attempts per relation mention.
- Response-cap violations and JSON parsing failures are logged separately.
- Evaluation distinguishes ontology schema availability from an actual relation mention linked to that schema.

## Scientific constraints

The experiment:

- changes no file under `src/neoolaf`;
- does not run the previous direct DocRED extractor;
- does not provide source/gold entity anchoring;
- does not add post-extraction relation closure;
- does not use gold-derived examples or relation pairs;
- loads gold only after the run for strict evaluation and diagnosis;
- maps only native NeoOLAF relation candidates/triples to canonical property IDs.

The experiment-side Layer 4 subclass changes orchestration only: it executes the existing NeoOLAF prompt and one-assertion-per-relation-mention task concurrently. It does not introduce a second extraction task.

## Primary files

```text
examples/RAGTreeDatasets/
├── DocRED_NeoOLAF_Native_Layer_Ablation_SkaiTV_v3.ipynb
├── PATCH_V3_NOTES.md
├── README_DOCRED_NATIVE_ABLATION_V3.md
├── configs/
│   ├── docred_profile_native_ablation_v3.json
│   └── guidance_docred_native_ablation_v3.json
├── data/
│   ├── docred_skai_tv_input.jsonl
│   └── docred_skai_tv_gold.jsonl
├── ontology/
│   ├── docred_redocred_original.ttl
│   ├── docred_redocred_neoolaf_compatible.ttl
│   ├── docred_relation_catalog.json
│   └── docred_relation_aliases.json
└── tools/
    ├── docred_native_ablation.py
    └── docred_native_ablation_v3.py
```

## Runtime artifacts

The default run directory is:

```text
examples/RAGTreeDatasets/runs/docred_native_layer_ablation/skai_tv_guided_parallel_v3/
```

It stores:

- the effective merged guidance and original input task metadata;
- layer state/output/metadata/prompt artifacts;
- raw provider responses;
- API errors, JSON parse errors, response-cap violations, and Layer 4 errors;
- ontology retrieval queries and selected properties;
- native lexical triples;
- ontology-canonical triples;
- strict DocRED predictions;
- predictions absent from gold, marked for manual review;
- cumulative strict metrics for every layer;
- a relation-instance trace showing the first failure stage.
