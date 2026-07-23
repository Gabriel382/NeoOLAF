# Native NeoOLAF DocRED layer ablation

This package adds a one-document, layer-by-layer diagnostic for the first DocRED smoke-test document (`Skai TV`). It is designed to determine exactly where supported DocRED relations disappear inside the native NeoOLAF pipeline.

## Start here

Open:

```text
examples/RAGTreeDatasets/DocRED_NeoOLAF_Native_Layer_Ablation_SkaiTV.ipynb
```

Set `OPENROUTER_API_KEY` in the environment, then execute the notebook from the repository. The default model is `openai/gpt-oss-20b` through OpenRouter.

## Scientific constraints

The primary run:

- executes all 13 existing NeoOLAF components, Layer 0 through Layer 12;
- does not modify anything under `src/neoolaf`;
- does not call the direct DocRED extractor from the previous v6 experiment;
- does not expose the DocRED entity inventory or gold relations to the pipeline;
- does not use benchmark closure rules, relation invention, or a raw-text fallback extractor;
- uses gold data only after execution for strict scoring and failure tracing;
- maps only triples already produced by NeoOLAF to DocRED property IDs for evaluation.

## Added files

```text
examples/RAGTreeDatasets/
├── DocRED_NeoOLAF_Native_Layer_Ablation_SkaiTV.ipynb
├── README_DOCRED_NATIVE_ABLATION.md
├── configs/
│   ├── docred_profile_native_ablation.json
│   ├── docred_profile_native_ablation_relaxed.json
│   ├── guidance_docred_native_ablation.json
│   └── guidance_docred_native_ablation_relaxed.json
├── data/
│   ├── docred_skai_tv_input.jsonl
│   ├── docred_skai_tv_gold.jsonl
│   ├── docred_smoke5_input.jsonl
│   └── docred_smoke5_gold.jsonl
├── ontology/
│   ├── docred_redocred_original.ttl
│   ├── docred_redocred_neoolaf_compatible.ttl
│   ├── docred_relation_catalog.json
│   └── docred_relation_aliases.json
├── reference/
│   └── v6_calibrated_baseline_summary.json
└── tools/
    └── docred_native_ablation.py
```

## Ontology compatibility

The supplied Re-DocRED ontology defines its 96 predicates as `rdf:Property`. The current NeoOLAF `SeedOntologyLoader` indexes only `owl:ObjectProperty` and `owl:DatatypeProperty`, so loading the original file directly exposes zero relation properties to NeoOLAF.

The file `docred_redocred_neoolaf_compatible.ttl` preserves the original ontology and license text, then adds `owl:ObjectProperty` type assertions for the same 96 predicates. No domain, range, label, comment, or relation semantics are changed, and no source-code modification is required.

## Runtime artifacts

The balanced run writes to:

```text
examples/RAGTreeDatasets/runs/docred_native_layer_ablation/skai_tv_balanced/
```

It saves:

- restartable `state.json` after every layer;
- compact `output.json` and `metadata.json` after every layer;
- captured prompts and prompt statistics;
- raw LLM responses and call timing;
- ontology retrieval queries/results;
- console output and structured error logs;
- Layer 12 ontology/KG exports;
- `analysis/layer_summary.csv`;
- `analysis/gold_relation_trace.csv`;
- `analysis/native_mapped_predictions.jsonl`;
- `analysis/strict_docred_evaluation.json`;
- `analysis/analysis_summary.json`.

## Whole-document processing and workers

The Skai TV document fits safely in one chunk. The notebook chooses a single chunk dynamically, with a configurable safety ceiling of 24,000 characters. Layer 2 and Layer 3 use four workers by default. Later ontology-construction layers use existing deterministic NeoOLAF strategies to avoid unnecessary extra LLM calls while retaining the complete layer sequence.

## Previous v6 results

`reference/v6_calibrated_baseline_summary.json` records the previous five-document score and native layer counts. It also marks that the v6 benchmark-facing result used an additional direct extraction call in replacement mode plus calibration/closure, so it must not be interpreted as a native-only NeoOLAF result.
