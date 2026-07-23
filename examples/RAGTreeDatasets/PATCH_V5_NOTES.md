# DocRED native ablation patch v5

This patch keeps every file under `src/neoolaf` unchanged and adds only
experiment-side files under `examples/RAGTreeDatasets`.

## Changes

- One whole-document Layer 1 call now performs a country-coverage self-check.
- Country relations must originate in Layer 1; no post-run closure creates them.
- Layer 2 uses ontology RAG but sends only a top-five compact property shortlist.
- Layer 2 prompts contain relation-specific rules and at most two examples.
- Transparent profile guardrails canonicalize existing relation instances:
  - corporate/media-group `part of` -> P127;
  - explicit subsidiary/branch/division/lab -> P749;
  - organization base or relaunch city -> P159;
  - entity establishment/relaunch date -> P571;
  - non-human entity/place -> country -> P17.
- Guardrails never create a new relation instance and every override is logged.
- Mention-aware entity projection runs only after the pipeline for benchmark
  evaluation. It does not modify native artifacts.
- Layer 2 defaults to 16 workers and all relation calls can run concurrently for
  the one-document smoke test.

## Main files

- `DocRED_NeoOLAF_Native_Layer_Ablation_SkaiTV_v5.ipynb`
- `tools/docred_native_ablation_v5.py`
- `configs/docred_profile_native_ablation_v5.json`
- `configs/guidance_docred_native_ablation_v5.json`
- `data/docred_skai_tv_input_v5.jsonl`

## New runtime artifacts

- `run_logs/layer01_country_coverage_audit.json`
- `run_logs/layer02_compact_prompt_audit.json`
- `analysis/entity_projection_audit_v5.csv`
- `analysis/gold_relation_trace_v5.csv`
- `analysis/cumulative_strict_evaluation_v5.csv`
- `analysis/strict_docred_predictions_v5.csv`
