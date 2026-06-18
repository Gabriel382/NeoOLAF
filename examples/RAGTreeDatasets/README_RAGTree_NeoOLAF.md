# NeoOLAF RAGTree datasets benchmark files

Copy this folder tree at the root of the NeoOLAF repository.

Main files:

```text
examples/RAGTreeDatasets/RAGTreeDatasets_NeoOLAF.ipynb
examples/RAGTreeDatasets/configs/guidance_docred.json
examples/RAGTreeDatasets/configs/guidance_causalbank.json
examples/RAGTreeDatasets/configs/guidance_eventstoryline.json
examples/RAGTreeDatasets/configs/guidance_fincausal.json
examples/RAGTreeDatasets/configs/guidance_maven_ere.json
experiments/methods/run_neoolaf.py
experiments/methods/README_neoolaf_document_parallel.md
```

The notebook has DocRED active by default. The other four datasets are present but commented.

The runner supports:

```bash
--document-workers N
```

This controls how many documents are processed in parallel. Start with `--document-workers 2` and increase only if the provider does not rate-limit.

## Smoke-test evaluation warning

When `MAX_DOCS = 5`, the notebook now creates a matching smoke-test gold JSONL under `./runs/`. This prevents the evaluator from comparing five NeoOLAF predictions against the full DocRED development split, which would incorrectly inflate `missing_predictions`.

The run also writes:

```text
./runs/neoolaf_docred_run_summary.json
./runs/neoolaf_docred_errors.jsonl
```

Use these files before checking F1. If `relations=0`, inspect the run summary and the document-level error reports first.
