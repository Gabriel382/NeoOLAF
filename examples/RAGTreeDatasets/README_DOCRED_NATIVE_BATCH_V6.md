# DocRED native NeoOLAF batch v6

Main notebook:

`RAGTreeDatasets_DocRED_NativeNeoOLAF_Batch_v6_RunEval.ipynb`

This patch freezes the successful v5.1 single-document scientific configuration and applies it unchanged to the full DocRED JSONL used by the sibling RAGTree repository.

## Required external corpus

The notebook resolves the first existing path among:

- `../ragtree/data/preprocessed/docred_causal.jsonl`
- `../RAGTree/data/preprocessed/docred_causal.jsonl`
- `NeoOLAF/ragtree/data/preprocessed/docred_causal.jsonl`

The first candidate is represented in the notebook relative to `examples/RAGTreeDatasets` as `../../../ragtree/data/preprocessed/docred_causal.jsonl`.

## Smoke then full run

The only required switch is:

```python
RUN_ALL_DOCUMENTS = False  # first five
RUN_ALL_DOCUMENTS = True   # every JSONL record
```

Both modes use the same batch root. Completed document fingerprints are resumed, so the first five are not executed again during the full run.

## Parallelism

- `DOCUMENT_WORKERS = 4`: four isolated NeoOLAF document processes.
- `LAYER_WORKERS = 16`: v5.1 per-document Layer 2 workers.

Each document process has an independent run directory, stdout/error log, checkpoint set, raw responses, retrieval log, and evaluation artifacts. The process boundary is intentional because NeoOLAF redirects stdout while saving per-document logs; thread-level document launches would mix those streams.

## Evaluation outputs

Under `examples/RAGTreeDatasets/runs/docred_native_v5_1_batch/aggregate_analysis/`:

- `batch_summary.json`
- `per_document_metrics.csv`
- `per_relation_metrics.csv`
- `cumulative_layer_micro_evaluation.csv`
- `predictions.jsonl`
- `failed_documents.jsonl`

The aggregate report includes relation micro/macro metrics, entity-inventory metrics, relation-endpoint metrics, per-property metrics, cumulative layer contribution, timing, and first-failure counts.

## Scientific constraints

The pipeline input excludes gold `entities` and `relations`. The same frozen v5.1 task guidance, profile, ontology, relation catalog, and projection rules are used for every document. Gold is loaded only after execution. No direct DocRED extractor, source anchoring, post-run relation closure, or relation invention is enabled.

No file under `src/neoolaf` is changed by this patch.
