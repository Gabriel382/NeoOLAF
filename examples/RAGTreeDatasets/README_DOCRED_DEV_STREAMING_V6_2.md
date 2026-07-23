# DocRED dev-only bounded streaming batch — v6.2

This patch keeps the frozen NeoOLAF v5.1 scientific configuration and the v6.1
analysis recovery fix. It changes only orchestration and aggregation.

## Main notebook

`examples/RAGTreeDatasets/RAGTreeDatasets_DocRED_NativeNeoOLAF_DevStreaming_v6_2_RunEval.ipynb`

## Workflow

1. Keep `RUN_ALL_DEV_DOCUMENTS = False` to run the first five records whose exact
   JSON key `type` equals `dev`.
2. Inspect the strict relation/entity evaluation.
3. Change only `RUN_ALL_DEV_DOCUMENTS = True`.
4. Keep the same batch root and rerun the batch cell.

## Memory behavior

- The source JSONL is read one line at a time.
- No complete list of dev records is built.
- The parent process holds at most `DOCUMENT_WORKERS` submitted document jobs.
- `LAYER_WORKERS` remains active inside each document process.
- Aggregate evaluation reads one saved document result at a time and keeps only
  scalar counters, the 96 relation buckets, 13 layer buckets, and numeric timing
  values in memory.

## Exact filtering

The filter is deliberately strict:

```python
str(record.get("type") or "").strip().lower() == "dev"
```

It does not fall back to a `split` field.

## Files changed

Only files under `examples/RAGTreeDatasets` are added or updated. No file under
`src/neoolaf` is modified.
