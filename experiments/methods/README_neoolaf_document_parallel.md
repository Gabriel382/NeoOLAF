# NeoOLAF RAGTree benchmark runner with document-level parallelism

This patch adds a drop-in replacement for:

```bash
../../experiments/methods/run_neoolaf.py
```

The runner uses the existing NeoOLAF library and keeps one full pipeline execution per document. It adds the new option:

```bash
--document-workers N
```

`--document-workers` controls how many dataset documents are processed in parallel. The existing `--max-workers` option is kept for intra-document/chunk-level compatibility.

Recommended first test:

```bash
--document-workers 2 \
--max-workers 1 \
--max-docs 5
```

Then increase to 4 or 8 depending on provider rate limits.

Example:

```bash
python ../../experiments/methods/run_neoolaf.py \
  --dataset-jsonl-path "../../../ragtree/data/preprocessed/docred_causal.jsonl" \
  --ontology-path "../../../ragtree/data/ontology/DocREDOntology/ontology.ttl" \
  --output-jsonl-path "./runs/neoolaf_docred_predictions.docred_constrained.canonical.jsonl" \
  --backend-name openrouter \
  --host "https://openrouter.ai/api" \
  --api-key "$OPENROUTER_API_KEY" \
  --model-name "openai/gpt-oss-20b" \
  --type-filter dev \
  --user-guidance-path "./configs/guidance_docred.json" \
  --few-shot-from-dataset \
  --few-shot-source-type dev \
  --few-shot-k 1 \
  --output-format canonical \
  --artifacts-root "./runs/neoolaf_docred_artifacts_docred_constrained" \
  --chunk-size 10000000 \
  --chunk-overlap 0 \
  --max-chunks 1 \
  --max-expressions 20 \
  --max-relation-mentions 20 \
  --max-workers 1 \
  --document-workers 4 \
  --no-checkpoints \
  --no-chunk-checkpoints \
  --no-resume
```

The output JSONL order remains stable and follows the selected dataset order.

