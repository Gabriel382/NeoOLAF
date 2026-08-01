# EventStoryLine native NeoOLAF one-document ablation v1.4

Run the notebook from inside the NeoOLAF repository after extracting the patch at the project root.

The experiment keeps all native NeoOLAF Layers 0--12 while implementing EventStoryLine-specific orchestration only under `examples/RAGTreeDatasets`.

## Inputs and ontology

- Pipeline input: `data/eventstoryline_one_input_v1.jsonl`
- Post-run gold: `data/eventstoryline_one_gold_v1.jsonl`
- Seed ontology: sibling RAGTree `data/ontology/OWLTime/time.ttl`
- Controlled task schema: `PRECONDITION`, `FALLING_ACTION`
- `null` relations are excluded.

## v1.4 sequence

1. Parallel sentence-level event mention extraction.
2. Three whole-document missing-event reviews.
3. Exact source-token validation and bounded span repair.
4. Deterministic unordered relation pair construction over the closed event inventory.
5. Batched five-way Layer-2 classification with both directions considered.
6. Evidence validation, one recovery pass, batch response caching and `NONE` filtering.
7. Native Layers 3--12 over accepted event-to-event relations only.
8. Exact-span-only projected evaluation plus strict native span evaluation.

## Important logs

- `run_logs/layer01_event_inventory.json`
- `run_logs/layer01_pair_pool.json`
- `run_logs/layer02_relation_decisions.json`
- `run_logs/layer02_compact_prompt_audit.json`
- `run_logs/layer02_batch_audit.json`
- `run_logs/layer02_closure_pair_pool.json`
- `analysis/native_span_relation_evaluation.json`
- `analysis/native_span_event_evaluation.json`
- `analysis/candidate_pool_evaluation.json`
- `analysis/relation_confusion_matrix.csv`
- `analysis/gold_relation_trace.csv`

Gold entities and relations are never loaded by the pipeline and are used only by the notebook analysis after Layer 12.
