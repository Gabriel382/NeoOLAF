# DocRED native ablation patch v3

This patch changes only `examples/RAGTreeDatasets`; `src/neoolaf` remains byte-for-byte unchanged.

## Scientific changes

- One Skai TV document only.
- Full native NeoOLAF Layer 0–12 run.
- Explicit input-level `task_guidance` metadata lists the target relation schema, direction, and synthetic examples. It contains no gold entities or gold relation pairs and is not prepended to the article text.
- Layer 2 is explicitly required to emit `controlled_relation:Pxxx : label` and `promote_to_ontology:true` hints.
- Relation schema candidates are promoted; named instances are not.
- Strict DocRED scoring remains unchanged.
- No direct DocRED extraction call, source anchoring, closure rules, or gold-derived relation subset.

## Performance changes

- Layer 1: one whole-document request.
- Layer 2: 12 workers, 768 output tokens/request, 90-second timeout, one retry.
- Layer 4: 8 workers through an experiment-side parallel orchestration subclass, 512 output tokens/request, 90-second timeout, at most two attempts.
- The Layer 4 subclass performs the same existing NeoOLAF prompt/task in parallel; it does not add another extraction task.
- Per-layer backends prevent a single enrichment response from consuming the global 8192-token budget.

## Diagnostics

- Raw responses, API/parse errors, response-cap violations, ontology retrievals, pipeline logs, and layer states are saved.
- Cumulative strict evaluation is written for every layer.
- The gold trace distinguishes schema availability from an actual predicate mention.
- Native lexical triples, ontology-canonical triples, strict predictions, and predictions absent from gold are saved separately.
