# DocRED native batch v6.1 patch

This patch fixes a post-pipeline evaluation failure:

`ValueError: dict contains fields not in fieldnames: 'candidate_entity_ids'`

Cause: only ambiguous entity projections contain `candidate_entity_ids`, while the
CSV writer derived its field list from the first row.

Changes:

1. CSV schemas are now the stable union of keys across every row.
2. Lists/dictionaries are JSON-serialized in CSV cells.
3. A completed Layer 0-12 run is detected from `run_manifest.json`.
4. Evaluation is rebuilt from saved artifacts without rerunning OpenRouter calls.
5. Failed progress lines now print error type, error text, and transient status.
6. `pipeline=None` is displayed as `pipeline=n/a`.

Use the same batch root:

`examples/RAGTreeDatasets/runs/docred_native_v5_1_batch`

Open `RAGTreeDatasets_DocRED_NativeNeoOLAF_Batch_v6_1_RunEval.ipynb` and rerun
the batch cell with `RUN_ALL_DOCUMENTS = False`. The two successful documents
will be resumed, and the three post-processing failures will be re-evaluated
from their existing pipeline artifacts. No LLM calls are needed for those three
recoveries.
