# EventStoryLine NeoOLAF v1.6.3 portable bundle

Extract this archive at the NeoOLAF repository root. It contains the full experiment-side dependency chain under `examples/RAGTreeDatasets/`; it does **not** modify `src/neoolaf`.

## Windows cache hotfix

Layer-2 source-centric relation responses are cached outside the deep run tree. Default:

- Windows: `%TEMP%\neoolaf_esl16`
- Linux/macOS: the platform temporary directory + `neoolaf_esl16`

Optional override: `NEOOLAF_EVENTSTORYLINE_CACHE_DIR`. Cache filenames use only 24 SHA-256 characters. Cache reads/writes are best-effort; filesystem/cache errors cannot invalidate a successfully parsed LLM response.

Open `examples/RAGTreeDatasets/EventStoryLine_NeoOLAF_Native_Layer_Ablation_OneDoc_v1_6_3.ipynb`, restart the kernel, and run from the top.

The notebook discovers `RAGTree/preprocessed/eventstoryline.jsonl` through `RAGTREE_PREPROCESSED_DIR`, sibling-repository paths, or the previously supplied Windows path. OWL-Time can be overridden with `EVENTSTORYLINE_ONTOLOGY_PATH`.
