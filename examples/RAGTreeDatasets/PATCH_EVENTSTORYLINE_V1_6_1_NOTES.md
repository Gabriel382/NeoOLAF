# EventStoryLine v1.6.1 Windows dataset-path patch

This is the v1.6 source-centric relation experiment with dataset discovery updated for the current RAGTree layout.

Default preprocessed root:

```text
C:\Users\galencarmedeiro\Documents\git\postdoc\RAGTree\preprocessed
```

The notebook also searches the sibling `RAGTree/preprocessed` directory relative to the NeoOLAF repository and accepts an override through `RAGTREE_PREPROCESSED_DIR`.

Expected files:

- `causalbank.jsonl`
- `docred_causal.jsonl`
- `eventstoryline.jsonl`
- `fincausal.jsonl`
- `maven_ere.jsonl`

For the one-document EventStoryLine run, the notebook streams `eventstoryline.jsonl`, selects `EventStoryLine - 1_10ecbplus`, writes a stripped pipeline input without `entities`/`relations`, and keeps the complete record in a separate gold JSONL used only after Layer 12. The first five source records are prepared the same way for the later smoke batch.

No file under `src/neoolaf` is modified.
