# Run the v4 one-document DocRED ablation

1. Open `DocRED_NeoOLAF_Native_Layer_Ablation_SkaiTV_v4.ipynb` from the repository.
2. Activate the normal NeoOLAF environment.
3. Set `OPENROUTER_API_KEY` in the environment or enter it through the hidden notebook prompt.
4. Run all cells.

The output directory is:

`examples/RAGTreeDatasets/runs/docred_native_layer_ablation/skai_tv_structured_contrastive_v4`

The notebook uses the full supplied ontology, one whole-document chunk, 12 workers for
relation-only Layer 2 calls, and 8 workers for the Layer 4 fallback path.
