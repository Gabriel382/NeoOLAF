# NeoOLAF native DocRED ablation v5

Open:

```text
examples/RAGTreeDatasets/DocRED_NeoOLAF_Native_Layer_Ablation_SkaiTV_v5.ipynb
```

Set `OPENROUTER_API_KEY` in the environment and run the notebook from inside the
NeoOLAF repository. It processes only the Skai TV document and saves a complete
Layer 0–12 run under:

```text
examples/RAGTreeDatasets/runs/docred_native_layer_ablation/skai_tv_country_compact_v5
```

The pipeline input contains no gold entity clusters or gold relation pairs.
Gold is loaded only after execution for strict DocRED evaluation and audited
mention projection.
