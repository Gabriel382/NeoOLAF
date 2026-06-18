# NeoOLAF RAGTree datasets benchmark files

Copy this folder tree at the root of the NeoOLAF repository.

Main files:

```text
examples/RAGTreeDatasets/RAGTreeDatasets_NeoOLAF.ipynb
examples/RAGTreeDatasets/configs/guidance_docred.json
examples/RAGTreeDatasets/configs/guidance_causalbank.json
examples/RAGTreeDatasets/configs/guidance_eventstoryline.json
examples/RAGTreeDatasets/configs/guidance_fincausal.json
examples/RAGTreeDatasets/configs/guidance_maven_ere.json
experiments/methods/run_neoolaf.py
experiments/methods/README_neoolaf_document_parallel.md
```

The notebook has DocRED active by default. The other four datasets are present but commented.

The runner supports:

```bash
--document-workers N
```

This controls how many documents are processed in parallel. Start with `--document-workers 2` and increase only if the provider does not rate-limit.
