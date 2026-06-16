# NeoOLAF prompts

This folder is reserved for external prompt templates and prompt-review material.

For the ablation workflow, each executed layer now writes the actual prompts it sent to the model under:

```text
<run_dir>/<layer_name>/prompts/
  prompt_001_system.txt
  prompt_001_user.txt
  prompt_001_full.txt
  ...
<run_dir>/<layer_name>/prompt_stats.json
```

These files are the safest source for prompt review because they include the real document/chunk context, ontology context, RAG context, and previous artifacts used during the run.
