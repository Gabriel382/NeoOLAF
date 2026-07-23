# DocRED native layer ablation patch v4

Main notebook:

`DocRED_NeoOLAF_Native_Layer_Ablation_SkaiTV_v4.ipynb`

## Scientific constraints

- No file under `src/neoolaf` is modified.
- The complete Layer 0–12 pipeline is executed.
- Gold entities and relations are not present in the pipeline input.
- No direct DocRED extraction fallback is added.
- No source-entity anchoring, relation closure, or post-hoc relation invention is used.
- The supplied 96-property DocRED ontology remains the relation inventory.

## Changes from v3

1. Layer 1 uses one whole-document call and emits exact endpoint expressions plus one
   `SOURCE || PREDICATE || TARGET` expression per relation instance.
2. Layer 2 enriches endpoint nodes deterministically and calls the LLM only for relation
   instances. Relation calls run in parallel and use a contrastive ontology-property prompt.
3. Layer 3 keeps the existing role-based typing path but receives an empty
   `relations.allowed` list, so it does not inject mention-free vocabulary candidates.
4. Layer 4 resolves exact structured endpoints deterministically. Its existing LLM endpoint
   task remains available as a parallel fallback only when exact resolution fails.
5. Coarse DocRED type constraints reject impossible assertions, including temporal
   properties whose target is a location.
6. The relation trace now attributes failures to Layer 1 relation-instance extraction,
   Layer 2 canonical property selection, Layer 3 resolution, Layer 4 endpoint/type
   validation, or Layer 5 materialization.

## Additional artifacts

- `run_logs/layer01_relation_instances.json`
- `run_logs/layer02_contrastive_decisions.json`
- `run_logs/layer04_endpoint_assignment.json`
- `run_logs/layer04_constraint_rejections.json`
- `analysis/gold_relation_trace_v4.csv`
- normal NeoOLAF layer states, outputs, metadata, prompts, checkpoints and exports

## Expected speed behavior

The number of Layer 2 calls is the number of relation instances, not the total number of
expressions. Exact Layer 4 endpoint resolution normally avoids Layer 4 LLM calls entirely.
