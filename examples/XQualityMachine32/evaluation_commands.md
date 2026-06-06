# XQuality Machine 32 evaluation commands

After running Layer 12, evaluate the exported KG and ontology against the gold triples:

```bash
python -m neoolaf.evaluation evaluate \
  --dataset xquality \
  --method neoolaf \
  --profile xquality_relaxed_recall \
  --input examples/XQualityMachine32/runs/xquality_machine32/layer12_from_l11/exports \
  --gold data/XQuality/Examples/XQuality_all_triplets_flat_en.json \
  --ontology-path data/ontology/ContextOntology-COInd4.owl \
  --output examples/XQualityMachine32/runs/xquality_machine32/eval_layer12
```

The same evaluation can be launched with explicit files:

```bash
python -m neoolaf.evaluation evaluate \
  --dataset xquality \
  --method neoolaf \
  --profile xquality_relaxed_recall \
  --gold data/XQuality/Examples/XQuality_all_triplets_flat_en.json \
  --kg-local-json examples/XQualityMachine32/runs/xquality_machine32/layer12_from_l11/exports/kg_local.json \
  --kg-inferred-json examples/XQualityMachine32/runs/xquality_machine32/layer12_from_l11/exports/kg_inferred.json \
  --ontology-local-ttl examples/XQualityMachine32/runs/xquality_machine32/layer12_from_l11/exports/ontology_local.ttl \
  --ontology-inferred-ttl examples/XQualityMachine32/runs/xquality_machine32/layer12_from_l11/exports/ontology_inferred.ttl \
  --output examples/XQualityMachine32/runs/xquality_machine32/eval_layer12
```

Important output files:

- `metrics.summary.json`: global entity, relation, ontology and validation-oriented metrics.
- `metrics.flat.csv`: compact one-line metrics table.
- `per_relation_metrics.csv`: precision, recall and F1 by relation type.
- `matched_relations.json`: matched predicted/gold relations.
- `unmatched_relations.json`: false positives and false negatives.
- `ontology_metrics.json`: ontology quality/conformance indicators.

To compare multiple evaluations stored in one folder:

```bash
python -m neoolaf.evaluation compare \
  --runs-dir examples/XQualityMachine32/runs/xquality_machine32 \
  --output examples/XQualityMachine32/runs/xquality_machine32/eval_comparison
```
