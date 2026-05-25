# NeoOLAF Evaluation Runner

Main entry point:

```bash
python scripts/evaluation/run_eval.py --help
```

Supported commands:

```bash
python scripts/evaluation/run_eval.py evaluate
python scripts/evaluation/run_eval.py evaluate-jsonl
python scripts/evaluation/run_eval.py compare
python scripts/evaluation/run_eval.py batch-evaluate
```

Example for XQuality + NeoOLAF:

```bash
python scripts/evaluation/run_eval.py evaluate \
  --dataset xquality \
  --method neoolaf \
  --profile xquality_loose \
  --input runs/run_20260408_091832/data/exports \
  --gold data/XQuality/Examples/XQuality_all_triplets_flat_en.json \
  --ontology-path data/ontology/ContextOntology-COInd4.owl \
  --output outputs/evaluation/xquality/neoolaf/xquality_loose
```
