# Unified4 v1.3.1 — FinCausal one-doc validity hotfix

This is a **small overlay on v1.3**. It does not modify `src/neoolaf` and does not change the FinCausal/MAVEN/CausalBank extraction adapters.

## Problem fixed

The v1.3 one-document FinCausal run selected the first normalized row (`FinCausal - bfdb8bd54bcc5044`). Its evaluation had zero gold endpoints and zero gold CAUSE relations, so the paid run could not measure extraction quality.

The full normalized FinCausal dataset nevertheless has 929 CAUSE relations across 967 records. v1.3.1 therefore makes the development-sanity selection criterion explicit and deterministic.

## New one-doc FinCausal selection contract

- choose the **first source-order record with at least one scored `CAUSE` relation**;
- freeze its stable `record_key` in the persistent manifest as `one_doc_record_key`;
- print the selected `document_id`, title, line index, gold entity count and gold target relation count before any paid/API call;
- refuse a paid FinCausal one-doc run if the selected record has zero target relations or fewer than two gold endpoints;
- after Layer 12, verify that the evaluator still sees non-zero gold whenever the controller did.

This criterion is only for the one-document *development sanity* gate. It is not score-based selection and does not alter the later smoke-5 selection.

## Gold isolation remains unchanged

The selected record is converted with `strip_gold(...)` before NeoOLAF runs. `entities`, `relations`, `pred_relations`, `ontology_links`, and other gold keys are absent from the pipeline-visible JSONL. The pre-run gold count is controller-only and is never passed into NeoOLAF.

## Safe default

The v1.3.1 notebook starts with:

```python
RUN_PAID = False
RUN_MODE = "one_doc"
RUN_DATASETS = ["fincausal"]
```

This prevents accidentally paying for another MAVEN run while its relation-layer precision fix is still pending. EventStoryLine and CausalBank remain READY_5 from their accepted configurations.
