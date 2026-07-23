# DocRED native ablation v5.1 hotfix

This hotfix fixes the preflight crash:

```text
TypeError: RelationExample.__init__() got an unexpected keyword argument
'candidate_relation_ids'
```

## Cause

The v5 input keeps `candidate_relation_ids` beside synthetic relation examples so
the compact DocRED prompt builder can audit candidate sets.  The generic
`RelationExample` dataclass does not define that experiment-only field.  The v3
input-guidance merger forwarded the complete dictionary to the dataclass.

## Fix

`merge_input_task_guidance` now constructs `RelationExample` explicitly from its
stable fields (`text`, `source_label`, `relation_label`, `target_label`, and
`explanation`) and leaves extra experiment metadata in the raw task-guidance
record where v5 can still use or audit it.

No file under `src/neoolaf` was changed.  No extraction, mapping, evaluation, or
scientific setting was altered.
