# EventStoryLine native NeoOLAF one-document ablation v1.3

This experiment runs native NeoOLAF Layers 0--12 on one EventStoryLine document with the same OWL-Time seed ontology used by RAGTree.

## v1.3 Layer 1

1. Parallel sentence-level event inventories.
2. Two whole-document missing-event reviews, emphasizing eventive nominals, explicit states, embedded predicates, and repeated mentions.
3. Deterministic validation against source tokens with bounded phrasal/hyphen span repair.
4. Parallel source-batched PLOT_LINK proposal generation over a closed event inventory.
5. Parallel Layer 2 mapping to PRECONDITION, FALLING_ACTION, or found=false.

Gold entities and relations are loaded only after Layer 12. The profile is intended to be frozen after this one-document development run, then reused unchanged for the five-document smoke and all 443 records (`full` + `test`).
