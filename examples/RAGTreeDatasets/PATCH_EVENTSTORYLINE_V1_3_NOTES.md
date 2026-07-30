# EventStoryLine v1.3 patch notes

- Adds parallel sentence-level high-recall event extraction.
- Adds two whole-document coverage-review passes.
- Broadens explicit event coverage to verbs, nominals, processes, and states without using gold.
- Preserves repeated mention identity and exact source-token spans.
- Replaces one global relation-generation call with bounded source batches for complete source coverage.
- Keeps OWL-Time, the two controlled relations, native Layers 0--12, and post-run-only evaluation unchanged.
- Uses a fresh v1.3 run directory.
- Does not modify `src/neoolaf`.
