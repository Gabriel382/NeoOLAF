# EventStoryLine v1.2 patch notes

- Adds two-phase Layer 1 extraction: exhaustive structured event inventory, then
  closed-inventory directed pair generation.
- Validates all event spans against source tokens.
- Repairs complete trigger text, common phrasal verbs, and tokenized hyphenated
  triggers using source tokens only.
- Rejects connective-only and auxiliary-only event candidates.
- Keeps OWL-Time as the external seed ontology and keeps the controlled task
  relations `PRECONDITION` and `FALLING_ACTION`.
- Uses a new run directory so v1.1 artifacts are not reused.
- Fixes no code under `src/neoolaf`.
