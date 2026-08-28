# Unified 4-Dataset patch v1.1

This overlay corrects two issues found by the zero-cost preflight.

1. **CausalBank full relation vocabulary**: the adapter now supports both `BECAUSE` and `THEREFORE` (the five inspected `resulted_from` examples only exposed `BECAUSE`). The record-level family is selected from the pipeline-visible CausalBank `type`/text only; gold remains unavailable until post-Layer-12 evaluation.
2. **EventKG seed loading**: the original 644-byte EventKG schema is valid Turtle but contains no explicit OWL/RDFS class/property typing, so NeoOLAF's `SeedOntologyLoader` sees zero classes/properties. Use the separately supplied `EventKGSchema_FIXED.ttl` as the external `RAGTree/data/ontology/EventKG/EventKGSchema.ttl`. The notebook now checks this before any paid call.

The patch changes no file under `src/neoolaf` and keeps `RUN_PAID = False` by default.
