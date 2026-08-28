# Unified 4-Dataset v1.2 patch

This is intentionally a small compatibility/resume patch.

## Fixes

1. **WordNet CausalBank crash**
   - The previous local `wordnet.ttl` can be a synset/instance graph with no TBox declarations visible to NeoOLAF's `SeedOntologyLoader`.
   - v1.2 bundles `ontology_compat/wordnet_neoolaf_seed.ttl`, generated from the W3C WordNet 2.0 Full `wnfull.rdfs` schema.
   - NeoOLAF sees 18 classes and 32 object/datatype properties.
   - The huge WordNet instance graph is not parsed by NeoOLAF.

2. **EventKG loader compatibility**
   - v1.2 bundles `ontology_compat/EventKGSchema_NeoOLAF.ttl`.
   - It preserves the EventKG schema triples and adds explicit OWL typing required by the current NeoOLAF loader.
   - NeoOLAF sees 1 class and 3 properties.

3. **Paid-run resume**
   - v1.2 scans existing `runs/unified4_v1_1/*/one_doc/*/posthoc_evaluation.json`.
   - A completed MAVEN/FinCausal/CausalBank one-doc is marked complete in the persistent manifest instead of being paid for twice.

4. **Failure persistence**
   - One dataset failure no longer prevents summary/failure JSON files from being written.

## Install

Extract the patch ZIP over the NeoOLAF repository root.

Open:
`examples/RAGTreeDatasets/RAGTreeDatasets_NeoOLAF_Unified_4Datasets_v1_2.ipynb`

Run once with `RUN_PAID = False`.
Only after `ALL FOUR ONTOLOGY SEEDS: OK` should `RUN_PAID` be changed to `True`.

The notebook keeps the original RAGTree ontology files untouched. The compact compatibility views are used only by NeoOLAF.
