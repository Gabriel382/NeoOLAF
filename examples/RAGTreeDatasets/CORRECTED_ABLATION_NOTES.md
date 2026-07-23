# Corrected Skai TV native ablation

The first generated notebook completed technically, but its strict DocRED score
was zero because the experiment exposed two configuration/integration problems:

1. Layer 3 used `generic_llm_typing`. The model replaced proper names with generic
   canonical labels. NeoOLAF then grouped candidates by normalized canonical
   label, merging `Skai TV` and `Skai Group` into one candidate called `entity`,
   turning `Piraeus` into a relation-like candidate, and losing endpoint identity.
2. The notebook adapter delegated to `OntologySpace.retrieve`, which concatenates
   classes before properties and truncates the result. The observed Layer 4 RAG
   logs therefore contained classes but no DocRED property descriptions.

This corrected package changes only files under `examples/RAGTreeDatasets`:

- the profile selects NeoOLAF's existing
  `ontology_aware_role_based_typing` strategy;
- profile node-role mappings preserve each source surface form;
- the ontology adapter performs balanced class/property retrieval and logs the
  returned property IDs;
- generic query expansions are profile-controlled and do not contain Skai TV
  facts or gold relations;
- the diagnostic trace now distinguishes Layer 1 extraction loss, Layer 2
  survival loss, Layer 3 resolution loss, Layer 4 endpoint assignment, and Layer
  5 materialization/mapping;
- the corrected run uses `skai_tv_balanced_v2`, so old artifacts are not reused.

No file under `src/neoolaf` was modified. No direct DocRED extractor, source
entity anchoring, gold relation hint, or closure rule was added.

A strict DocRED score still does not measure every valid native relation. For
example, the text explicitly says that Skai TV is a member of Digea, but that
relation is absent from this document's gold relation set. The notebook keeps
native and mapped outputs separate so this distinction remains visible.
