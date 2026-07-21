# NeoOLAF Studio read-only integration contract

## Boundary

NeoOLAF remains the scientific L0-L12 execution engine. NeoOLAF Studio is a separate application that may invoke the pinned CLI and read generated artifacts, but it must not modify layers, prompts, profiles, serializers, or original run outputs.

## Supported read operations

- inspect current and historical run layouts;
- locate per-layer `state.json`, `output.json`, and `metadata.json` artifacts;
- locate document-named legacy layer payloads;
- locate the six canonical ontology/KG exports;
- calculate SHA-256 integrity hashes;
- load the latest complete JSON `PipelineState`;
- optionally load an explicitly trusted local checkpoint.

## Original outputs

The canonical outputs are:

- `ontology_local.ttl`
- `ontology_inferred.ttl`
- `kg_local.ttl`
- `kg_inferred.ttl`
- `kg_local.json`
- `kg_inferred.json`

Studio-derived provenance, query indexes, Cytoscape data, review exports, and Neo4j projections must be stored outside the scientific layer directories.

## Security

Gzipped pickle checkpoints are not safe inputs from untrusted users. `load_final_state()` only loads them after `allow_trusted_checkpoint=True`, and only if the selected file is inside `<run>/checkpoints`.
