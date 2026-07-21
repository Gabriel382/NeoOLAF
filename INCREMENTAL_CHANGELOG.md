# Incremental change: read-only NeoOLAF Studio integration

This branch-ready increment implements the GitHub issue **“Add a read-only integration API for NeoOLAF Studio without modifying the scientific pipeline.”**

## Added

- package/scientific version identity metadata;
- immutable run snapshot models;
- current and legacy run discovery;
- canonical export discovery and SHA-256 hashing;
- safe-by-default final JSON state loading;
- explicit trusted-local checkpoint fallback;
- `neoolaf version` command;
- `neoolaf inspect-run` command;
- clean source-archive builder with secret checks;
- publication dependency snapshot;
- read-only, compatibility, and output-integrity tests.

## Unchanged scientific boundary

No file under `src/neoolaf/layers/`, no prompt, no document profile, no domain model, no pipeline runner, and no existing ontology/KG serializer was modified.
