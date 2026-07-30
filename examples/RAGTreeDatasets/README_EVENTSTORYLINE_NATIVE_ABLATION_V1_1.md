# EventStoryLine native NeoOLAF one-document experiment v1.1

Run the notebook from inside the NeoOLAF repository. The expected repository
layout is:

```text
postdoc/
├── NeoOLAF/
└── ragtree/
    └── data/ontology/OWLTime/time.ttl
```

The notebook automatically searches the lowercase and uppercase sibling paths.
For a different layout, set:

```bash
export EVENTSTORYLINE_ONTOLOGY_PATH=/absolute/path/to/OWLTime/time.ttl
```

The external seed ontology is OWL-Time. The controlled benchmark relation schema
contains exactly `PRECONDITION` and `FALLING_ACTION`; `null` pairs are excluded.
Gold events and relation pairs are loaded only after Layer 12.
