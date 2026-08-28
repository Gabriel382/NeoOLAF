# NeoOLAF × RAGTree — Unified 4-Dataset Patch v1.1

This is an **overlay patch** for the NeoOLAF repository. Extract it at the NeoOLAF repository root (for example `C:\Users\galencarmedeiro\NeoOLAF`). It adds only files under `examples/RAGTreeDatasets`; it does **not** modify `src/neoolaf`.

## Notebook

Open:

`examples/RAGTreeDatasets/RAGTreeDatasets_NeoOLAF_Unified_4Datasets_v1_1.ipynb`

The notebook auto-detects these RAGTree locations, in this order:

- `RAGTREE_ROOT` environment variable;
- `C:\Users\galencarmedeiro\RAGTree`;
- `C:\Users\galencarmedeiro\Documents\git\postdoc\RAGTree`;
- a sibling `RAGTree` directory next to NeoOLAF.

It expects the ontology root at `RAGTree\data\ontology` (or `RAGTREE_ONTOLOGY_DIR`) and validates exactly:

- EventStoryLine: `OWLTime\time.ttl`
- FinCausal: `FIBO-CorePlus\fibo-core-plus.ttl`
- MAVEN-ERE: `EventKG\EventKGSchema.ttl`
- CausalBank: `WordNet-Full\wordnet.ttl`

It expects normalized JSONLs under `RAGTree\data\preprocessed` or `RAGTree\preprocessed`:

- `eventstoryline.jsonl`
- `fincausal.jsonl`
- `maven_ere.jsonl`
- `causalbank.jsonl`

## EventKG ontology compatibility fix

The original 644-byte EventKG schema is valid RDF/Turtle, but it only uses `rdfs:subClassOf`, `rdfs:domain`, and `rdfs:range`; it does not explicitly type `eventKG-s:Relation` as an OWL/RDFS class or its predicates as OWL properties. NeoOLAF's `SeedOntologyLoader` therefore sees **0 classes / 0 properties** and aborts before Layer 0. Patch v1.1 adds an offline preflight for this exact failure. Replace the external RAGTree file with the provided corrected ontology:

`C:\Users\galencarmedeiro\RAGTree\data\ontology\EventKG\EventKGSchema.ttl`

The corrected file preserves the original EventKG triples and adds only explicit OWL typing/labels needed by NeoOLAF's loader.

## Paid-run guard

The notebook starts with `RUN_PAID = False`. Nothing paid runs until this is explicitly changed.

Supported modes:

- `one_doc`: one development sanity document;
- `smoke5`: the single protected five-document smoke run;
- `full`: full benchmark, permitted by default only after the dataset is locked/frozen.

Persistent state is stored in:

`examples/RAGTreeDatasets/state/development_manifest_v1.json`

The patch ships only a template (`development_manifest_TEMPLATE_v1.json`). The live manifest is created locally on first notebook run and is not overwritten by later patch extraction.

For `smoke5`, the selected five record keys are persisted before the first paid call. If execution is interrupted, already-completed smoke records are skipped on resume; the notebook does not repeat successful smoke records. Once all five complete, `smoke5_already_run=true` blocks accidental reruns unless `FORCE_RUN[dataset]=True` is deliberately set.

## Gold isolation

Before any NeoOLAF execution, the notebook removes:

- `entities`
- `relations`
- `pred_relations`
- `ontology_links`
- other recognized gold fields

Only the sanitized record is passed to the Layer 0–12 pipeline. The post-hoc gold JSONL is deliberately created **after Layer 12 returns**.

## Scientific adapters

### EventStoryLine

Reuses the previously executed v1.7 adapter and its native Layer 0–12 stack. The fixed smoke IDs remain `1_10ecbplus` through `1_14ecbplus`.

### FinCausal

Layer 1 extracts proposition/fact spans rather than atomic event triggers. Layer 2 classifies semantic direction as `A_CAUSES_B`, `B_CAUSES_A`, or `NONE`, then emits controlled `CAUSE` relations. Text order is not used as causal direction. FIBO CorePlus remains the external semantic seed.

### MAVEN-ERE

Layer 1 extracts exact event mentions and groups true coreferent mentions into event-cluster endpoints. Layer 2 constructs a high-recall local pair pool plus a bounded long-range proposal, then performs:

1. `LINK` vs `NONE`;
2. `CAUSE` vs `PRECONDITION` plus semantic direction for linked pairs.

No gold temporal relation is used and no automatic causal transitive closure is added. EventKG remains the external semantic seed.

### CausalBank

Layer 1 derives lexical endpoint candidates only from source tokens. It uses a no-gold union of lexical forms/stems (NLTK Lancaster/Porter/WordNet forms when available, with a small dependency-free fallback) because the normalized benchmark endpoint inventory is stem/lemma-like. The full normalized benchmark has **two controlled relation families: `BECAUSE` and `THEREFORE`**. Layer 2 first chooses the record-level family from the visible CausalBank `type`/text only (never from gold relations), then classifies unordered lexical pairs as `BOTH`, `A_TO_B`, `B_TO_A`, or `NONE` inside that fixed family. This preserves the dense normalized lexical graph without pretending it is one sparse proposition-level edge. Post-Layer-12 projection uses the deterministic `EVENT_ + first16(md5(label))` identity rule. WordNet Full remains the external seed.

## API / concurrency

Set the key in the environment, not in the notebook:

PowerShell:

```powershell
$env:OPENROUTER_API_KEY="..."
```

Default execution is sequential at the document/dataset level. If the provider returns 429s, reduce document-level concurrency first; the notebook already defaults to `DOCUMENT_WORKERS = 1` and `LAYER_WORKERS = 4`.

## Artifacts

Each record stores its sanitized input, full Layer states/checkpoints, prompts/responses, provider errors, ontology retrieval logs, run manifest, console log and post-Layer-12 evaluation under:

`examples/RAGTreeDatasets/runs/unified4_v1_1/`

The shared response cache defaults to the OS temporary directory and can be moved with `NEOOLAF_RAGTREE_CACHE_DIR`.
