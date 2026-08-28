# Unified4 v1.4 — MAVEN-only development patch

This patch is intentionally narrow. It does **not** modify `src/neoolaf`, EventStoryLine, FinCausal, or CausalBank behavior.

## Why
The v1.3 MAVEN run improved final relation-endpoint recall from 0.083 to 0.75, but relation precision remained 0.037 (1 TP / 26 FP). The prior endpoint metric also mixed event extraction with relation filtering because it only counted endpoints that survived into final triples.

## v1.4 changes
- Layer 1A keeps sentence-batched exhaustive event extraction.
- Layer 1B adds a deterministic source-token lexical rescue pool, validated by the LLM. It explicitly targets eventive nouns/nominalizations, states/changes, starts/ends, decisions, outcomes, movements, communications, etc.
- Layer 1C performs one final whole-document missed-event review.
- Offset/trigger repair is source-token-only; no gold is available to the pipeline.
- Post-L12 evaluation now reports Layer-1 inventory recall separately from final relation-endpoint recall.
- Layer 2 uses three semantic steps: strict directed LINK/NONE gate -> independent positive verifier -> fixed-direction CAUSE/PRECONDITION/REJECT subtype.
- The v1.3 local-NONE promotion path is removed.
- Candidate generation uses same/adjacent-sentence pairs plus a bounded nonlocal proposal pass to preserve long-range MAVEN links without classifying every possible pair.

## Budget guard
The notebook defaults to `RUN_PAID=False`, `RUN_MODE="one_doc"`, `RUN_DATASETS=["maven_ere"]`. ESL, FinCausal, and CausalBank are not rerun.

## Validation performed before packaging
- Adapter imports successfully against the uploaded NeoOLAF snapshot.
- Native pipeline constructs all 13 layers with `MavenLayer1V14` and `MavenLayer2V14`.
- All Python/config/notebook cells compile/parse.
- The source-only rescue **candidate pool** reaches 100% of relation-endpoint clusters in each of the five supplied MAVEN development documents; this is a reachability ceiling, not a claim that the LLM will accept every candidate.
