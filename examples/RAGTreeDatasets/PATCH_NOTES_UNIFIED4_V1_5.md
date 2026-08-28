# Unified4 v1.5 patch

This patch changes **MAVEN-ERE relation reasoning only**.

The executed v1.4 development document reached all 21/21 gold event clusters and all 12/12 relation endpoint clusters in the Layer-1 posthoc diagnostic, while relation extraction remained 1 TP / 26 FP / 13 FN (F1 0.04878). v1.5 therefore freezes event extraction and replaces relation selection.

- Reuse the exact validated v1.4 Layer-1 mention audit when available, validating every span against sanitized source tokens; otherwise fall back to unchanged v1.4 Layer 1.
- Conservative relation-time coreference only; Layer-1 inventory is not mutated.
- No all-local-pair enumeration. Two sparse whole-document graph proposals plus source-only causal-cue fallback.
- Strict definition verifier defaults to REJECT.
- Independent false-positive skeptic.
- Ambiguous opposite directions are rejected deterministically.
- EventStoryLine, FinCausal and CausalBank stay frozen.
- No `src/neoolaf` modifications.

Notebook defaults: `RUN_PAID=False`, `RUN_MODE="one_doc"`, `RUN_DATASETS=["maven_ere"]`. If the v1.4 audit exists, Layer 1 makes zero new LLM calls.
