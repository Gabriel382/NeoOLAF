# Unified4 v1.3 patch

This is an experiment-side patch only. It does **not** modify `src/neoolaf`.

## Changes

- FinCausal: high-recall full fact/proposition endpoint inventory, deterministic source-boundary union, semantic CAUSE direction, role-hint assisted pair classification.
- MAVEN-ERE: sentence-batched exhaustive event mention extraction with one coverage review; singleton mention endpoints for recall; direct five-way CAUSE/PRECONDITION relation classification; local distance-4 candidate pool plus long-range proposal pass.
- CausalBank: deterministic all ordered non-self lexical pairs under the visible BECAUSE/THEREFORE record family; no pair-level relation LLM pruning.
- State guard: the notebook checks actual saved one-document metrics before READY_5. Old completion booleans alone are not accepted.
- CausalBank: zero-cost source-only dense projection preview is evaluated post-hoc before any new paid call.
- Existing compact NeoOLAF-compatible EventKG and WordNet schema views are retained.
- `RUN_PAID=False` by default.

## Expected paid one-doc work

After the zero-cost preflight, already-passed EventStoryLine and CausalBank should be skipped. FinCausal and MAVEN-ERE run only if no prior metric-bearing one-document evaluation passes the v1.3 sanity gates.

## Smoke budget

Only one paid smoke-5 is allowed per dataset. The fixed five record keys are persisted before execution and successful partial records are not repeated on resume.
