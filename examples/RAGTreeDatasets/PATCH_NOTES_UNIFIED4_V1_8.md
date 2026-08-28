# Unified4 v1.8 Patch Notes

MAVEN-ERE relation-only correction.

- Frozen v1.4 Layer 1 is unchanged.
- Generate every ordered A->B and B->A mention pair whose minimum sentence distance is <=4.
- No textual-forward direction filter.
- No pre-classification candidate cap.
- Neutral subtype hint before LLM adjudication.
- The adjudicator chooses KEEP_PRECONDITION / KEEP_CAUSE / REJECT for each explicit direction.
- Gold is unavailable to the pipeline and remains post-L12 evaluation only.
