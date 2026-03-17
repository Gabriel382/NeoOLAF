from __future__ import annotations

# Standard library imports
from typing import Dict, List, Tuple
# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.candidate_triple import CandidateTriple


class CandidateTripleGenerationLayer(BaseLayer):
    """
    Layer 5: candidate triple generation.

    Responsibilities:
    - transform candidate relation assertions into candidate triples
    - preserve provenance and confidence
    - deduplicate graph assertions
    """

    name = "layer05_candidate_triple_generation"

    def __init__(
        self,
        max_assertions: int | None = None,
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize Layer 5.

        Args:
            max_assertions:
                Optional debug limit on the number of relation assertions processed.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Wheter to show logs or not.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.max_assertions = max_assertions

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Convert Layer 4 candidate relation assertions into candidate triples.
        """
        assertions = state.candidate_relation_assertions

        # Optional debug limit
        if self.max_assertions is not None:
            assertions = assertions[: self.max_assertions]

        triples: List[CandidateTriple] = []
        triple_counter = 0

        assertion_iterator = assertions
        if self.verbose:
            assertion_iterator = tqdm(assertions, desc="Layer 5 - assertions", leave=False)

        for assertion in assertion_iterator:
            triples.append(
                CandidateTriple(
                    triple_id=f"triple_{triple_counter:05d}",
                    subject_id=assertion.source_candidate_id,
                    subject_label=assertion.source_candidate_label,
                    subject_type=assertion.source_candidate_type,
                    predicate_id=assertion.relation_candidate_id,
                    predicate_label=assertion.relation_label,
                    object_id=assertion.target_candidate_id,
                    object_label=assertion.target_candidate_label,
                    object_type=assertion.target_candidate_type,
                    chunk_id=assertion.chunk_id,
                    justification=assertion.justification,
                    confidence=assertion.confidence,
                    provenance=assertion.evidence,
                )
            )
            triple_counter += 1

        # Deduplicate by semantic triple identity inside the same chunk
        dedup = {}
        for triple in triples:
            key = (
                triple.subject_id,
                triple.predicate_id,
                triple.object_id,
                triple.chunk_id,
            )
            if key not in dedup:
                dedup[key] = triple

        state.candidate_triples = list(dedup.values())
        state.log(
            f"[layer05_candidate_triple_generation] generated "
            f"{len(state.candidate_triples)} candidate triples"
        )
        return state

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize candidate triples for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "num_candidate_triples": len(state.candidate_triples),
            "candidate_triples": [
                {
                    "triple_id": triple.triple_id,
                    "subject": {
                        "id": triple.subject_id,
                        "label": triple.subject_label,
                        "type": triple.subject_type,
                    },
                    "predicate": {
                        "id": triple.predicate_id,
                        "label": triple.predicate_label,
                    },
                    "object": {
                        "id": triple.object_id,
                        "label": triple.object_label,
                        "type": triple.object_type,
                    },
                    "chunk_id": triple.chunk_id,
                    "justification": triple.justification,
                    "confidence": triple.confidence,
                    "provenance": [
                        {
                            "chunk_id": ev.chunk_id,
                            "chunk_start_char": ev.chunk_start_char,
                            "chunk_end_char": ev.chunk_end_char,
                            "doc_start_char": ev.doc_start_char,
                            "doc_end_char": ev.doc_end_char,
                            "snippet": ev.snippet,
                        }
                        for ev in triple.provenance
                    ],
                }
                for triple in state.candidate_triples
            ],
        }