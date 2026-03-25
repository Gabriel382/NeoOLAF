from __future__ import annotations

# Standard library imports
from typing import List

# Local imports
from neoolaf.core.pipeline_state import PipelineState


class StateMerger:
    """
    Merge chunk-level PipelineState outputs into one document-level state.

    This first implementation merges the most important intermediate artifacts:
    - linguistic expressions
    - enriched expressions
    - typed candidates
    - relation assertions
    - candidate triples

    Later, more sophisticated re-resolution can be added.
    """

    def merge_chunk_states(
        self,
        base_state: PipelineState,
        chunk_states: List[PipelineState],
    ) -> PipelineState:
        """
        Merge several chunk-level states into a document-level state.

        Args:
            base_state:
                The base full-document state, usually the result of preprocessing.
            chunk_states:
                States produced by running chunk-level layers.

        Returns:
            One merged document-level PipelineState.
        """
        # ---------------------------------------------------------
        # Merge Layer 1 outputs
        # ---------------------------------------------------------
        base_state.linguistic_expressions = []
        for state in chunk_states:
            base_state.linguistic_expressions.extend(getattr(state, "linguistic_expressions", []))

        # ---------------------------------------------------------
        # Merge Layer 2 outputs
        # ---------------------------------------------------------
        base_state.enriched_expressions = []
        for state in chunk_states:
            base_state.enriched_expressions.extend(getattr(state, "enriched_expressions", []))

        # ---------------------------------------------------------
        # Merge Layer 3 outputs
        # ---------------------------------------------------------
        base_state.entity_candidates = []
        base_state.relation_candidates = []
        base_state.attribute_candidates = []
        base_state.event_candidates = []

        for state in chunk_states:
            base_state.entity_candidates.extend(getattr(state, "entity_candidates", []))
            base_state.relation_candidates.extend(getattr(state, "relation_candidates", []))
            base_state.attribute_candidates.extend(getattr(state, "attribute_candidates", []))
            base_state.event_candidates.extend(getattr(state, "event_candidates", []))

        # ---------------------------------------------------------
        # Merge Layer 4 outputs
        # ---------------------------------------------------------
        base_state.candidate_relation_assertions = []
        for state in chunk_states:
            base_state.candidate_relation_assertions.extend(
                getattr(state, "candidate_relation_assertions", [])
            )

        # ---------------------------------------------------------
        # Merge Layer 5 outputs
        # ---------------------------------------------------------
        base_state.candidate_triples = []
        for state in chunk_states:
            base_state.candidate_triples.extend(getattr(state, "candidate_triples", []))

        # ---------------------------------------------------------
        # Deduplicate merged outputs
        # ---------------------------------------------------------
        base_state.linguistic_expressions = self._dedup_by_id(
            base_state.linguistic_expressions,
            id_attr="expr_id",
        )

        base_state.enriched_expressions = self._dedup_enriched_expressions(
            base_state.enriched_expressions
        )

        base_state.entity_candidates = self._dedup_by_id(
            base_state.entity_candidates,
            id_attr="candidate_id",
        )
        base_state.relation_candidates = self._dedup_by_id(
            base_state.relation_candidates,
            id_attr="candidate_id",
        )
        base_state.attribute_candidates = self._dedup_by_id(
            base_state.attribute_candidates,
            id_attr="candidate_id",
        )
        base_state.event_candidates = self._dedup_by_id(
            base_state.event_candidates,
            id_attr="candidate_id",
        )

        base_state.candidate_relation_assertions = self._dedup_by_id(
            base_state.candidate_relation_assertions,
            id_attr="assertion_id",
        )

        base_state.candidate_triples = self._dedup_by_id(
            base_state.candidate_triples,
            id_attr="triple_id",
        )

        return base_state

    def _dedup_by_id(self, items: list, id_attr: str) -> list:
        """
        Deduplicate objects by one identifier attribute while preserving order.
        """
        seen = set()
        merged = []

        for item in items:
            item_id = getattr(item, id_attr, None)
            if item_id is None:
                merged.append(item)
                continue

            if item_id in seen:
                continue

            seen.add(item_id)
            merged.append(item)

        return merged

    def _dedup_enriched_expressions(self, items: list) -> list:
        """
        Deduplicate enriched expressions using the base expression identifier.
        """
        seen = set()
        merged = []

        for item in items:
            expr_id = getattr(item.base_expression, "expr_id", None)
            if expr_id is None:
                merged.append(item)
                continue

            if expr_id in seen:
                continue

            seen.add(expr_id)
            merged.append(item)

        return merged