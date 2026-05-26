from __future__ import annotations

# Standard library imports
import re
from typing import Dict, List, Tuple, Any
# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.candidate_triple import CandidateTriple
from neoolaf.domain.linguistic_expression import Evidence


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
        strategy = (state.profile_config or {}).get("layers", {}).get(
            self.name, {}
        ).get("strategy", "generic")
        if strategy == "alarm_record_to_triples" and getattr(state.document, "alarm_records", None):
            return self._run_alarm_record_to_triples(state)

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
                    metadata={},
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


    def _run_alarm_record_to_triples(self, state: PipelineState) -> PipelineState:
        """Generate triples from structured alarm records using profile mappings.

        This keeps the Machine32 table rules configurable: relation labels,
        triplet types, and defaults come from the document profile.
        """
        profile = state.profile_config or {}
        defaults = profile.get("defaults", {}) if isinstance(profile.get("defaults"), dict) else {}
        category = defaults.get("category", "PLC Alarm")
        triplet_type_by_relation = profile.get("triplet_type_by_relation", {}) or {}

        triples: list[CandidateTriple] = []
        counter = 0

        for record in getattr(state.document, "alarm_records", []) or []:
            alarm_label = str(record.get("alarm_label_en") or record.get("alarm_label_fr") or "").strip()
            alarm_no = str(record.get("alarm_no") or "").strip()
            chunk_id = str(record.get("chunk_id") or f"alarm_{alarm_no or 'unknown'}")
            page = record.get("page")
            if not alarm_label:
                continue

            def add(head: str, relation: str, tail: str, object_type: str, field: str, item: Any) -> None:
                nonlocal counter
                head = str(head or "").strip()
                tail = str(tail or "").strip()
                if not head or not tail:
                    return
                evidence_text = ""
                if isinstance(item, dict):
                    evidence_text = str(item.get("text_fr") or item.get("text_en") or tail)
                else:
                    evidence_text = str(item or tail)
                triples.append(
                    CandidateTriple(
                        triple_id=f"triple_{counter:05d}",
                        subject_id=self._stable_id(head),
                        subject_label=head,
                        subject_type="cause" if relation == "TRIGGERS" else "alarm",
                        predicate_id=relation.lower(),
                        predicate_label=relation,
                        object_id=self._stable_id(tail),
                        object_label=tail,
                        object_type=object_type,
                        chunk_id=chunk_id,
                        justification=f"Generated from structured alarm record field '{field}'.",
                        confidence=1.0,
                        provenance=[
                            Evidence(
                                chunk_id=chunk_id,
                                chunk_start_char=-1,
                                chunk_end_char=-1,
                                doc_start_char=-1,
                                doc_end_char=-1,
                                snippet=evidence_text,
                            )
                        ],
                        metadata={
                            "alarm_no": alarm_no,
                            "category": category,
                            "triplet_type": triplet_type_by_relation.get(relation),
                            "field": field,
                            "page": page,
                            "source_record": record,
                        },
                    )
                )
                counter += 1

            for item in record.get("cause_items", []) or []:
                add(self._item_text(item), "TRIGGERS", alarm_label, "alarm", "cause", item)
            for item in record.get("effect_items", []) or []:
                add(alarm_label, "CAUSES", self._item_text(item), "effect", "effect", item)
            for item in record.get("intervention_items", []) or []:
                add(alarm_label, "REQUIRES", self._item_text(item), "intervention", "intervention", item)
            for item in record.get("responsible_items", []) or []:
                add(alarm_label, "HANDLED_BY", self._item_text(item), "responsible", "responsible", item)
            for item in record.get("reference_items", []) or []:
                add(alarm_label, "REFERENCES", self._item_text(item), "reference", "reference", item)

        dedup: dict[tuple[str, str, str, str], CandidateTriple] = {}
        for triple in triples:
            key = (
                triple.subject_label.lower(),
                triple.predicate_label.upper(),
                triple.object_label.lower(),
                str(triple.metadata.get("alarm_no", "")),
            )
            dedup.setdefault(key, triple)

        state.candidate_triples = list(dedup.values())
        state.log(f"[{self.name}] generated {len(state.candidate_triples)} triples from alarm records")
        return state

    @staticmethod
    def _item_text(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("text_en") or item.get("text_fr") or "").strip()
        return str(item or "").strip()

    @staticmethod
    def _stable_id(label: str) -> str:
        text = str(label or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        return text or "node"

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
                    "metadata": getattr(triple, "metadata", {}),
                }
                for triple in state.candidate_triples
            ],
        }