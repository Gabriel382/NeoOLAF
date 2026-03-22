from __future__ import annotations

# Standard library imports
import json
from pathlib import Path


class KGJSONSerialiser:
    """
    Serialize NeoOLAF KG outputs into JSON.
    """

    def serialise_local(self, state, output_path: str) -> None:
        """
        Serialize local KG from candidate triples.
        """
        payload = {
            "triples": [
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
            ]
        }
        self._write_json(payload, output_path)

    def serialise_inferred(self, state, output_path: str) -> None:
        """
        Serialize inferred/completed KG into JSON.
        """
        inferred_triples = []
        if state.reasoning_report is not None:
            inferred_triples.extend(state.reasoning_report.inferred_triples)

        for completion in state.completion_candidates:
            if completion.completed_triple is not None:
                inferred_triples.append(completion.completed_triple)

        dedup = {}
        for triple in inferred_triples:
            key = (triple.subject_id, triple.predicate_id, triple.object_id, triple.chunk_id)
            if key not in dedup:
                dedup[key] = triple

        payload = {
            "triples": [
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
                }
                for triple in dedup.values()
            ]
        }
        self._write_json(payload, output_path)

    def _write_json(self, payload: dict, output_path: str) -> None:
        """
        Write JSON to disk.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)