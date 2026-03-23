from __future__ import annotations

# Standard library imports
import re
from typing import Dict, List
# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.candidates import (
    CandidateMention,
    EntityCandidate,
    RelationCandidate,
    AttributeCandidate,
    EventCandidate,
)
from neoolaf.layers.layer03_candidate_typing_resolution.prompt import (
    build_system_prompt,
    build_user_prompt,
)
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend


class CandidateTypingResolutionLayer(BaseLayer):
    """
    Layer 3: candidate typing and resolution.

    Responsibilities:
    - assign one provisional semantic type to each enriched expression
    - canonicalize labels
    - merge equivalent mentions
    - assign stable candidate identifiers
    """

    name = "layer03_candidate_typing_resolution"

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        max_expressions: int | None = None,
        temperature: float = 0.0,
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize Layer 3.

        Args:
            ollama_backend:
                LLM backend used for candidate typing.
            max_expressions:
                Optional debug limit.
            temperature:
                Generation temperature for the typing prompt.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Wheter to show logs or not.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.max_expressions = max_expressions
        self.temperature = temperature

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Type all enriched expressions and resolve them into canonical candidates.
        """
        enriched_expressions = state.enriched_expressions
        if self.max_expressions is not None:
            enriched_expressions = enriched_expressions[: self.max_expressions]

        typed_items = []

        # ---------------------------------------------------------
        # Step 1: type each enriched expression independently
        # ---------------------------------------------------------
        typing_iterator = enriched_expressions
        if self.verbose:
            typing_iterator = tqdm(enriched_expressions, desc="Layer 3 - typing", leave=False)

        for item in typing_iterator:
            messages = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(item, state.user_guidance, state.seed_ontology)},
            ]

            raw = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )
            parsed = self.ollama_backend.extract_json(raw)

            candidate_type = parsed["candidate_type"].strip()
            canonical_label = parsed["canonical_label"].strip()
            justification = parsed["justification"].strip()
            confidence = parsed.get("confidence")

            # Heuristic correction:
            # if the LLM did not assign relation but the expression strongly
            # looks like a relation-bearing phrase, override softly.
            source_text = item.base_expression.text
            if candidate_type != "relation" and self._looks_like_relation(source_text):
                candidate_type = "relation"

                # If the canonical label is too noun-like, prefer the original text
                if len(source_text.strip()) > 0:
                    canonical_label = source_text.strip()

                # Append traceable explanation
                justification = (
                    justification
                    + " | Heuristic override: expression looks relation-bearing."
                )

            # Normalize canonical label for resolution
            normalized_label = self._normalize_label(canonical_label)

            typed_items.append(
                {
                    "enriched_expression": item,
                    "candidate_type": candidate_type,
                    "canonical_label": canonical_label,
                    "normalized_label": normalized_label,
                    "justification": justification,
                    "confidence": confidence,
                }
            )

        # ---------------------------------------------------------
        # Step 2: group typed items by (candidate_type, normalized_label)
        # ---------------------------------------------------------
        grouped: Dict[tuple, List[dict]] = {}
        for item in typed_items:
            key = (item["candidate_type"], item["normalized_label"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)

        # ---------------------------------------------------------
        # Step 3: build canonical candidate objects with stable IDs
        # ---------------------------------------------------------
        entity_candidates: List[EntityCandidate] = []
        relation_candidates: List[RelationCandidate] = []
        attribute_candidates: List[AttributeCandidate] = []
        event_candidates: List[EventCandidate] = []

        entity_count = 0
        relation_count = 0
        attribute_count = 0
        event_count = 0

        for (candidate_type, normalized_label), items in grouped.items():
            canonical_label = self._choose_canonical_label(items)
            confidence = self._average_confidence(items)

            # Merge mentions
            mentions = []
            aliases = []
            synonyms = []
            lexical_variants = []
            ontology_hints = []
            definitions = []

            for item in items:
                expr = item["enriched_expression"].base_expression
                enriched = item["enriched_expression"]

                mentions.append(
                    CandidateMention(
                        expr_id=expr.expr_id,
                        text=expr.text,
                        evidence=expr.evidence,
                    )
                )

                aliases.extend(enriched.aliases)
                synonyms.extend(enriched.synonyms)
                lexical_variants.extend(enriched.lexical_variants)
                ontology_hints.extend(enriched.ontology_hints)
                if enriched.definition:
                    definitions.append(enriched.definition)

            aliases = self._dedup(aliases)
            synonyms = self._dedup(synonyms)
            lexical_variants = self._dedup(lexical_variants)
            ontology_hints = self._dedup(ontology_hints)
            definition = definitions[0] if definitions else None

            if candidate_type == "entity":
                candidate = EntityCandidate(
                    candidate_id=f"cand_e_{entity_count:05d}",
                    canonical_label=canonical_label,
                    normalized_label=normalized_label,
                    candidate_type="entity",
                    mentions=mentions,
                    confidence=confidence,
                    ontology_hints=ontology_hints,
                    definition=definition,
                    aliases=aliases,
                    synonyms=synonyms,
                    lexical_variants=lexical_variants,
                )
                entity_candidates.append(candidate)
                entity_count += 1

            elif candidate_type == "relation":
                candidate = RelationCandidate(
                    candidate_id=f"cand_r_{relation_count:05d}",
                    canonical_label=canonical_label,
                    normalized_label=normalized_label,
                    candidate_type="relation",
                    mentions=mentions,
                    confidence=confidence,
                    ontology_hints=ontology_hints,
                    definition=definition,
                    aliases=aliases,
                    synonyms=synonyms,
                    lexical_variants=lexical_variants,
                )
                relation_candidates.append(candidate)
                relation_count += 1

            elif candidate_type == "attribute":
                candidate = AttributeCandidate(
                    candidate_id=f"cand_v_{attribute_count:05d}",
                    canonical_label=canonical_label,
                    normalized_label=normalized_label,
                    candidate_type="attribute",
                    mentions=mentions,
                    confidence=confidence,
                    ontology_hints=ontology_hints,
                    definition=definition,
                    aliases=aliases,
                    synonyms=synonyms,
                    lexical_variants=lexical_variants,
                )
                attribute_candidates.append(candidate)
                attribute_count += 1

            elif candidate_type == "event":
                candidate = EventCandidate(
                    candidate_id=f"cand_s_{event_count:05d}",
                    canonical_label=canonical_label,
                    normalized_label=normalized_label,
                    candidate_type="event",
                    mentions=mentions,
                    confidence=confidence,
                    ontology_hints=ontology_hints,
                    definition=definition,
                    aliases=aliases,
                    synonyms=synonyms,
                    lexical_variants=lexical_variants,
                )
                event_candidates.append(candidate)
                event_count += 1

        # ---------------------------------------------------------
        # Step 4: save to state
        # ---------------------------------------------------------
        state.entity_candidates = entity_candidates
        state.relation_candidates = relation_candidates
        state.attribute_candidates = attribute_candidates
        state.event_candidates = event_candidates

        state.log(
            "[layer03_candidate_typing_resolution] "
            f"entities={len(entity_candidates)}, "
            f"relations={len(relation_candidates)}, "
            f"attributes={len(attribute_candidates)}, "
            f"events={len(event_candidates)}"
        )
        return state

    def _normalize_label(self, text: str) -> str:
        """
        Normalize a label for resolution and grouping.
        """
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\-]", "", text)
        return text
    
    def _looks_like_relation(self, text: str) -> bool:
        """
        Lightweight heuristic to detect relation-bearing expressions.

        This is not a final solution, but it helps avoid losing obvious
        verbal and linking phrases before Layer 4.

        The heuristic is intentionally simple and language-agnostic enough
        to still work on French/English industrial text.
        """
        if not text:
            return False

        candidate = text.lower().strip()

        # Common relation-bearing patterns in English/French technical text
        relation_markers = [
            "by",
            "of",
            "into",
            "in",
            "to",
            "from",
            "with",
            "part of",
            "caused by",
            "detected by",
            "emitted by",
            "located in",
            "belongs to",
            "classified in",
            "divided into",
            "indicates",
            "causes",
            "compromises",
            "émis par",
            "émises par",
            "divisés en",
            "divisées en",
            "indiquent",
            "indique",
            "classés dans",
            "classé dans",
            "compromet",
            "détection de",
            "renvoyons à",
        ]

        # If a known marker is present, strongly suspect relation
        for marker in relation_markers:
            if marker in candidate:
                return True

        # Very rough verbal-pattern fallback:
        # multiword expressions containing verbs often behave like relations
        if len(candidate.split()) >= 2:
            # French infinitive / participle or English-like verb-ish endings
            verbish_suffixes = [
                "ed",
                "ing",
                "ize",
                "ise",
                "ant",
                "ent",
                "é",
                "ée",
                "és",
                "ées",
                "ant",
                "er",
                "ir",
                "re",
            ]
            for token in candidate.split():
                if any(token.endswith(suf) for suf in verbish_suffixes):
                    return True

        return False

    def _dedup(self, items: List[str]) -> List[str]:
        """
        Deduplicate strings while preserving order.
        """
        cleaned = [x.strip() for x in items if x and x.strip()]
        return list(dict.fromkeys(cleaned))

    def _choose_canonical_label(self, items: List[dict]) -> str:
        """
        Choose a canonical label for a candidate group.

        Current strategy:
        prefer the shortest non-empty canonical label among the grouped items.
        """
        labels = [item["canonical_label"].strip() for item in items if item["canonical_label"].strip()]
        labels = sorted(labels, key=len)
        return labels[0] if labels else "unknown_candidate"

    def _average_confidence(self, items: List[dict]) -> float | None:
        """
        Average confidence over grouped typed items.
        """
        values = [item["confidence"] for item in items if isinstance(item.get("confidence"), (int, float))]
        if not values:
            return None
        return sum(values) / len(values)

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize typed candidates for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "entity_candidates": [self._serialize_candidate(c) for c in state.entity_candidates],
            "relation_candidates": [self._serialize_candidate(c) for c in state.relation_candidates],
            "attribute_candidates": [self._serialize_candidate(c) for c in state.attribute_candidates],
            "event_candidates": [self._serialize_candidate(c) for c in state.event_candidates],
        }

    def _serialize_candidate(self, candidate) -> dict:
        """
        Serialize a candidate object into a JSON-friendly dictionary.
        """
        return {
            "candidate_id": candidate.candidate_id,
            "canonical_label": candidate.canonical_label,
            "normalized_label": candidate.normalized_label,
            "candidate_type": candidate.candidate_type,
            "confidence": candidate.confidence,
            "ontology_hints": candidate.ontology_hints,
            "definition": candidate.definition,
            "aliases": candidate.aliases,
            "synonyms": candidate.synonyms,
            "lexical_variants": candidate.lexical_variants,
            "mentions": [
                {
                    "expr_id": m.expr_id,
                    "text": m.text,
                    "evidence": [
                        {
                            "chunk_id": ev.chunk_id,
                            "chunk_start_char": ev.chunk_start_char,
                            "chunk_end_char": ev.chunk_end_char,
                            "doc_start_char": ev.doc_start_char,
                            "doc_end_char": ev.doc_end_char,
                            "snippet": ev.snippet,
                        }
                        for ev in m.evidence
                    ],
                }
                for m in candidate.mentions
            ],
        }