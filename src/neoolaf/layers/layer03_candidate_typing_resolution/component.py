from __future__ import annotations

# Standard library imports
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.state_serialization import dump_json, load_json
from neoolaf.domain.enriched_expression import EnrichedExpression
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
from neoolaf.grounding.rag.types import GroundingRequest
from neoolaf.grounding.rag.formatting import build_grounding_context


class CandidateTypingResolutionLayer(BaseLayer):
    """
    Layer 3: candidate typing and resolution.

    The generic strategy keeps the original LLM-based local typing followed by a
    deterministic merge. Profiles can select an ontology-aware role-based typing
    strategy. In that strategy, Layer 3 uses the semantic role already assigned
    at Layer 1/2, links each node candidate to an ontology class, and also creates
    controlled ontology-linked relation candidates. This keeps the XQuality path
    aligned with the seed ontology without allowing open-domain typing drift.
    """

    name = "layer03_candidate_typing_resolution"

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        max_expressions: int | None = None,
        temperature: float = 0.0,
        save_intermediate: bool = True,
        verbose: bool = False,
        rag_adapter=None,
        max_concurrency: int = 1,
        retry_failed_calls: int = 0,
        retry_sleep_seconds: float = 2.0,
        failed_items_file: str | None = None,
    ) -> None:
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.max_expressions = max_expressions
        self.temperature = temperature
        self.rag_adapter = rag_adapter
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.retry_failed_calls = max(0, int(retry_failed_calls or 0))
        self.retry_sleep_seconds = float(retry_sleep_seconds or 0.0)
        self.failed_items_file = failed_items_file
        self._failed_details: list[dict[str, Any]] = []
        self._typing_strategy: str = "generic_llm_typing"

    def _run(self, state: PipelineState) -> PipelineState:
        """Type all enriched expressions and resolve them into canonical candidates."""
        enriched_expressions = list(state.enriched_expressions)
        enriched_expressions = self._filter_failed_items(enriched_expressions)
        if self.max_expressions is not None:
            enriched_expressions = enriched_expressions[: self.max_expressions]

        self._failed_details = []
        self._typing_strategy = self._strategy(state)
        if self.verbose:
            print(f"[NeoOLAF][Layer 3] strategy={self._typing_strategy}")

        if self._is_role_based_strategy(self._typing_strategy):
            typed_items = self._type_items_role_based(enriched_expressions, state)
            self._save_failed_items(state)
        else:
            typed_items = self._type_items_generic(enriched_expressions, state)
            self._save_failed_items(state)

        # ---------------------------------------------------------
        # Step 2: group typed items by (candidate_type, normalized_label)
        # ---------------------------------------------------------
        grouped: Dict[tuple, List[dict]] = {}
        for item in typed_items:
            key = (item["candidate_type"], item["normalized_label"])
            grouped.setdefault(key, []).append(item)

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

            mentions = []
            aliases = []
            synonyms = []
            lexical_variants = []
            ontology_hints = []
            definitions = []

            for item in items:
                enriched = item.get("enriched_expression")
                if enriched is not None:
                    expr = enriched.base_expression
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

                aliases.extend(item.get("aliases", []))
                synonyms.extend(item.get("synonyms", []))
                lexical_variants.extend(item.get("lexical_variants", []))
                ontology_hints.extend(item.get("ontology_hints", []))
                if item.get("definition"):
                    definitions.append(item["definition"])

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

        # Add controlled ontology-linked relation candidates after node grouping.
        if self._is_role_based_strategy(self._typing_strategy):
            relation_candidates.extend(
                self._controlled_relation_candidates(
                    profile=state.profile_config or {},
                    start_index=len(relation_candidates),
                )
            )

        state.entity_candidates = entity_candidates
        state.relation_candidates = relation_candidates
        state.attribute_candidates = attribute_candidates
        state.event_candidates = event_candidates

        state.log(
            "[layer03_candidate_typing_resolution] "
            f"strategy={self._typing_strategy}; "
            f"entities={len(entity_candidates)}, "
            f"relations={len(relation_candidates)}, "
            f"attributes={len(attribute_candidates)}, "
            f"events={len(event_candidates)}, "
            f"failed={len(self._failed_details)}"
        )
        return state

    def _strategy(self, state: PipelineState) -> str:
        layer_cfg = (state.profile_config or {}).get("layers", {}).get(self.name, {})
        return str(layer_cfg.get("strategy", "generic_llm_typing"))

    def _is_role_based_strategy(self, strategy: str) -> bool:
        return strategy in {
            "role_based_typing",
            "ontology_aware_role_based_typing",
            "xquality_role_based_typing",
        }

    def _type_items_generic(self, enriched_expressions: list[EnrichedExpression], state: PipelineState) -> list[dict[str, Any]]:
        """Original parallel LLM local typing path."""
        typed_items: list[dict[str, Any]] = []
        if self.max_concurrency <= 1:
            iterator = enriched_expressions
            if self.verbose:
                iterator = tqdm(enriched_expressions, desc="Layer 3 - typing", leave=False)
            for index, item in enumerate(iterator):
                typed = self._type_item_with_retries(index, item, state)
                if typed is not None:
                    typed_items.append(typed)
        else:
            typed_by_index: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                futures = {
                    executor.submit(self._type_item_with_retries, index, item, state): index
                    for index, item in enumerate(enriched_expressions)
                }
                completed = as_completed(futures)
                if self.verbose:
                    completed = tqdm(
                        completed,
                        total=len(futures),
                        desc=f"Layer 3 - typing x{self.max_concurrency}",
                        leave=False,
                    )
                for future in completed:
                    index = futures[future]
                    try:
                        typed = future.result()
                    except Exception as exc:  # Defensive: _type_item should not raise.
                        self._record_failure(enriched_expressions[index], index, exc, attempt="unhandled")
                        continue
                    if typed is not None:
                        typed_by_index[index] = typed
            typed_items = [typed_by_index[i] for i in sorted(typed_by_index)]
        return typed_items

    def _type_items_role_based(self, enriched_expressions: list[EnrichedExpression], state: PipelineState) -> list[dict[str, Any]]:
        """Deterministically type node expressions from their semantic roles."""
        items: list[dict[str, Any]] = []
        profile = state.profile_config or {}
        iterator = enriched_expressions
        if self.verbose:
            iterator = tqdm(enriched_expressions, desc="Layer 3 - role typing", leave=False)
        for enriched in iterator:
            items.append(self._type_item_role_based(enriched, profile))
        return items

    def _type_item_role_based(self, item: EnrichedExpression, profile: dict[str, Any]) -> dict[str, Any]:
        expr = item.base_expression
        role = self._normalize_role(expr.label)
        link = self._node_role_link(role, profile)

        candidate_type = str(link.get("candidate_family") or "entity").lower()
        if candidate_type not in {"entity", "relation", "attribute", "event"}:
            candidate_type = "entity"

        canonical_label = self._canonical_label_for_item(item, role, profile)
        ontology_hints = self._dedup(
            [
                *item.ontology_hints,
                f"semantic_role:{role}",
                f"ontology_status:{link.get('ontology_status', 'candidate_instance')}",
                f"promote_to_ontology:{str(link.get('promote_to_ontology', False)).lower()}",
                link.get("class_uri"),
                link.get("class_label"),
                *(link.get("additional_hints", []) or []),
            ]
        )
        return {
            "enriched_expression": item,
            "candidate_type": candidate_type,
            "canonical_label": canonical_label,
            "normalized_label": self._normalize_label(canonical_label),
            "justification": f"Role-based typing from Layer 1 label '{expr.label}' and profile ontology mapping.",
            "confidence": 1.0,
            "ontology_hints": ontology_hints,
            "definition": link.get("definition"),
            "aliases": item.aliases,
            "synonyms": item.synonyms,
            "lexical_variants": item.lexical_variants,
        }

    def _controlled_relation_candidates(self, profile: dict[str, Any], start_index: int = 0) -> list[RelationCandidate]:
        """Create controlled relation candidates linked to ontology properties."""
        relation_links = profile.get("ontology_linking", {}).get("relation_roles", {})
        field_to_relation = profile.get("field_to_relation", {}) if isinstance(profile.get("field_to_relation", {}), dict) else {}
        allowed = profile.get("relations", {}).get("allowed", []) if isinstance(profile.get("relations", {}), dict) else []

        relation_labels: list[str] = []
        relation_labels.extend([str(x) for x in allowed if x])
        relation_labels.extend([str(x) for x in field_to_relation.values() if x])
        relation_labels = self._dedup(relation_labels)

        candidates: list[RelationCandidate] = []
        for offset, relation_label in enumerate(relation_labels):
            link = relation_links.get(relation_label, {}) if isinstance(relation_links, dict) else {}
            ontology_hints = self._dedup(
                [
                    f"controlled_relation:{relation_label}",
                    f"ontology_status:{link.get('ontology_status', 'candidate_relation')}",
                    f"promote_to_ontology:{str(link.get('promote_to_ontology', False)).lower()}",
                    link.get("property_uri"),
                    link.get("property_label"),
                    f"domain:{link.get('domain', '')}" if link.get("domain") else None,
                    f"range:{link.get('range', '')}" if link.get("range") else None,
                ]
            )
            candidates.append(
                RelationCandidate(
                    candidate_id=f"cand_r_{start_index + offset:05d}",
                    canonical_label=relation_label,
                    normalized_label=self._normalize_label(relation_label),
                    candidate_type="relation",
                    mentions=[],
                    confidence=1.0,
                    ontology_hints=ontology_hints,
                    definition=link.get("definition") or f"Controlled relation {relation_label} from the document profile.",
                    aliases=self._dedup([relation_label, link.get("property_label")]),
                    synonyms=self._dedup(link.get("synonyms", []) if isinstance(link, dict) else []),
                    lexical_variants=self._dedup([self._normalize_label(relation_label)]),
                )
            )
        return candidates

    def _filter_failed_items(self, items: list[EnrichedExpression]) -> list[EnrichedExpression]:
        """Restrict the run to expression IDs listed in a previous failed_items.json."""
        if not self.failed_items_file:
            return items
        path = Path(self.failed_items_file)
        if not path.exists():
            raise FileNotFoundError(f"Layer 3 failed items file not found: {path}")
        data = load_json(path)
        ids: set[str] = set()
        for item in data.get("failed_items", data if isinstance(data, list) else []):
            if isinstance(item, str):
                ids.add(item)
            elif isinstance(item, dict):
                value = item.get("expr_id") or item.get("id")
                if value:
                    ids.add(str(value))
        return [item for item in items if item.base_expression.expr_id in ids]

    def _type_item_with_retries(
        self,
        index: int,
        item: EnrichedExpression,
        state: PipelineState,
    ) -> dict[str, Any] | None:
        """Type one enriched expression, retrying transient provider/JSON failures."""
        last_exc: Exception | None = None
        for attempt in range(self.retry_failed_calls + 1):
            try:
                return self._type_item(item, state)
            except Exception as exc:
                last_exc = exc
                if attempt < self.retry_failed_calls:
                    if self.retry_sleep_seconds > 0:
                        time.sleep(self.retry_sleep_seconds)
                    continue
        if last_exc is not None:
            self._record_failure(item, index, last_exc, attempt=self.retry_failed_calls)
        return None

    def _type_item(self, item: EnrichedExpression, state: PipelineState) -> dict[str, Any]:
        """Run the LLM typing prompt for one enriched expression."""
        grounding_context = ""
        if self.rag_adapter is not None:
            grounding_result = self.rag_adapter.ground(
                GroundingRequest(
                    layer_name="layer03_candidate_typing_resolution",
                    query=item.base_expression.text,
                    payload={
                        "expression_text": item.base_expression.text,
                        "aliases": item.aliases,
                        "synonyms": item.synonyms,
                        "ontology_hints": item.ontology_hints,
                    },
                    preferred_sources=["ontology", "artifacts", "wikidata", "wikipedia", "wordnet"],
                    top_k=5,
                )
            )
            grounding_context = build_grounding_context(grounding_result)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_user_prompt(
                    enriched_expression=item,
                    guidance=state.user_guidance,
                    seed_ontology=state.seed_ontology,
                    grounding_context=grounding_context,
                ),
            },
        ]

        raw = self.ollama_backend.chat(
            model=state.llm_model,
            messages=messages,
            temperature=self.temperature,
        )
        parsed = self.ollama_backend.extract_json(raw)

        candidate_type = str(parsed.get("candidate_type", "entity")).strip().lower()
        if candidate_type not in {"entity", "relation", "attribute", "event"}:
            candidate_type = "entity"
        canonical_label = str(parsed.get("canonical_label") or item.base_expression.text).strip()
        justification = str(parsed.get("justification") or "").strip()
        confidence = parsed.get("confidence")

        source_text = item.base_expression.text
        if candidate_type != "relation" and self._looks_like_relation(source_text):
            candidate_type = "relation"
            if source_text.strip():
                canonical_label = source_text.strip()
            justification = (
                justification
                + " | Heuristic override: expression looks relation-bearing."
            ).strip()

        normalized_label = self._normalize_label(canonical_label)
        return {
            "enriched_expression": item,
            "candidate_type": candidate_type,
            "canonical_label": canonical_label,
            "normalized_label": normalized_label,
            "justification": justification,
            "confidence": confidence,
        }

    def _record_failure(self, item: EnrichedExpression, index: int, exc: Exception, attempt: int | str) -> None:
        """Record one failed typing item without crashing the full layer."""
        expr = item.base_expression
        self._failed_details.append(
            {
                "index": index,
                "expr_id": expr.expr_id,
                "text": expr.text,
                "label": expr.label,
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    def _save_failed_items(self, state: PipelineState) -> None:
        """Persist failed item IDs and diagnostics for restartable runs."""
        if state.artifact_dir is None:
            return
        layer_dir = Path(state.artifact_dir) / self.name
        layer_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "layer": self.name,
            "failed_count": len(self._failed_details),
            "failed_items": self._failed_details,
        }
        dump_json(layer_dir / "failed_items.json", payload)
        details_dir = layer_dir / "failed_item_details"
        details_dir.mkdir(parents=True, exist_ok=True)
        for item in self._failed_details:
            safe_id = str(item.get("expr_id", item.get("index", "unknown"))).replace("/", "_")
            dump_json(details_dir / f"{safe_id}.json", item)

    def _normalize_role(self, label: str | None) -> str:
        role = (label or "unknown").strip().lower()
        role = re.sub(r"[^a-z0-9_\-]+", "_", role)
        role = role.strip("_") or "unknown"
        role_aliases = {
            "alarm_label": "alarm",
            "alarm_text": "alarm",
            "message_label": "message",
            "message_text": "message",
            "responsible_actor": "responsible",
            "responsible_items": "responsible",
            "reference_items": "reference",
            "intervention_items": "intervention",
            "effect_items": "effect",
            "cause_items": "cause",
        }
        return role_aliases.get(role, role)

    def _node_role_link(self, role: str, profile: dict[str, Any]) -> dict[str, Any]:
        node_roles = profile.get("ontology_linking", {}).get("node_roles", {})
        if isinstance(node_roles, dict) and isinstance(node_roles.get(role), dict):
            return dict(node_roles[role])
        return {
            "class_label": role.title(),
            "class_uri": f"http://neoolaf.org/ontology/{role.title().replace('_', '')}",
            "candidate_family": "entity",
            "ontology_status": "candidate_instance",
            "promote_to_ontology": False,
        }

    def _canonical_label_for_item(self, item: EnrichedExpression, role: str, profile: dict[str, Any]) -> str:
        text = item.base_expression.text.strip() or "unknown_candidate"
        mappings = profile.get("canonical_node_label_mappings", {})
        if isinstance(mappings, dict):
            role_mapping = mappings.get(role, {}) or mappings.get("global", {}) or {}
            if isinstance(role_mapping, dict):
                lowered = text.lower()
                for source, target in role_mapping.items():
                    if lowered == str(source).lower():
                        return str(target)
        # Keep the extracted source expression as the canonical node label unless
        # a profile-level canonical mapping explicitly rewrites it. Aliases such
        # as "cause" or "effect" are role hints, not node labels.
        return text

    def _normalize_label(self, text: str) -> str:
        """Normalize a label for resolution and grouping."""
        text = str(text or "").lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\-]", "", text)
        return text

    def _looks_like_relation(self, text: str) -> bool:
        """Lightweight heuristic to detect relation-bearing expressions."""
        if not text:
            return False
        candidate = text.lower().strip()
        relation_markers = [
            "by", "of", "into", "in", "to", "from", "with", "part of",
            "caused by", "detected by", "emitted by", "located in", "belongs to",
            "classified in", "divided into", "indicates", "causes", "compromises",
            "émis par", "émises par", "divisés en", "divisées en", "indiquent",
            "indique", "classés dans", "classé dans", "compromet", "détection de",
            "renvoyons à",
        ]
        for marker in relation_markers:
            if marker in candidate:
                return True
        if len(candidate.split()) >= 2:
            verbish_suffixes = ["ed", "ing", "ize", "ise", "ant", "ent", "é", "ée", "és", "ées", "er", "ir", "re"]
            for token in candidate.split():
                if any(token.endswith(suf) for suf in verbish_suffixes):
                    return True
        return False

    def _dedup(self, items: List[Any]) -> List[str]:
        """Deduplicate strings while preserving order."""
        cleaned = []
        for item in items or []:
            if item is None:
                continue
            if not isinstance(item, str):
                item = str(item)
            item = item.strip()
            if item:
                cleaned.append(item)
        return list(dict.fromkeys(cleaned))

    def _choose_canonical_label(self, items: List[dict]) -> str:
        """Choose a canonical label for a candidate group."""
        labels = [item["canonical_label"].strip() for item in items if item["canonical_label"].strip()]
        labels = sorted(labels, key=lambda x: (len(x), x.lower()))
        return labels[0] if labels else "unknown_candidate"

    def _average_confidence(self, items: List[dict]) -> float | None:
        """Average confidence over grouped typed items."""
        values = [item["confidence"] for item in items if isinstance(item.get("confidence"), (int, float))]
        if not values:
            return None
        return sum(values) / len(values)

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """Serialize typed candidates for debugging and reproducibility."""
        return {
            "layer": self.name,
            "strategy": getattr(self, "_typing_strategy", "unknown"),
            "failed_count": len(getattr(self, "_failed_details", [])),
            "entity_candidates": [self._serialize_candidate(c) for c in state.entity_candidates],
            "relation_candidates": [self._serialize_candidate(c) for c in state.relation_candidates],
            "attribute_candidates": [self._serialize_candidate(c) for c in state.attribute_candidates],
            "event_candidates": [self._serialize_candidate(c) for c in state.event_candidates],
        }

    def _serialize_candidate(self, candidate) -> dict:
        """Serialize a candidate object into a JSON-friendly dictionary."""
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
