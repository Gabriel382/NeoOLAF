from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm.auto import tqdm


# ============================================================
# Resolve project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
COMMON_DIR = PROJECT_ROOT / "experiments" / "common"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))


# ============================================================
# Local common imports
# ============================================================

from jsonl_adapter import count_documents, iter_documents  # type: ignore


# ============================================================
# NeoOLAF imports
# ============================================================

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner
from neoolaf.core.execution_config import ExecutionConfig

from neoolaf.domain.documents import Document
from neoolaf.domain.user_guidance import (
    UserGuidance,
    TypingExample,
    RelationExample,
    PromotionExample,
    NegativeExample,
)

from neoolaf.resources.llm_backends.openai_backend import OpenAIBackend
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend
from neoolaf.resources.translation.deep_translator_backend import DeepTranslatorBackend

from neoolaf.resources.knowledge_sources.wordnet_source import WordNetSource
from neoolaf.resources.knowledge_sources.wikipedia_source import WikipediaSource
from neoolaf.resources.knowledge_sources.wikidata_source import WikidataSource
from neoolaf.resources.knowledge_sources.web_search_source import WebSearchSource

from neoolaf.ontology.factory import build_seed_ontology

from neoolaf.grounding.rag.registry import RetrievalRegistry
from neoolaf.grounding.rag.engine import SemanticRAGEngine
from neoolaf.grounding.rag.adapters.neoolaf_semantic_rag_adapter import NeoOLAFSemanticRAGAdapter
from neoolaf.grounding.rag.spaces.ontology_space import OntologySpace
from neoolaf.grounding.rag.spaces.artifact_space import ArtifactSpace
from neoolaf.grounding.rag.spaces.web_space import WebSpace
from neoolaf.grounding.rag.spaces.wikidata_space import WikidataSpace
from neoolaf.grounding.rag.spaces.wikipedia_space import WikipediaSpace
from neoolaf.grounding.rag.spaces.wordnet_space import WordNetSpace

from neoolaf.layers.layer00_preprocessing.component import PreprocessingLayer
from neoolaf.layers.layer01_linguistic_expression_extraction.component import LinguisticExpressionExtractionLayer
from neoolaf.layers.layer02_candidate_enrichment.component import CandidateEnrichmentLayer
from neoolaf.layers.layer03_candidate_typing_resolution.component import CandidateTypingResolutionLayer
from neoolaf.layers.layer04_candidate_relation_extraction.component import CandidateRelationExtractionLayer
from neoolaf.layers.layer05_candidate_triple_generation.component import CandidateTripleGenerationLayer
from neoolaf.layers.layer06_concept_relation_induction.component import ConceptRelationInductionLayer
from neoolaf.layers.layer07_hierarchisation.component import HierarchisationLayer
from neoolaf.layers.layer08_axiom_schemata_extraction.component import AxiomSchemataExtractionLayer
from neoolaf.layers.layer09_general_axiom_extraction.component import GeneralAxiomExtractionLayer
from neoolaf.layers.layer10_validation_reasoning.component import ValidationReasoningLayer
from neoolaf.layers.layer11_inference_completion.component import InferenceCompletionLayer
from neoolaf.layers.layer12_serialization.component import SerializationLayer


# ============================================================
# Generic helpers
# ============================================================

def normalize_text_field(value: Any) -> str:
    """Convert a possibly missing field into a clean string."""
    if value is None:
        return ""
    return str(value).strip()


def ensure_parent_dir(path: Path) -> None:
    """Create parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSON object as one JSONL line."""
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json_if_exists(path: str | Path | None) -> Any:
    """Read JSON file if path is provided and exists."""
    if path is None:
        return None

    path_str = str(path).strip()
    if not path_str:
        return None

    json_path = Path(path_str)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dotenv_file(env_path: Path) -> None:
    """Minimal .env loader."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def safe_filename(value: str, max_len: int = 120) -> str:
    """Create a filesystem-safe name."""
    value = normalize_text_field(value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("_")

    if not value:
        value = "document"

    return value[:max_len]


def safe_document_id(doc: Dict[str, Any], fallback_index: int) -> str:
    """Return a robust document id."""
    doc_id = normalize_text_field(doc.get("document_id"))

    if doc_id:
        return doc_id

    title = normalize_text_field(doc.get("title"))

    if title:
        return f"doc_{fallback_index:07d}_{title}"

    return f"doc_{fallback_index:07d}"


def build_document_text(doc: Dict[str, Any]) -> str:
    """
    Build document text from common fields.

    Priority:
    1. text
    2. sentences
    3. tokens
    """
    text = normalize_text_field(doc.get("text"))

    if text:
        return text

    sentences = doc.get("sentences")

    if isinstance(sentences, list) and sentences:
        merged = "\n".join(str(x).strip() for x in sentences if str(x).strip())

        if merged.strip():
            return merged

    tokens = doc.get("tokens")

    if isinstance(tokens, list) and tokens:
        rebuilt_sentences: List[str] = []

        for sent in tokens:
            if isinstance(sent, list):
                rebuilt = " ".join(str(tok) for tok in sent).strip()

                if rebuilt:
                    rebuilt_sentences.append(rebuilt)

        if rebuilt_sentences:
            return "\n".join(rebuilt_sentences)

    return ""


def load_processed_ids(output_jsonl_path: Path) -> Set[str]:
    """Load already processed ids from output JSONL for resume mode."""
    processed: Set[str] = set()

    if not output_jsonl_path.exists():
        return processed

    with output_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
                doc_id = normalize_text_field(row.get("document_id"))

                if doc_id:
                    processed.add(doc_id)

            except Exception:
                continue

    return processed


def to_jsonable(value: Any) -> Any:
    """
    Convert dataclasses and complex objects into JSON-serializable structures.

    This is used only for compact raw summaries/debug output.
    """
    if dataclasses.is_dataclass(value):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(x) for x in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


# ============================================================
# Backend handling
# ============================================================

class OpenAIBackendCompatAdapter:
    """
    Adapter around NeoOLAF OpenAIBackend.

    NeoOLAF layers call:
        backend.chat(model=..., messages=..., temperature=...)

    Some external code may call:
        backend.chat(model_name=..., messages=..., max_tokens=...)

    This adapter accepts both.
    """

    def __init__(self, backend: OpenAIBackend) -> None:
        self.backend = backend

    def chat(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        **_: Any,
    ) -> str:
        """Forward chat call to NeoOLAF OpenAIBackend."""
        selected_model = model or model_name

        if not selected_model:
            raise ValueError("No model/model_name provided to backend.chat().")

        if messages is None:
            raise ValueError("No messages provided to backend.chat().")

        return self.backend.chat(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )

    @staticmethod
    def extract_json(text: str) -> Any:
        """Expose OpenAIBackend JSON extraction helper."""
        return OpenAIBackend.extract_json(text)


class OllamaBackendCompatAdapter:
    """
    Adapter around NeoOLAF OllamaBackend.

    It keeps the NeoOLAF expected interface while tolerating max_tokens/model_name.
    """

    def __init__(self, backend: OllamaBackend) -> None:
        self.backend = backend

    def chat(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        model_name: Optional[str] = None,
        **_: Any,
    ) -> str:
        """Forward chat call to NeoOLAF OllamaBackend."""
        selected_model = model or model_name

        if not selected_model:
            raise ValueError("No model/model_name provided to backend.chat().")

        if messages is None:
            raise ValueError("No messages provided to backend.chat().")

        return self.backend.chat(
            model=selected_model,
            messages=messages,
            temperature=temperature,
        )

    @staticmethod
    def extract_json(text: str) -> Any:
        """Expose OllamaBackend JSON extraction helper."""
        return OllamaBackend.extract_json(text)


def normalize_openai_compatible_host(backend_name: str, host: str) -> str:
    """
    Normalize host for NeoOLAF OpenAIBackend.

    NeoOLAF OpenAIBackend appends:
        /v1/chat/completions

    Therefore:
    - vLLM host http://localhost:8000/v1 must become http://localhost:8000
    - OpenRouter host https://openrouter.ai/api should stay https://openrouter.ai/api
    """
    host = host.rstrip("/")
    backend_name = backend_name.strip().lower()

    if backend_name == "vllm" and host.endswith("/v1"):
        host = host[:-3].rstrip("/")

    return host


def build_neoolaf_backend(
    backend_name: str,
    host: str,
    api_key: str,
    timeout: int,
    max_retries: int,
    retry_wait_seconds: float,
    referer: str,
    title: str,
) -> Any:
    """Build vllm/openrouter/ollama backend accepted by NeoOLAF layers."""
    backend_name = backend_name.strip().lower()

    if backend_name == "ollama":
        return OllamaBackendCompatAdapter(
            OllamaBackend(
                host=host,
                timeout=timeout,
            )
        )

    if backend_name in {"vllm", "openrouter"}:
        normalized_host = normalize_openai_compatible_host(backend_name, host)

        return OpenAIBackendCompatAdapter(
            OpenAIBackend(
                host=normalized_host,
                api_key=api_key or "dummy",
                timeout=timeout,
                max_retries=max_retries,
                retry_wait_seconds=retry_wait_seconds,
                referer=referer or None,
                title=title or None,
                retry_on_empty=True,
            )
        )

    raise ValueError(
        f"Unsupported backend_name={backend_name}. "
        "Use one of: vllm, openrouter, ollama."
    )


# ============================================================
# UserGuidance and few-shot handling
# ============================================================

def build_default_guidance() -> UserGuidance:
    """Build a generic guidance object for document-level KG construction."""
    return UserGuidance(
        domain_focus=(
            "document-level knowledge graph construction from text, including named entities, "
            "events, concepts, and explicit relations supported by the document"
        ),
        abstraction_level=(
            "Extract concrete named entities as entities, explicit events or states as events, "
            "and relation phrases as relations. Keep labels close to the wording or ontology labels "
            "used by the dataset when possible."
        ),
        priority_relations=[],
        population_policy=(
            "Keep document-specific named entities as individuals. Promote stable recurrent patterns "
            "only when clearly supported."
        ),
        event_modeling_preference=(
            "Treat explicit events, states, actions, and causal situations as events when relevant. "
            "Treat persons, organizations, locations, dates, quantities, and domain objects as entities."
        ),
        ontology_depth="balanced",
    )


def parse_typing_examples(items: Any) -> List[TypingExample]:
    """Parse typing examples from JSON."""
    out: List[TypingExample] = []

    if not isinstance(items, list):
        return out

    for item in items:
        if not isinstance(item, dict):
            continue

        text = normalize_text_field(item.get("text"))
        expected_type = normalize_text_field(item.get("expected_type"))

        if not text or not expected_type:
            continue

        out.append(
            TypingExample(
                text=text,
                expected_type=expected_type,
                explanation=item.get("explanation"),
            )
        )

    return out


def parse_relation_examples(items: Any) -> List[RelationExample]:
    """Parse relation examples from JSON."""
    out: List[RelationExample] = []

    if not isinstance(items, list):
        return out

    for item in items:
        if not isinstance(item, dict):
            continue

        text = normalize_text_field(item.get("text"))
        source_label = normalize_text_field(item.get("source_label"))
        relation_label = normalize_text_field(item.get("relation_label"))
        target_label = normalize_text_field(item.get("target_label"))

        if not source_label or not relation_label or not target_label:
            continue

        if not text:
            text = f"{source_label} {relation_label} {target_label}"

        out.append(
            RelationExample(
                text=text,
                source_label=source_label,
                relation_label=relation_label,
                target_label=target_label,
                explanation=item.get("explanation"),
            )
        )

    return out


def parse_promotion_examples(items: Any) -> List[PromotionExample]:
    """Parse promotion examples from JSON."""
    out: List[PromotionExample] = []

    if not isinstance(items, list):
        return out

    for item in items:
        if not isinstance(item, dict):
            continue

        text = normalize_text_field(item.get("text"))

        if not text:
            continue

        out.append(
            PromotionExample(
                text=text,
                promote=bool(item.get("promote", True)),
                promoted_label=item.get("promoted_label"),
                explanation=item.get("explanation"),
            )
        )

    return out


def parse_negative_examples(items: Any) -> List[NegativeExample]:
    """Parse negative examples from JSON."""
    out: List[NegativeExample] = []

    if not isinstance(items, list):
        return out

    for item in items:
        if not isinstance(item, dict):
            continue

        text = normalize_text_field(item.get("text"))

        if not text:
            continue

        out.append(
            NegativeExample(
                text=text,
                explanation=item.get("explanation"),
                target_layer=item.get("target_layer"),
            )
        )

    return out


def parse_user_guidance_json(data: Any) -> UserGuidance:
    """
    Parse UserGuidance from a JSON object.

    Supported fields:
    - domain_focus
    - abstraction_level
    - priority_relations
    - population_policy
    - event_modeling_preference
    - ontology_depth
    - promotion_min_confidence
    - hierarchy_min_confidence
    - concept_promotion_bias
    - typing_examples
    - relation_examples
    - promotion_examples
    - negative_examples
    """
    if not isinstance(data, dict):
        raise ValueError("User guidance JSON must be an object.")

    guidance = UserGuidance(
        domain_focus=data.get("domain_focus"),
        abstraction_level=data.get("abstraction_level"),
        priority_relations=data.get("priority_relations") or [],
        population_policy=data.get("population_policy"),
        event_modeling_preference=data.get("event_modeling_preference"),
        ontology_depth=data.get("ontology_depth", "balanced"),
        promotion_min_confidence=float(data.get("promotion_min_confidence", 0.50)),
        hierarchy_min_confidence=float(data.get("hierarchy_min_confidence", 0.50)),
        concept_promotion_bias=float(data.get("concept_promotion_bias", 0.50)),
        typing_examples=parse_typing_examples(data.get("typing_examples", [])),
        relation_examples=parse_relation_examples(data.get("relation_examples", [])),
        promotion_examples=parse_promotion_examples(data.get("promotion_examples", [])),
        negative_examples=parse_negative_examples(data.get("negative_examples", [])),
    )

    return guidance


def load_user_guidance(user_guidance_path: str | Path | None) -> UserGuidance:
    """Load user guidance JSON or fallback to generic guidance."""
    data = read_json_if_exists(user_guidance_path)

    if data is None:
        return build_default_guidance()

    return parse_user_guidance_json(data)


def add_xquality_style_few_shots(guidance: UserGuidance, few_shots: List[Dict[str, Any]], max_examples: int) -> None:
    """
    Add XQuality-style few-shot examples.

    Expected shape:
    {
      "alarm_label": "...",
      "triples": [
        {"node_1": "...", "relation": "...", "node_2": "..."}
      ]
    }
    """
    for example in few_shots[:max_examples]:
        if not isinstance(example, dict):
            continue

        alarm_label = normalize_text_field(example.get("alarm_label"))

        if alarm_label:
            guidance.typing_examples.append(
                TypingExample(text=alarm_label, expected_type="event")
            )
            guidance.promotion_examples.append(
                PromotionExample(
                    text=alarm_label,
                    promote=True,
                    promoted_label=alarm_label.title().replace(" ", ""),
                )
            )

        triples = example.get("triples", [])

        if isinstance(triples, list):
            for triple in triples[:8]:
                if not isinstance(triple, dict):
                    continue

                src = normalize_text_field(triple.get("node_1") or triple.get("head") or triple.get("source"))
                rel = normalize_text_field(triple.get("relation") or triple.get("rel") or triple.get("predicate"))
                tgt = normalize_text_field(triple.get("node_2") or triple.get("tail") or triple.get("target"))

                if src and rel and tgt:
                    guidance.relation_examples.append(
                        RelationExample(
                            text=f"{src} {rel} {tgt}",
                            source_label=src,
                            relation_label=rel,
                            target_label=tgt,
                        )
                    )


def add_guidance_style_few_shots(guidance: UserGuidance, few_shots: Any) -> None:
    """
    Add few-shots that already use guidance-like fields.

    Supported keys:
    - typing_examples
    - relation_examples
    - promotion_examples
    - negative_examples
    """
    if isinstance(few_shots, dict):
        guidance.typing_examples.extend(parse_typing_examples(few_shots.get("typing_examples", [])))
        guidance.relation_examples.extend(parse_relation_examples(few_shots.get("relation_examples", [])))
        guidance.promotion_examples.extend(parse_promotion_examples(few_shots.get("promotion_examples", [])))
        guidance.negative_examples.extend(parse_negative_examples(few_shots.get("negative_examples", [])))

    elif isinstance(few_shots, list):
        for item in few_shots:
            if isinstance(item, dict) and any(
                key in item for key in ["typing_examples", "relation_examples", "promotion_examples", "negative_examples"]
            ):
                add_guidance_style_few_shots(guidance, item)


def enrich_guidance_with_few_shots(
    guidance: UserGuidance,
    few_shot_path: str | Path | None,
    few_shot_k: int,
) -> UserGuidance:
    """Load and inject few-shot examples into UserGuidance."""
    few_shots = read_json_if_exists(few_shot_path)

    if few_shots is None:
        return guidance

    # Case 1: JSON object already containing UserGuidance-style examples.
    add_guidance_style_few_shots(guidance, few_shots)

    # Case 2: XQuality-style list of alarm/triple examples.
    if isinstance(few_shots, list):
        add_xquality_style_few_shots(guidance, few_shots, max_examples=few_shot_k)

    return guidance


def select_dataset_few_shot_examples(
    dataset_jsonl_path: Path,
    current_document_id: str,
    type_filter: str,
    few_shot_k: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Select few-shot examples from the same dataset.

    This is optional and used only when --few-shot-from-dataset is enabled.
    It builds simple relation examples from gold labels.
    """
    if few_shot_k <= 0:
        return []

    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    for doc in iter_documents(dataset_jsonl_path, type_filter=type_filter):
        document_id = normalize_text_field(doc.get("document_id"))

        if document_id == current_document_id:
            continue

        relations = doc.get("relations", [])

        has_relations = (
            (isinstance(relations, list) and len(relations) > 0)
            or (isinstance(relations, dict) and len(relations) > 0)
        )

        if not has_relations:
            continue

        seen += 1

        if len(reservoir) < few_shot_k:
            reservoir.append(doc)
        else:
            j = rng.randint(1, seen)

            if j <= few_shot_k:
                reservoir[j - 1] = doc

    return reservoir


def add_dataset_examples_to_guidance(
    guidance: UserGuidance,
    examples: List[Dict[str, Any]],
    max_relations_per_doc: int = 8,
) -> None:
    """Inject simple relation examples from normalized dataset entries."""
    for doc in examples:
        relations = doc.get("relations", [])

        if isinstance(relations, list):
            for rel in relations[:max_relations_per_doc]:
                if not isinstance(rel, dict):
                    continue

                head = normalize_text_field(rel.get("head_text") or rel.get("head"))
                tail = normalize_text_field(rel.get("tail_text") or rel.get("tail"))
                label = normalize_text_field(rel.get("relation") or rel.get("rel"))

                if head and label and tail:
                    guidance.relation_examples.append(
                        RelationExample(
                            text=f"{head} {label} {tail}",
                            source_label=head,
                            relation_label=label,
                            target_label=tail,
                        )
                    )

        # Original DocRED-like format is usually already normalized by iter_documents.
        # This fallback is kept for safety.
        elif isinstance(relations, dict):
            entities = doc.get("entities", {})
            id_to_label: Dict[str, str] = {}

            if isinstance(entities, dict):
                for entity_id, entity_info in entities.items():
                    if not isinstance(entity_info, dict):
                        continue

                    mentions = entity_info.get("mentions", [])

                    if isinstance(mentions, list) and mentions:
                        mention = mentions[0]

                        if isinstance(mention, dict):
                            trigger = normalize_text_field(mention.get("trigger_word"))

                            if trigger:
                                id_to_label[str(entity_id)] = trigger

            relation_count = 0

            for rel_label, pairs in relations.items():
                if relation_count >= max_relations_per_doc:
                    break

                if not isinstance(pairs, list):
                    continue

                for pair in pairs:
                    if relation_count >= max_relations_per_doc:
                        break

                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue

                    src = id_to_label.get(str(pair[0]), "")
                    tgt = id_to_label.get(str(pair[1]), "")
                    rel_name = normalize_text_field(rel_label)

                    if src and rel_name and tgt:
                        guidance.relation_examples.append(
                            RelationExample(
                                text=f"{src} {rel_name} {tgt}",
                                source_label=src,
                                relation_label=rel_name,
                                target_label=tgt,
                            )
                        )
                        relation_count += 1


# ============================================================
# NeoOLAF pipeline construction
# ============================================================

def build_semantic_rag_adapter(
    state: PipelineState,
    llm_backend: Any,
    model_name: str,
    use_web_search: bool,
    wikipedia_source: WikipediaSource,
    wikidata_source: WikidataSource,
    wordnet_source: WordNetSource,
    web_search_source: WebSearchSource,
) -> NeoOLAFSemanticRAGAdapter:
    """Build SemanticRAG adapter for one document state."""
    registry = RetrievalRegistry()

    if state.seed_ontology is not None:
        registry.register(OntologySpace(state.seed_ontology))

    registry.register(ArtifactSpace(state))
    registry.register(WikidataSpace(wikidata_source))
    registry.register(WikipediaSpace(wikipedia_source))
    registry.register(WordNetSpace(wordnet_source))

    if use_web_search:
        registry.register(WebSpace(web_search_source))

    semantic_rag_engine = SemanticRAGEngine(
        registry=registry,
        ollama_backend=llm_backend,
        model_name=model_name,
    )

    return NeoOLAFSemanticRAGAdapter(semantic_rag_engine)


def build_neoolaf_pipeline(
    llm_backend: Any,
    model_name: str,
    state: PipelineState,
    chunk_size: Optional[int],
    chunk_overlap: int,
    translate_to_english: bool,
    use_web_search: bool,
    max_chunks: Optional[int],
    max_expressions: Optional[int],
    max_relation_mentions: Optional[int],
    max_workers: int,
    verbose: bool,
    output_subdir: str,
    base_uri: str,
    wikipedia_source: WikipediaSource,
    wikidata_source: WikidataSource,
    wordnet_source: WordNetSource,
    web_search_source: WebSearchSource,
) -> Tuple[Pipeline, ExecutionConfig]:
    """Build the full NeoOLAF pipeline and execution config."""
    translator = DeepTranslatorBackend(max_chars=4000) if translate_to_english else None

    rag_adapter = build_semantic_rag_adapter(
        state=state,
        llm_backend=llm_backend,
        model_name=model_name,
        use_web_search=use_web_search,
        wikipedia_source=wikipedia_source,
        wikidata_source=wikidata_source,
        wordnet_source=wordnet_source,
        web_search_source=web_search_source,
    )

    pipeline = Pipeline(
        layers=[
            PreprocessingLayer(
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                translate=translate_to_english,
                translator=translator,
                verbose=verbose,
            ),
            LinguisticExpressionExtractionLayer(
                ollama_backend=llm_backend,
                max_chunks=max_chunks,
                temperature=0.0,
                verbose=verbose,
            ),
            CandidateEnrichmentLayer(
                ollama_backend=llm_backend,
                wikipedia_source=wikipedia_source,
                wikidata_source=wikidata_source,
                wordnet_source=wordnet_source,
                web_search_source=web_search_source,
                max_expressions=max_expressions,
                use_web_search=use_web_search,
                rag_adapter=rag_adapter,
                verbose=verbose,
            ),
            CandidateTypingResolutionLayer(
                ollama_backend=llm_backend,
                max_expressions=max_expressions,
                temperature=0.0,
                rag_adapter=rag_adapter,
                verbose=verbose,
            ),
            CandidateRelationExtractionLayer(
                ollama_backend=llm_backend,
                max_relation_mentions=max_relation_mentions,
                temperature=0.0,
                rag_adapter=rag_adapter,
                verbose=verbose,
            ),
            CandidateTripleGenerationLayer(
                max_assertions=None,
                verbose=verbose,
            ),
            ConceptRelationInductionLayer(
                ollama_backend=llm_backend,
                max_concept_inputs=None,
                max_relation_inputs=None,
                temperature=0.0,
                rag_adapter=rag_adapter,
                verbose=verbose,
            ),
            HierarchisationLayer(
                ollama_backend=llm_backend,
                max_concept_pairs=None,
                max_relation_pairs=None,
                temperature=0.0,
                rag_adapter=rag_adapter,
                verbose=verbose,
            ),
            AxiomSchemataExtractionLayer(
                ollama_backend=llm_backend,
                max_relation_schema_inputs=None,
                max_subclass_inputs=None,
                temperature=0.0,
                rag_adapter=rag_adapter,
                verbose=verbose,
            ),
            GeneralAxiomExtractionLayer(
                ollama_backend=llm_backend,
                max_schema_inputs=None,
                max_description_inputs=None,
                temperature=0.0,
                verbose=verbose,
            ),
            ValidationReasoningLayer(
                max_triples=None,
                verbose=verbose,
            ),
            InferenceCompletionLayer(
                max_inferred_triples=None,
                verbose=verbose,
            ),
            SerializationLayer(
                output_subdir=output_subdir,
                base_uri=base_uri,
                verbose=verbose,
            ),
        ],
        verbose=verbose,
        continue_from_last=False,
    )

    execution_config = ExecutionConfig(
        mode="chunk_iterative_mode",
        chunk_loop_enabled=True,
        chunk_layer_names=[
            "layer01_linguistic_expression_extraction",
            "layer02_candidate_enrichment",
            "layer03_candidate_typing_resolution",
            "layer04_candidate_relation_extraction",
            "layer05_candidate_triple_generation",
        ],
        global_layer_names=[
            "layer06_concept_relation_induction",
            "layer07_hierarchisation",
            "layer08_axiom_schemata_extraction",
            "layer09_general_axiom_extraction",
            "layer10_validation_reasoning",
            "layer11_inference_completion",
            "layer12_serialization",
        ],
        max_chunks=max_chunks,
    )

    return pipeline, execution_config


# ============================================================
# Canonicalization for eval_relations.py
# ============================================================

def evidence_to_text(evidence_items: Any) -> str:
    """Convert NeoOLAF evidence/provenance list into a compact evidence string."""
    if not evidence_items:
        return ""

    snippets: List[str] = []

    for item in evidence_items:
        if dataclasses.is_dataclass(item):
            item_dict = dataclasses.asdict(item)
        elif isinstance(item, dict):
            item_dict = item
        else:
            text = normalize_text_field(item)
            if text:
                snippets.append(text)
            continue

        for key in ["text", "snippet", "sentence", "content"]:
            value = normalize_text_field(item_dict.get(key))
            if value:
                snippets.append(value)
                break

    return " | ".join(snippets[:3])


def candidate_to_canonical_entity(candidate: Any) -> Optional[Dict[str, str]]:
    """Convert a NeoOLAF candidate into evaluator-compatible entity format."""
    label = normalize_text_field(getattr(candidate, "canonical_label", ""))
    candidate_type = normalize_text_field(getattr(candidate, "candidate_type", ""))
    definition = normalize_text_field(getattr(candidate, "definition", ""))

    if not label:
        return None

    return {
        "label": label,
        "type": candidate_type,
        "description": definition,
    }


def triple_to_canonical_relation(triple: Any) -> Optional[Dict[str, str]]:
    """Convert a NeoOLAF CandidateTriple into evaluator-compatible relation format."""
    head = normalize_text_field(getattr(triple, "subject_label", ""))
    relation = normalize_text_field(getattr(triple, "predicate_label", ""))
    tail = normalize_text_field(getattr(triple, "object_label", ""))
    justification = normalize_text_field(getattr(triple, "justification", ""))

    provenance_text = evidence_to_text(getattr(triple, "provenance", []))
    evidence = justification or provenance_text

    if not head or not relation or not tail:
        return None

    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "evidence": evidence,
    }


def canonicalize_neoolaf_state(
    final_state: PipelineState,
    document_id: str,
    title: str,
    doc_type: str,
) -> Dict[str, Any]:
    """Convert final NeoOLAF state into eval_relations.py-compatible row."""
    canonical_entities: List[Dict[str, str]] = []
    canonical_relations: List[Dict[str, str]] = []

    seen_entities: Set[Tuple[str, str]] = set()
    seen_relations: Set[Tuple[str, str, str]] = set()

    candidate_groups = [
        final_state.entity_candidates,
        final_state.event_candidates,
        final_state.attribute_candidates,
    ]

    for group in candidate_groups:
        for candidate in group:
            entity = candidate_to_canonical_entity(candidate)

            if entity is None:
                continue

            key = (
                entity["label"].lower(),
                entity["type"].lower(),
            )

            if key not in seen_entities:
                seen_entities.add(key)
                canonical_entities.append(entity)

    for triple in final_state.candidate_triples:
        relation = triple_to_canonical_relation(triple)

        if relation is None:
            continue

        key = (
            relation["head"].lower(),
            relation["relation"].lower(),
            relation["tail"].lower(),
        )

        if key not in seen_relations:
            seen_relations.add(key)
            canonical_relations.append(relation)

    return {
        "document_id": document_id,
        "title": title,
        "type": doc_type,
        "method": "neoolaf",
        "parsed_ok": True,
        "prediction": {
            "entities": canonical_entities,
            "relations": canonical_relations,
        },
        "raw_counts": {
            "linguistic_expressions": len(final_state.linguistic_expressions),
            "enriched_expressions": len(final_state.enriched_expressions),
            "entity_candidates": len(final_state.entity_candidates),
            "event_candidates": len(final_state.event_candidates),
            "attribute_candidates": len(final_state.attribute_candidates),
            "relation_candidates": len(final_state.relation_candidates),
            "candidate_relation_assertions": len(final_state.candidate_relation_assertions),
            "candidate_triples": len(final_state.candidate_triples),
            "concept_candidates": len(final_state.concept_candidates),
            "ontology_relation_candidates": len(final_state.ontology_relation_candidates),
            "completion_candidates": len(final_state.completion_candidates),
            "canonical_entities": len(canonical_entities),
            "canonical_relations": len(canonical_relations),
        },
        "artifact_dir": final_state.artifact_dir,
    }


def build_raw_neoolaf_summary(final_state: PipelineState) -> Dict[str, Any]:
    """Build compact raw debug summary from final NeoOLAF state."""
    return {
        "artifact_dir": final_state.artifact_dir,
        "document_chunks": len(final_state.document.chunks),
        "linguistic_expressions": len(final_state.linguistic_expressions),
        "enriched_expressions": len(final_state.enriched_expressions),
        "entity_candidates": len(final_state.entity_candidates),
        "event_candidates": len(final_state.event_candidates),
        "attribute_candidates": len(final_state.attribute_candidates),
        "relation_candidates": len(final_state.relation_candidates),
        "candidate_relation_assertions": len(final_state.candidate_relation_assertions),
        "candidate_triples": len(final_state.candidate_triples),
        "concept_candidates": len(final_state.concept_candidates),
        "ontology_relation_candidates": len(final_state.ontology_relation_candidates),
        "axiom_schema_candidates": len(final_state.axiom_schema_candidates),
        "general_axiom_candidates": len(final_state.general_axiom_candidates),
        "completion_candidates": len(final_state.completion_candidates),
        "candidate_triples_preview": [
            to_jsonable(x) for x in final_state.candidate_triples[:20]
        ],
    }


def make_output_row(
    canonical_row: Dict[str, Any],
    final_state: Optional[PipelineState],
    output_format: str,
) -> Dict[str, Any]:
    """Select row format to write."""
    output_format = output_format.strip().lower()

    if output_format == "canonical":
        return canonical_row

    if output_format == "raw":
        if final_state is None:
            return canonical_row

        return {
            "document_id": canonical_row.get("document_id", ""),
            "title": canonical_row.get("title", ""),
            "type": canonical_row.get("type", ""),
            "method": "neoolaf",
            "status": "ok",
            "raw_neoolaf": build_raw_neoolaf_summary(final_state),
        }

    if output_format == "both":
        if final_state is not None:
            canonical_row["raw_neoolaf"] = build_raw_neoolaf_summary(final_state)

        return canonical_row

    raise ValueError("output_format must be one of: raw, canonical, both")


# ============================================================
# Per-document execution
# ============================================================

def run_one_document(
    doc_idx: int,
    doc: Dict[str, Any],
    dataset_jsonl_path: Path,
    ontology_path: Path,
    llm_backend: Any,
    model_name: str,
    base_guidance: UserGuidance,
    few_shot_from_dataset: bool,
    few_shot_k: int,
    few_shot_source_type: str,
    seed_ontology: Any,
    artifacts_root: Path,
    chunk_size: int,
    chunk_overlap: int,
    translate_to_english: bool,
    use_web_search: bool,
    max_chunks: Optional[int],
    max_expressions: Optional[int],
    max_relation_mentions: Optional[int],
    max_workers: int,
    enable_checkpoints: bool,
    save_chunk_checkpoints: bool,
    output_subdir: str,
    base_uri: str,
    verbose: bool,
    debug: bool,
) -> Tuple[Dict[str, Any], PipelineState]:
    """Run NeoOLAF for one JSONL document."""
    document_id = normalize_text_field(doc.get("document_id"))
    title = normalize_text_field(doc.get("title"))
    doc_type = normalize_text_field(doc.get("type"))
    full_text = build_document_text(doc)

    if not full_text:
        raise ValueError(f"Document {document_id} has empty text.")

    # Make per-document guidance copy by JSON round-trip through dataclasses.
    guidance = to_jsonable(base_guidance)
    guidance = parse_user_guidance_json(guidance)

    if few_shot_from_dataset:
        examples = select_dataset_few_shot_examples(
            dataset_jsonl_path=dataset_jsonl_path,
            current_document_id=document_id,
            type_filter=few_shot_source_type,
            few_shot_k=few_shot_k,
            seed=42,
        )
        add_dataset_examples_to_guidance(guidance, examples)

    document = Document(
        doc_id=document_id,
        source_path=str(dataset_jsonl_path),
        raw_text=full_text,
    )

    state = PipelineState(
        document=document,
        llm_model=model_name,
        user_guidance=guidance,
        seed_ontology=seed_ontology,
    )

    # Build sources once per document to avoid state leakage.
    wordnet_source = WordNetSource()
    wikipedia_source = WikipediaSource(language="en")
    wikidata_source = WikidataSource(language="en")
    web_search_source = WebSearchSource()

    pipeline, execution_config = build_neoolaf_pipeline(
        llm_backend=llm_backend,
        model_name=model_name,
        state=state,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        translate_to_english=translate_to_english,
        use_web_search=use_web_search,
        max_chunks=max_chunks,
        max_expressions=max_expressions,
        max_relation_mentions=max_relation_mentions,
        max_workers=max_workers,
        verbose=verbose,
        output_subdir=output_subdir,
        base_uri=base_uri,
        wikipedia_source=wikipedia_source,
        wikidata_source=wikidata_source,
        wordnet_source=wordnet_source,
        web_search_source=web_search_source,
    )

    document_artifacts_root = artifacts_root / safe_filename(document_id)

    runner = Runner(
        pipeline=pipeline,
        runs_root=str(document_artifacts_root),
        verbose=verbose,
        execution_config=execution_config,
        max_workers=max_workers,
        enable_checkpoints=enable_checkpoints,
        save_chunk_checkpoints=save_chunk_checkpoints,
    )

    if debug:
        print("\n" + "=" * 80)
        print(f"[DEBUG] NeoOLAF document #{doc_idx}")
        print(f"[DEBUG] document_id: {document_id}")
        print(f"[DEBUG] title      : {title}")
        print(f"[DEBUG] type       : {doc_type}")
        print(f"[DEBUG] chars      : {len(full_text)}")
        print(f"[DEBUG] artifacts  : {document_artifacts_root}")
        print("=" * 80)

    final_state = runner.run(state)

    canonical_row = canonicalize_neoolaf_state(
        final_state=final_state,
        document_id=document_id,
        title=title,
        doc_type=doc_type,
    )

    return canonical_row, final_state


# ============================================================
# Dataset runner
# ============================================================

def run_neoolaf_dataset(
    dataset_jsonl_path: str | Path,
    ontology_path: str | Path,
    output_jsonl_path: str | Path,
    backend_name: str,
    host: str,
    model_name: str,
    api_key: str = "dummy",
    type_filter: str = "all",
    few_shot_path: str = "",
    user_guidance_path: str = "",
    few_shot_from_dataset: bool = False,
    few_shot_source_type: str = "all",
    few_shot_k: int = 3,
    output_format: str = "canonical",
    artifacts_root: str | Path = "./runs/neoolaf_artifacts",
    chunk_size: int = 5000,
    chunk_overlap: int = 0,
    translate_to_english: bool = False,
    use_web_search: bool = False,
    max_chunks: Optional[int] = None,
    max_expressions: Optional[int] = None,
    max_relation_mentions: Optional[int] = None,
    max_workers: int = 14,
    enable_checkpoints: bool = True,
    save_chunk_checkpoints: bool = True,
    resume: bool = True,
    timeout: int = 900,
    max_retries: int = 5,
    retry_wait_seconds: float = 5.0,
    referer: str = "http://localhost",
    title: str = "NeoOLAF-Benchmark",
    output_subdir: str = "data/exports",
    base_uri: str = "http://neoolaf.org/resource/",
    verbose: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run NeoOLAF line by line on a JSONL dataset."""
    dataset_jsonl_path = Path(dataset_jsonl_path)
    ontology_path = Path(ontology_path)
    output_jsonl_path = Path(output_jsonl_path)
    artifacts_root = Path(artifacts_root)

    if not dataset_jsonl_path.exists():
        raise FileNotFoundError(f"Dataset JSONL not found: {dataset_jsonl_path}")

    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

    output_format = output_format.strip().lower()

    if output_format not in {"raw", "canonical", "both"}:
        raise ValueError("output_format must be one of: raw, canonical, both")

    ensure_parent_dir(output_jsonl_path)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    processed_ids: Set[str] = set()

    if resume:
        processed_ids = load_processed_ids(output_jsonl_path)

    llm_backend = build_neoolaf_backend(
        backend_name=backend_name,
        host=host,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        retry_wait_seconds=retry_wait_seconds,
        referer=referer,
        title=title,
    )

    base_guidance = load_user_guidance(user_guidance_path)
    base_guidance = enrich_guidance_with_few_shots(
        guidance=base_guidance,
        few_shot_path=few_shot_path,
        few_shot_k=few_shot_k,
    )

    seed_ontology = build_seed_ontology(str(ontology_path))

    total_docs = count_documents(dataset_jsonl_path, type_filter=type_filter)
    docs_iter = iter_documents(dataset_jsonl_path, type_filter=type_filter)

    start_time = dt.datetime.now()
    seen = 0
    done = 0
    skipped_resume = 0
    failed = 0

    for row_idx, doc in enumerate(
        tqdm(docs_iter, total=total_docs, desc="NeoOLAF docs", unit="doc"),
        start=1,
    ):
        seen += 1
        document_id = safe_document_id(doc, row_idx)

        if document_id in processed_ids:
            skipped_resume += 1
            continue

        try:
            doc = dict(doc)
            doc["document_id"] = document_id

            canonical_row, final_state = run_one_document(
                doc_idx=row_idx,
                doc=doc,
                dataset_jsonl_path=dataset_jsonl_path,
                ontology_path=ontology_path,
                llm_backend=llm_backend,
                model_name=model_name,
                base_guidance=base_guidance,
                few_shot_from_dataset=few_shot_from_dataset,
                few_shot_k=few_shot_k,
                few_shot_source_type=few_shot_source_type,
                seed_ontology=seed_ontology,
                artifacts_root=artifacts_root,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                translate_to_english=translate_to_english,
                use_web_search=use_web_search,
                max_chunks=max_chunks,
                max_expressions=max_expressions,
                max_relation_mentions=max_relation_mentions,
                max_workers=max_workers,
                enable_checkpoints=enable_checkpoints,
                save_chunk_checkpoints=save_chunk_checkpoints,
                output_subdir=output_subdir,
                base_uri=base_uri,
                verbose=verbose,
                debug=debug,
            )

            output_row = make_output_row(
                canonical_row=canonical_row,
                final_state=final_state,
                output_format=output_format,
            )

            append_jsonl(output_jsonl_path, output_row)
            done += 1

        except KeyboardInterrupt:
            raise

        except Exception as e:
            failed += 1

            error_row = {
                "document_id": document_id,
                "title": normalize_text_field(doc.get("title")),
                "type": normalize_text_field(doc.get("type")),
                "method": "neoolaf",
                "parsed_ok": False,
                "prediction": {
                    "entities": [],
                    "relations": [],
                },
                "raw_counts": {
                    "canonical_entities": 0,
                    "canonical_relations": 0,
                },
                "error": str(e),
            }

            if output_format == "both":
                error_row["traceback"] = traceback.format_exc()
            elif output_format == "raw":
                error_row = {
                    "document_id": document_id,
                    "title": normalize_text_field(doc.get("title")),
                    "type": normalize_text_field(doc.get("type")),
                    "method": "neoolaf",
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc() if debug else "",
                }

            append_jsonl(output_jsonl_path, error_row)

            if debug:
                print(f"\n[DEBUG] ERROR on document {document_id}: {e}")
                traceback.print_exc()

    end_time = dt.datetime.now()

    return {
        "dataset_jsonl_path": str(dataset_jsonl_path),
        "ontology_path": str(ontology_path),
        "output_jsonl_path": str(output_jsonl_path),
        "backend_name": backend_name,
        "host": host,
        "model_name": model_name,
        "type_filter": type_filter,
        "few_shot_path": few_shot_path,
        "user_guidance_path": user_guidance_path,
        "few_shot_from_dataset": few_shot_from_dataset,
        "few_shot_source_type": few_shot_source_type,
        "few_shot_k": few_shot_k,
        "output_format": output_format,
        "artifacts_root": str(artifacts_root),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "translate_to_english": translate_to_english,
        "use_web_search": use_web_search,
        "max_chunks": max_chunks,
        "max_expressions": max_expressions,
        "max_relation_mentions": max_relation_mentions,
        "max_workers": max_workers,
        "enable_checkpoints": enable_checkpoints,
        "save_chunk_checkpoints": save_chunk_checkpoints,
        "resume": resume,
        "seen": seen,
        "done": done,
        "skipped_resume": skipped_resume,
        "failed": failed,
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "elapsed_seconds": (end_time - start_time).total_seconds(),
    }


# ============================================================
# CLI
# ============================================================

def parse_optional_int(value: str) -> Optional[int]:
    """Parse optional integer CLI values."""
    value = str(value).strip()

    if value.lower() in {"", "none", "null"}:
        return None

    return int(value)


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run NeoOLAF on a JSONL dataset line-by-line."
    )

    parser.add_argument("--dataset-jsonl-path", required=True, help="Path to input JSONL dataset.")
    parser.add_argument("--ontology-path", required=True, help="Path to ontology file.")
    parser.add_argument("--output-jsonl-path", required=True, help="Path to output JSONL predictions.")

    parser.add_argument(
        "--backend-name",
        required=True,
        choices=["vllm", "openrouter", "ollama"],
        help="Backend type.",
    )
    parser.add_argument("--host", required=True, help="Backend host URL.")
    parser.add_argument("--api-key", default="dummy", help="API key for OpenAI-compatible backends.")
    parser.add_argument("--model-name", required=True, help="Model name exposed by backend.")

    parser.add_argument("--type-filter", default="all", help='Filter by "type". Use "all" for no filter.')

    parser.add_argument(
        "--few-shot-path",
        default="",
        help="Optional JSON file containing few-shot examples.",
    )
    parser.add_argument(
        "--user-guidance-path",
        default="",
        help="Optional JSON file containing UserGuidance fields.",
    )
    parser.add_argument(
        "--few-shot-from-dataset",
        action="store_true",
        help="Sample few-shot relation examples directly from the gold JSONL dataset.",
    )
    parser.add_argument(
        "--few-shot-source-type",
        default="all",
        help='Type used to sample dataset few-shots. Use "all" for no filter.',
    )
    parser.add_argument("--few-shot-k", type=int, default=3, help="Number of few-shot examples.")

    parser.add_argument(
        "--output-format",
        default="canonical",
        choices=["raw", "canonical", "both"],
        help=(
            "Output JSONL format. "
            "'canonical' writes eval_relations.py-compatible predictions; "
            "'raw' writes compact NeoOLAF debug summaries; "
            "'both' writes canonical predictions plus compact raw debug summaries."
        ),
    )

    parser.add_argument(
        "--artifacts-root",
        default="./runs/neoolaf_artifacts",
        help="Directory where NeoOLAF internal run artifacts/checkpoints are stored.",
    )

    parser.add_argument("--chunk-size", type=parse_optional_int, default=5000)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--max-chunks", type=parse_optional_int, default=None)
    parser.add_argument("--max-expressions", type=parse_optional_int, default=None)
    parser.add_argument("--max-relation-mentions", type=parse_optional_int, default=None)
    parser.add_argument("--max-workers", type=int, default=14)

    parser.add_argument(
        "--translate-to-english",
        action="store_true",
        help="Enable DeepTranslator translation to English during preprocessing.",
    )
    parser.add_argument(
        "--use-web-search",
        action="store_true",
        help="Enable web search source in NeoOLAF enrichment/RAG.",
    )

    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-wait-seconds", type=float, default=5.0)

    parser.add_argument("--referer", default="http://localhost")
    parser.add_argument("--title", default="NeoOLAF-Benchmark")

    parser.add_argument("--output-subdir", default="data/exports")
    parser.add_argument("--base-uri", default="http://neoolaf.org/resource/")

    parser.add_argument("--no-resume", action="store_true", help="Disable resume mode.")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable NeoOLAF checkpoints.")
    parser.add_argument("--no-chunk-checkpoints", action="store_true", help="Disable chunk checkpoints.")

    parser.add_argument("--verbose", action="store_true", help="Enable NeoOLAF verbose mode.")
    parser.add_argument("--debug", action="store_true", help="Print debug traces on failures.")

    parser.add_argument(
        "--env-path",
        default="",
        help="Optional .env path. If omitted, PROJECT_ROOT/.env is loaded when available.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_argparser()
    args = parser.parse_args()

    env_path = Path(args.env_path) if args.env_path else PROJECT_ROOT / ".env"
    load_dotenv_file(env_path)

    # If the user passes an empty API key but env vars exist, use them.
    api_key = args.api_key
    if not api_key or api_key == "dummy":
        if args.backend_name == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", api_key)
        else:
            api_key = os.getenv("OPENAI_API_KEY", api_key)

    summary = run_neoolaf_dataset(
        dataset_jsonl_path=args.dataset_jsonl_path,
        ontology_path=args.ontology_path,
        output_jsonl_path=args.output_jsonl_path,
        backend_name=args.backend_name,
        host=args.host,
        model_name=args.model_name,
        api_key=api_key,
        type_filter=args.type_filter,
        few_shot_path=args.few_shot_path,
        user_guidance_path=args.user_guidance_path,
        few_shot_from_dataset=args.few_shot_from_dataset,
        few_shot_source_type=args.few_shot_source_type,
        few_shot_k=args.few_shot_k,
        output_format=args.output_format,
        artifacts_root=args.artifacts_root,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        translate_to_english=args.translate_to_english,
        use_web_search=args.use_web_search,
        max_chunks=args.max_chunks,
        max_expressions=args.max_expressions,
        max_relation_mentions=args.max_relation_mentions,
        max_workers=args.max_workers,
        enable_checkpoints=not args.no_checkpoints,
        save_chunk_checkpoints=not args.no_chunk_checkpoints,
        resume=not args.no_resume,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_wait_seconds=args.retry_wait_seconds,
        referer=args.referer,
        title=args.title,
        output_subdir=args.output_subdir,
        base_uri=args.base_uri,
        verbose=args.verbose,
        debug=args.debug,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()