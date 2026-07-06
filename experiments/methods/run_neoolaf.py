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
<<<<<<< HEAD
            text = normalize_text_field(item)
            if text:
                snippets.append(text)
=======
            relations.append(
                {
                    "head": str(raw_head).strip(),
                    "relation": str(raw_relation).strip(),
                    "tail": str(raw_tail).strip(),
                    "evidence": evidence,
                }
            )

    prediction: Dict[str, Any] = {"entities": list(entities_by_label.values()), "relations": relations}
    if constrained:
        prediction["projection_diagnostics"] = {
            "constrained": True,
            "allowed_relation_count": len(allowed_relations),
            "source_entity_count": len(source_entities),
            "accepted_relations": len(relations),
            "rejected_triples": len(rejected),
            "rejected_triples_preview": rejected[:20],
        }
    return prediction


# ---------------------------------------------------------------------------
# Direct DocRED constrained extraction helper
# ---------------------------------------------------------------------------

def compact_source_entities_for_prompt(source_entities: List[Dict[str, Any]], max_entities: Optional[int] = None) -> str:
    """Render source entity clusters as a compact prompt table."""
    rows: List[str] = []
    entities = source_entities[: max_entities or len(source_entities)]
    for ent in entities:
        aliases = ", ".join(str(a) for a in (ent.get("aliases") or [])[:8] if str(a).strip())
        rows.append(f"- {ent.get('id')} | {ent.get('type')} | {ent.get('label')} | aliases: {aliases}")
    return "\n".join(rows)


def compact_allowed_relations_for_prompt(allowed_relations: List[Dict[str, Any]], max_relations: Optional[int] = None) -> str:
    """Render allowed relation vocabulary as a compact prompt table."""
    rows: List[str] = []
    rels = allowed_relations[: max_relations or len(allowed_relations)]
    for rel in rels:
        rel_id = rel.get("id") or ""
        label = rel.get("label") or ""
        canonical = rel.get("canonical") or label or rel_id
        rows.append(f"- {rel_id} | {canonical}")
    return "\n".join(rows)


DOCRED_RELATION_FAMILIES: Dict[str, set[str]] = {
    "location": {"P17", "P131", "P150", "P30", "P19", "P159", "P276", "P495"},
    "organization": {"P127", "P361", "P749", "P355", "P159", "P571", "P112", "P108"},
    "person": {"P19", "P27", "P69", "P108", "P463", "P569", "P570"},
    "creative_work": {"P175", "P170", "P162", "P264", "P577", "P155", "P495"},
    "date_numeric": {"P569", "P570", "P571", "P577"},
}

DOCRED_HIGH_YIELD_RELATION_IDS: set[str] = {
    "P17", "P27", "P69", "P159", "P127", "P361", "P175", "P264", "P577",
    "P19", "P569", "P570", "P749", "P355", "P571", "P30", "P495", "P108",
}

COUNTRY_WORDS: set[str] = {
    "greece", "greek", "united states", "american", "brazil", "brazilian", "canada", "canadian",
    "france", "french", "england", "british", "united kingdom", "uk", "ireland", "irish",
    "japan", "japanese", "china", "chinese", "germany", "german", "italy", "italian",
    "spain", "spanish", "mexico", "mexican", "australia", "australian",
}


def relation_specs_by_id(allowed_relations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(rel.get("id")): rel for rel in allowed_relations or [] if rel.get("id")}


def relation_spec_by_id(allowed_relations: List[Dict[str, Any]], relation_id: str) -> Optional[Dict[str, Any]]:
    return relation_specs_by_id(allowed_relations).get(str(relation_id))


def entity_text(ent: Dict[str, Any]) -> str:
    return " ".join([str(ent.get("label") or ""), *[str(a) for a in (ent.get("aliases") or [])]])


def entity_is_country_like(ent: Dict[str, Any]) -> bool:
    text = normalize_key(entity_text(ent))
    return any(country in text for country in COUNTRY_WORDS)


def entity_is_probably_creative_work(ent: Dict[str, Any]) -> bool:
    text = normalize_key(entity_text(ent))
    typ = normalize_key(ent.get("type"))
    return typ == "misc" or any(w in text for w in ["song", "album", "single", "film", "episode", "book"])


def infer_docred_relation_family_ids(record: Dict[str, Any], source_entities: List[Dict[str, Any]]) -> set[str]:
    """Infer a gold-free relation subset from entity types and document trigger words."""
    text = normalize_key(document_text_from_record(record))
    title = normalize_key(title_from_record(record, document_id_from_record(record, 0)))
    types = {normalize_key(ent.get("type")) for ent in source_entities}
    labels = normalize_key(" ".join(entity_text(ent) for ent in source_entities))
    ids: set[str] = set(DOCRED_HIGH_YIELD_RELATION_IDS)

    if {"loc", "gpe", "location"} & types or any(w in text for w in ["city", "county", "state", "country", "province", "located", "born in"]):
        ids |= DOCRED_RELATION_FAMILIES["location"]
    if {"org", "organization"} & types or any(w in text for w in ["company", "group", "organization", "research", "headquartered", "based in", "subsidiary", "parent", "owned", "founded"]):
        ids |= DOCRED_RELATION_FAMILIES["organization"]
    if {"per", "person"} & types or any(w in text for w in ["born", "died", "educated", "graduated", "employer", "citizen", "nationality"]):
        ids |= DOCRED_RELATION_FAMILIES["person"]
    if any(w in (text + " " + title + " " + labels) for w in ["song", "single", "album", "rapper", "singer", "record label", "released", "publication", "producer", "performed", "film"]):
        ids |= DOCRED_RELATION_FAMILIES["creative_work"]
    if re.search(r"\b(?:18|19|20)\d{2}\b", text):
        ids |= DOCRED_RELATION_FAMILIES["date_numeric"]
    return ids


def filter_allowed_relations_for_direct_extractor(
    allowed_relations: List[Dict[str, Any]],
    *,
    focus_relation_ids: Optional[str] = None,
    record: Optional[Dict[str, Any]] = None,
    relation_family_filter: bool = False,
) -> List[Dict[str, Any]]:
    """Restrict the relation vocabulary shown to the direct DocRED extractor.

    This is gold-free. It uses either an explicit user-provided ID list or an
    automatic subset inferred from entity types and document trigger words.
    """
    if focus_relation_ids:
        wanted = {x.strip() for x in str(focus_relation_ids).split(",") if x.strip()}
        if wanted:
            filtered = [rel for rel in allowed_relations if str(rel.get("id") or rel.get("canonical") or "") in wanted or str(rel.get("id") or "") in wanted]
            return filtered or allowed_relations

    if relation_family_filter and record is not None:
        wanted = infer_docred_relation_family_ids(record, source_entities_from_record(record))
        filtered = [rel for rel in allowed_relations if str(rel.get("id") or "") in wanted]
        return filtered or allowed_relations

    return allowed_relations


def docred_relation_disambiguation_hints() -> str:
    """Gold-free relation-selection hints for common DocRED confusions."""
    return """
Relation-selection hints for DocRED/Wikidata labels:
- Location family:
  * Use P159 : headquarters location for organizations/broadcasters "based in" or "headquartered in" a city.
  * Use P17 : country when the tail is a country entity such as Greece/United States/Brazil.
  * Use P27 : country of citizenship only for a person.
  * Use P131 : located in the administrative territorial entity only for city/county/state containment, not for country facts.
  * Use P495 : country of origin for creative works/products when the document states origin/nationality of the work.
- Organization family:
  * Use P127 : owned by for corporate ownership/control or "part of [media/corporate group]".
  * Use P361 : part of only for explicit generic part-whole membership, not corporate ownership.
  * Use P749 : parent organization and P355 : subsidiary only for explicit parent/subsidiary relations.
  * Use P571 : inception for founding/start dates.
- Person family:
  * Use P19 : place of birth for "born in" places; P569/P570 for birth/death dates.
  * Use P69 : educated at for schools/universities attended.
  * Use P108 : employer only when employment/working for is explicit.
- Creative-work family:
  * For songs, "song by [artist]" means P175 : performer, not P170 : creator.
  * Use P264 : record label for labels/record companies; P577 : publication date for release dates.
  * Use P162 : producer only when the text explicitly says produced by/producer.
  * Avoid P155 : follows unless the text explicitly states predecessor/follows.
- Weak/peripheral relations to avoid unless explicit and central: P400 platform, P1344 participant of, P155 follows, P162 producer, P463 member of, P112 founded by.
- Output fewer high-confidence relations rather than many plausible peripheral ones.
""".strip()


def docred_type_constraint_violation(rel: Dict[str, Any]) -> Optional[str]:
    """Return a rejection reason if a relation violates common DocRED type constraints."""
    rid = str(rel.get("relation_id") or "")
    head_type = normalize_key(rel.get("head_type"))
    tail_type = normalize_key(rel.get("tail_type"))
    evidence = normalize_key(rel.get("evidence"))
    head_text = normalize_key(str(rel.get("head") or "") + " " + str((rel.get("raw_prediction") or {}).get("head") or ""))
    tail_text = normalize_key(str(rel.get("tail") or "") + " " + str((rel.get("raw_prediction") or {}).get("tail") or ""))

    if rid == "P27" and head_type not in {"per", "person"}:
        return "P27_requires_person_head"
    if rid in {"P569", "P570"} and head_type not in {"per", "person"}:
        return f"{rid}_requires_person_head"
    if rid == "P175" and head_type in {"per", "person"}:
        return "P175_requires_work_head_not_person"
    if rid == "P162" and not any(w in evidence for w in ["producer", "produced by", "production"]):
        return "P162_requires_explicit_producer_evidence"
    if rid == "P155" and not any(w in evidence for w in ["follows", "preceded", "predecessor", "sequel"]):
        return "P155_requires_explicit_sequence_evidence"
    if rid == "P400" and not any(w in evidence for w in ["platform", "operating system", "software", "game platform"]):
        return "P400_requires_explicit_platform_evidence"
    if rid == "P131" and any(c in tail_text for c in COUNTRY_WORDS) and not any(w in evidence for w in ["administrative", "county", "state", "province", "municipality", "territorial"]):
        return "P131_country_tail_without_admin_evidence"
    if rid == "P463" and "member" not in evidence:
        return "P463_requires_member_evidence"
    if rid == "P112" and not any(w in evidence for w in ["founded", "founder", "founded by"]):
        return "P112_requires_founder_evidence"
    return None


def calibrate_one_docred_relation(
    rel: Dict[str, Any],
    *,
    allowed_relations: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Deterministically relabel/reject common DocRED relation confusions.

    Returns (new_relation_or_none, diagnostic_or_none). A None relation means
    reject the predicted triple from the benchmark-facing output.
    """
    rid = str(rel.get("relation_id") or "")
    head_type = normalize_key(rel.get("head_type"))
    tail_type = normalize_key(rel.get("tail_type"))
    evidence = normalize_key(rel.get("evidence"))
    head = normalize_key(rel.get("head"))
    tail = normalize_key(rel.get("tail"))
    by_id = relation_specs_by_id(allowed_relations)

    def relabel(new_id: str, reason: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        new_rel = copy.deepcopy(rel)
        spec = by_id.get(new_id)
        if spec:
            new_rel["relation_id"] = spec.get("id")
            new_rel["relation"] = spec.get("canonical") or spec.get("label")
            new_rel["relation_label"] = spec.get("label")
        diag = {"action": "relabel", "from": rid, "to": new_id, "reason": reason, "raw": rel}
        new_rel.setdefault("calibration", []).append(diag)
        return new_rel, diag

    def reject(reason: str) -> Tuple[None, Dict[str, Any]]:
        return None, {"action": "reject", "relation_id": rid, "reason": reason, "raw": rel}

    # Common confusions seen in the DocRED smoke tests.
    if rid == "P276" and head_type in {"org", "organization"} and any(w in evidence for w in ["based in", "headquartered", "headquarters"]):
        return relabel("P159", "organization_location_should_be_headquarters_location")
    if rid == "P361" and head_type in {"org", "organization"} and any(w in evidence for w in ["part of", "belongs to", "owned by", "group"]):
        return relabel("P127", "corporate_group_part_of_preferred_as_owned_by")
    if rid == "P170" and head_type in {"misc", "work", "entity"} and any(w in evidence for w in ["song by", "single by", "performed by", "rapper", "singer"]):
        return relabel("P175", "song_by_artist_should_be_performer")
    if rid == "P131" and any(c in tail for c in COUNTRY_WORDS):
        if head_type in {"per", "person"}:
            return relabel("P27", "person_country_relation_should_be_citizenship")
        if head_type in {"org", "organization", "loc", "location"}:
            return relabel("P17", "country_tail_should_use_country_relation")
        return reject("administrative_location_with_country_tail_rejected")
    if rid == "P400" and head_type in {"org", "organization"} and tail_type in {"org", "organization"}:
        return reject("platform_between_organizations_is_likely_subscription_service_false_positive")
    if rid == "P1344":
        return reject("participant_of_is_weak_peripheral_for_docred_smoke")

    violation = docred_type_constraint_violation(rel)
    if violation:
        return reject(violation)
    return rel, None


def calibrate_docred_relations(
    relations: List[Dict[str, Any]],
    *,
    allowed_relations: List[Dict[str, Any]],
    enable_calibration: bool = True,
    enable_strict_type_constraints: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply deterministic calibration/rejection and deduplicate triples."""
    if not enable_calibration and not enable_strict_type_constraints:
        return relations, {"enabled": False, "relabelled": 0, "rejected": 0, "diagnostics": []}

    calibrated: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for rel in relations:
        candidate = copy.deepcopy(rel)
        diag: Optional[Dict[str, Any]] = None
        if enable_calibration:
            candidate, diag = calibrate_one_docred_relation(candidate, allowed_relations=allowed_relations)
            if diag:
                diagnostics.append(diag)
        if candidate is not None and enable_strict_type_constraints:
            violation = docred_type_constraint_violation(candidate)
            if violation:
                diagnostics.append({"action": "reject", "relation_id": candidate.get("relation_id"), "reason": violation, "raw": candidate})
                candidate = None
        if candidate is None:
            continue
        key = (str(candidate.get("head_id") or candidate.get("head")), str(candidate.get("relation_id") or candidate.get("relation")), str(candidate.get("tail_id") or candidate.get("tail")))
        if key in seen:
            continue
        seen.add(key)
        calibrated.append(candidate)

    return calibrated, {
        "enabled": True,
        "input_relations": len(relations),
        "output_relations": len(calibrated),
        "relabelled": sum(1 for d in diagnostics if d.get("action") == "relabel"),
        "rejected": sum(1 for d in diagnostics if d.get("action") == "reject"),
        "diagnostics": diagnostics[:50],
    }


def build_docred_direct_extraction_messages(
    record: Dict[str, Any],
    allowed_relations: List[Dict[str, Any]],
    *,
    max_entities: Optional[int] = None,
    max_relations: Optional[int] = None,
    high_precision_hints: bool = True,
) -> List[Dict[str, str]]:
    """Build a direct constrained DocRED extraction prompt.

    This prompt intentionally exposes only the source entity clusters and the
    global allowed relation vocabulary. It does not expose gold pairs for the
    current document.
    """
    doc_id = document_id_from_record(record, 0)
    title = title_from_record(record, doc_id)
    text = document_text_from_record(record)
    source_entities = source_entities_from_record(record)
    hints = docred_relation_disambiguation_hints() if high_precision_hints else ""

    system = (
        "You are a strict DocRED relation extraction system. "
        "Extract document-level relations only between the provided source entity IDs. "
        "Use only the provided DocRED relation vocabulary. "
        "Do not invent entities. Do not invent predicates. "
        "Do not use entity types such as ORG, LOC, PER, MISC, person, location, institution, or entity as nodes. "
        "Return JSON only."
    )

    user = f"""
Task: extract all relations supported by the document.

Rules:
1. Heads and tails must be entity IDs from SOURCE ENTITIES.
2. Relations must be relation IDs from ALLOWED RELATIONS.
3. Use aliases only to recognize mentions, but output entity IDs.
4. Evidence must be a short quote or paraphrase from the document.
5. Do not output a relation if the document does not support it.
6. Do not use outside knowledge.
7. If no relation is supported, return {{"relations": []}}.
8. Prefer high precision: do not output peripheral or merely plausible relations.
9. Choose the most specific DocRED/Wikidata relation, not a generic neighbor.

RELATION DISAMBIGUATION HINTS:
{hints}

Output JSON schema:
{{
  "relations": [
    {{
      "head_id": "Event_...",
      "relation_id": "P...",
      "tail_id": "Event_...",
      "evidence": "short evidence from the document"
    }}
  ]
}}

DOCUMENT ID: {doc_id}
TITLE: {title}

SOURCE ENTITIES:
{compact_source_entities_for_prompt(source_entities, max_entities=max_entities)}

ALLOWED RELATIONS:
{compact_allowed_relations_for_prompt(allowed_relations, max_relations=max_relations)}

DOCUMENT TEXT:
{text}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def normalize_relation_item(item: Any) -> Optional[Dict[str, Any]]:
    """Normalize one relation-like object into a dictionary.

    The direct DocRED extractor is instructed to return dictionaries, but small
    open models sometimes return compact arrays such as:
        [head_id, relation_id, tail_id]
        [head_id, relation_id, tail_id, evidence]
    This helper keeps the runner robust and prevents errors such as
    AttributeError: 'list' object has no attribute 'get'.
    """
    if isinstance(item, dict):
        return item
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return {
            "head_id": item[0],
            "relation_id": item[1],
            "tail_id": item[2],
            "evidence": item[3] if len(item) >= 4 else "",
            "raw_sequence_prediction": list(item),
        }
    return None


def _relation_items_from_model_json(data: Any) -> List[Dict[str, Any]]:
    """Normalize the direct extractor JSON payload into relation dictionaries."""
    if isinstance(data, dict):
        candidates = data.get("relations") or data.get("triples") or data.get("predictions") or []
    elif isinstance(data, list):
        candidates = data
    else:
        candidates = []
    normalized: List[Dict[str, Any]] = []
    for item in candidates or []:
        norm = normalize_relation_item(item)
        if norm is not None:
            normalized.append(norm)
    return normalized


def canonical_entities_from_source(source_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return benchmark-facing source entities without leaking gold relations."""
    return [
        {
            "id": ent.get("id"),
            "label": ent.get("label"),
            "type": ent.get("type"),
            "aliases": ent.get("aliases") or [],
            "source": "source_document_entity",
        }
        for ent in source_entities
    ]


def validate_direct_docred_relations(
    relation_items: List[Dict[str, Any]],
    *,
    source_entities: List[Dict[str, Any]],
    allowed_relations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate direct DocRED predictions against source entities and allowed relations."""
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    for raw_item in relation_items:
        item = normalize_relation_item(raw_item)
        if item is None:
            rejected.append({"raw_prediction": raw_item, "reasons": ["invalid_relation_item_type"]})
            continue
        raw_head = (
            item.get("head_id")
            or item.get("head")
            or item.get("subject_id")
            or item.get("subject")
            or item.get("source_id")
            or item.get("source")
            or item.get("h")
        )
        raw_tail = (
            item.get("tail_id")
            or item.get("tail")
            or item.get("object_id")
            or item.get("object")
            or item.get("target_id")
            or item.get("target")
            or item.get("t")
        )
        raw_relation = (
            item.get("relation_id")
            or item.get("relation")
            or item.get("predicate_id")
            or item.get("predicate")
            or item.get("label")
            or item.get("r")
        )
        evidence = item.get("evidence") or item.get("justification") or item.get("sentence") or ""
        if isinstance(evidence, (list, tuple)):
            evidence = " | ".join(str(x) for x in evidence)
        elif isinstance(evidence, dict):
            evidence = evidence.get("text") or evidence.get("snippet") or json.dumps(evidence, ensure_ascii=False)

        head_ent = map_label_to_source_entity(raw_head, source_entities)
        tail_ent = map_label_to_source_entity(raw_tail, source_entities)
        rel_spec = map_relation_to_allowed(raw_relation, allowed_relations)

        reasons: List[str] = []
        if head_ent is None:
            reasons.append("head_not_source_entity")
        if tail_ent is None:
            reasons.append("tail_not_source_entity")
        if rel_spec is None:
            reasons.append("relation_not_allowed")
        if head_ent is not None and tail_ent is not None and head_ent.get("id") == tail_ent.get("id"):
            reasons.append("self_relation_rejected")

        if reasons:
            rejected.append(
                {
                    "raw_prediction": item,
                    "reasons": reasons,
                    "head": raw_head,
                    "relation": raw_relation,
                    "tail": raw_tail,
                }
            )
>>>>>>> d746806 (docred - relation and openrouter patch)
            continue

        for key in ["text", "snippet", "sentence", "content"]:
            value = normalize_text_field(item_dict.get(key))
            if value:
                snippets.append(value)
                break

    return " | ".join(snippets[:3])


<<<<<<< HEAD
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
=======
def call_docred_direct_json_with_retries(
    *,
    backend: OpenAICompatibleBackend,
    args: argparse.Namespace,
    model_name: str,
    messages: List[Dict[str, str]],
    fallback_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Call OpenRouter/OpenAI-compatible backend with retries for empty content."""
    direct_retries = max(1, int(getattr(args, "docred_direct_retries", 3) or 3))
    direct_temperature = float(getattr(args, "docred_direct_temperature", 0.0) or 0.0)
    last_error: Optional[BaseException] = None
    for attempt in range(1, direct_retries + 1):
        attempt_messages = messages
        if attempt > 1 and fallback_messages is not None:
            attempt_messages = fallback_messages
        try:
            return backend.chat(model_name, attempt_messages, temperature=direct_temperature)
        except Exception as exc:
            last_error = exc
            if attempt < direct_retries:
                time.sleep(float(getattr(args, "docred_direct_retry_sleep", 2.0) or 2.0))
    raise RuntimeError(
        "DocRED direct constrained extraction failed after "
        f"{direct_retries} attempt(s): {type(last_error).__name__}: {last_error}"
    )


def make_prediction_from_direct_items(
    *,
    record: Dict[str, Any],
    relation_items: List[Dict[str, Any]],
    allowed_relations: List[Dict[str, Any]],
    args: argparse.Namespace,
    mode_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate/calibrate direct relation items and return prediction + diagnostics."""
    source_entities = source_entities_from_record(record)
    accepted, rejected = validate_direct_docred_relations(
        relation_items,
        source_entities=source_entities,
        allowed_relations=allowed_relations,
    )
    calibrated, calibration_diagnostics = calibrate_docred_relations(
        accepted,
        allowed_relations=allowed_relations,
        enable_calibration=bool(getattr(args, "docred_calibrate_relations", False)),
        enable_strict_type_constraints=bool(getattr(args, "docred_strict_type_constraints", False)),
    )
    diagnostics = {
        "mode": mode_name,
        "source_entity_count": len(source_entities),
        "allowed_relation_count": len(allowed_relations),
        "focus_relation_ids": getattr(args, "docred_direct_focus_relation_ids", None),
        "relation_family_filter": bool(getattr(args, "docred_relation_family_filter", False)),
        "high_precision_hints": not bool(getattr(args, "docred_direct_disable_hints", False)),
        "raw_relation_items": len(relation_items),
        "accepted_before_calibration": len(accepted),
        "accepted_relations": len(calibrated),
        "rejected_relations": len(rejected),
        "rejected_preview": rejected[:20],
        "calibration": calibration_diagnostics,
    }
    prediction = {
        "entities": canonical_entities_from_source(source_entities),
        "relations": calibrated,
        "projection_diagnostics": diagnostics,
    }
    return prediction, diagnostics


def build_docred_probe_messages(
    record: Dict[str, Any],
    allowed_relations: List[Dict[str, Any]],
    *,
    family_name: str,
    max_entities: Optional[int] = None,
    max_relations: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Build a shorter targeted prompt for zero-relation recovery probes."""
    doc_id = document_id_from_record(record, 0)
    title = title_from_record(record, doc_id)
    text = document_text_from_record(record)
    source_entities = source_entities_from_record(record)
    system = (
        "You are a strict DocRED relation extraction verifier. Return JSON only. "
        "Use only provided source entity IDs and allowed relation IDs."
    )
    user = f"""
Targeted DocRED probe: {family_name}

Find high-confidence relations from this family only. If none are explicit, return {{"relations": []}}.
Output JSON only: {{"relations": [{{"head_id": "Event_...", "relation_id": "P...", "tail_id": "Event_...", "evidence": "..."}}]}}

SOURCE ENTITIES:
{compact_source_entities_for_prompt(source_entities, max_entities=max_entities)}

ALLOWED RELATIONS:
{compact_allowed_relations_for_prompt(allowed_relations, max_relations=max_relations)}

TITLE: {title}
DOCUMENT TEXT:
{text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def zero_relation_probe_relation_subsets(
    record: Dict[str, Any],
    allowed_relations: List[Dict[str, Any]],
    max_families: int = 3,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Choose a few relation-family subsets for zero-relation recovery."""
    relevant_ids = infer_docred_relation_family_ids(record, source_entities_from_record(record))
    by_id = relation_specs_by_id(allowed_relations)
    scored: List[Tuple[int, str, set[str]]] = []
    text = normalize_key(document_text_from_record(record) + " " + title_from_record(record, document_id_from_record(record, 0)))
    triggers = {
        "creative_work": ["song", "single", "album", "rapper", "record label", "released", "producer"],
        "person": ["born", "died", "educated", "graduated", "employer", "citizen"],
        "organization": ["company", "group", "research", "subsidiary", "parent", "headquartered", "based in", "founded"],
        "location": ["city", "county", "state", "country", "located", "province"],
    }
    for family, ids in DOCRED_RELATION_FAMILIES.items():
        if not (ids & relevant_ids):
            continue
        score = sum(1 for trig in triggers.get(family, []) if trig in text)
        scored.append((score, family, ids & relevant_ids))
    scored.sort(reverse=True)
    subsets: List[Tuple[str, List[Dict[str, Any]]]] = []
    for _, family, ids in scored[:max_families]:
        rels = [by_id[rid] for rid in sorted(ids) if rid in by_id]
        if rels:
            subsets.append((family, rels))
    return subsets


def run_docred_direct_constrained_extraction(
    *,
    record: Dict[str, Any],
    backend: OpenAICompatibleBackend,
    args: argparse.Namespace,
    artifact_dir: str,
) -> Dict[str, Any]:
    """Run direct DocRED-constrained extraction and optional calibration/probes.

    This is a benchmark adapter only. It does not change NeoOLAF native layer
    outputs, KG files, or generated ontology.
    """
    source_entities = source_entities_from_record(record)
    base_allowed_relations = list(getattr(args, "allowed_relation_specs", []) or [])
    allowed_relations = filter_allowed_relations_for_direct_extractor(
        base_allowed_relations,
        focus_relation_ids=getattr(args, "docred_direct_focus_relation_ids", None),
        record=record,
        relation_family_filter=bool(getattr(args, "docred_relation_family_filter", False)),
    )
    messages = build_docred_direct_extraction_messages(
        record,
        allowed_relations,
        max_entities=getattr(args, "docred_direct_max_entities", None),
        max_relations=getattr(args, "docred_direct_max_relations", None),
        high_precision_hints=not bool(getattr(args, "docred_direct_disable_hints", False)),
    )
    fallback_messages = build_docred_direct_extraction_messages(
        record,
        allowed_relations,
        max_entities=getattr(args, "docred_direct_max_entities", None),
        max_relations=getattr(args, "docred_direct_max_relations", None),
        high_precision_hints=False,
    )
    raw_response = call_docred_direct_json_with_retries(
        backend=backend,
        args=args,
        model_name=args.model_name,
        messages=messages,
        fallback_messages=fallback_messages,
    )
    parsed = backend.extract_json(raw_response)
    relation_items = _relation_items_from_model_json(parsed)
    prediction, diagnostics = make_prediction_from_direct_items(
        record=record,
        relation_items=relation_items,
        allowed_relations=allowed_relations,
        args=args,
        mode_name="docred_direct_constrained_extraction",
    )
    raw_responses: Dict[str, Any] = {"primary": raw_response}

    # If the high-precision extraction returns no relations, recover recall with
    # a few short family-specific probes. This is still gold-free: families are
    # inferred from entity types and trigger words, not gold pairs.
    if (
        bool(getattr(args, "docred_zero_relation_family_probes", False))
        and not prediction.get("relations")
    ):
        probe_relation_items: List[Dict[str, Any]] = []
        probe_diags: List[Dict[str, Any]] = []
        max_probe_families = max(1, int(getattr(args, "docred_zero_relation_probe_max_families", 3) or 3))
        for family_name, family_relations in zero_relation_probe_relation_subsets(record, allowed_relations, max_families=max_probe_families):
            probe_messages = build_docred_probe_messages(
                record,
                family_relations,
                family_name=family_name,
                max_entities=getattr(args, "docred_direct_max_entities", None),
                max_relations=getattr(args, "docred_direct_max_relations", None),
            )
            try:
                probe_raw = call_docred_direct_json_with_retries(
                    backend=backend,
                    args=args,
                    model_name=args.model_name,
                    messages=probe_messages,
                    fallback_messages=None,
                )
                raw_responses[f"probe_{family_name}"] = probe_raw
                probe_parsed = backend.extract_json(probe_raw)
                probe_items = _relation_items_from_model_json(probe_parsed)
                probe_relation_items.extend(probe_items)
                probe_diags.append({"family": family_name, "raw_relation_items": len(probe_items), "error": None})
            except Exception as exc:
                probe_diags.append({"family": family_name, "raw_relation_items": 0, "error": f"{type(exc).__name__}: {exc}"})
        if probe_relation_items:
            probe_prediction, probe_diagnostics = make_prediction_from_direct_items(
                record=record,
                relation_items=probe_relation_items,
                allowed_relations=allowed_relations,
                args=args,
                mode_name="docred_zero_relation_family_probes",
            )
            prediction = merge_canonical_predictions(prediction, probe_prediction)
            prediction.setdefault("projection_diagnostics", {})["zero_relation_probes"] = {
                "enabled": True,
                "probe_attempts": probe_diags,
                "probe_diagnostics": probe_diagnostics,
            }
        else:
            prediction.setdefault("projection_diagnostics", {})["zero_relation_probes"] = {
                "enabled": True,
                "probe_attempts": probe_diags,
            }

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "docred_direct_constrained_extraction.json").write_text(
        json.dumps(
            {
                "document_id": document_id_from_record(record, 0),
                "title": title_from_record(record, document_id_from_record(record, 0)),
                "prediction": prediction,
                "diagnostics": prediction.get("projection_diagnostics") or diagnostics,
                "raw_responses": raw_responses,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return prediction
>>>>>>> d746806 (docred - relation and openrouter patch)

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
<<<<<<< HEAD
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
=======
        "linguistic_expressions": len(state.linguistic_expressions or []),
        "enriched_expressions": len(state.enriched_expressions or []),
        "entity_candidates": len(state.entity_candidates or []),
        "event_candidates": len(state.event_candidates or []),
        "attribute_candidates": len(state.attribute_candidates or []),
        "relation_candidates": len(state.relation_candidates or []),
        "candidate_relation_assertions": len(state.candidate_relation_assertions or []),
        "candidate_triples": len(state.candidate_triples or []),
        "concept_candidates": len(state.concept_candidates or []),
        "ontology_relation_candidates": len(state.ontology_relation_candidates or []),
        "axiom_schema_candidates": len(state.axiom_schema_candidates or []),
        "general_axiom_candidates": len(state.general_axiom_candidates or []),
        "completion_candidates": len(state.completion_candidates or []),
        "canonical_entities": len(prediction.get("entities") or []) if isinstance(prediction, dict) else 0,
        "canonical_relations": len(prediction.get("relations") or []) if isinstance(prediction, dict) else 0,
        "projection_rejected_triples": int(diagnostics.get("rejected_triples") or 0),
        "allowed_relation_count": int(diagnostics.get("allowed_relation_count") or 0),
        "source_entity_count": int(diagnostics.get("source_entity_count") or 0),
        "docred_direct_raw_relation_items": int(diagnostics.get("raw_relation_items") or 0),
        "docred_direct_accepted_relations": int(diagnostics.get("accepted_relations") or 0),
        "docred_direct_rejected_relations": int(diagnostics.get("rejected_relations") or 0),
        "docred_calibration_relabelled": int(((diagnostics.get("calibration") or {}).get("relabelled") or 0)) if isinstance(diagnostics.get("calibration"), dict) else 0,
        "docred_calibration_rejected": int(((diagnostics.get("calibration") or {}).get("rejected") or 0)) if isinstance(diagnostics.get("calibration"), dict) else 0,
        "docred_zero_probe_enabled": int(bool((diagnostics.get("zero_relation_probes") or {}).get("enabled"))) if isinstance(diagnostics.get("zero_relation_probes"), dict) else 0,
>>>>>>> d746806 (docred - relation and openrouter patch)
    }


def build_raw_neoolaf_summary(final_state: PipelineState) -> Dict[str, Any]:
    """Build compact raw debug summary from final NeoOLAF state."""
    return {
<<<<<<< HEAD
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
=======
        "document_id": doc_id,
        "title": title_from_record(record, doc_id),
        "type": record.get("type") or record.get("split"),
        "method": method,
        "parsed_ok": False,
        "prediction": {"entities": [], "relations": []},
        "raw_counts": {"canonical_entities": 0, "canonical_relations": 0},
        "artifact_dir": artifact_dir,
        "runtime_seconds": None,
        "error": error_message,
        "error_type": error_type,
        "error_message": error_message,
        "error_traceback": traceback_text,
        "artifact_error_files": collect_artifact_error_files(artifact_dir),
    }


def run_one_document(
    *,
    args: argparse.Namespace,
    record: Dict[str, Any],
    index: int,
    guidance: Optional[UserGuidance],
    seed_ontology: Any,
    run_stamp: str,
) -> Tuple[int, Dict[str, Any]]:
    """Execute the full NeoOLAF pipeline for one dataset document."""
    doc_id = document_id_from_record(record, index)
    safe_doc_id = safe_filename(doc_id)
    artifact_dir = str(Path(args.artifacts_root) / safe_doc_id / f"run_{run_stamp}")

    try:
        document = record_to_document(record, index, args=args)
        backend = build_backend(args)
        pipeline = build_pipeline(args, backend)

        state = PipelineState(
            document=document,
            llm_model=args.model_name,
            user_guidance=copy.deepcopy(guidance),
            seed_ontology=seed_ontology,
            artifact_dir=artifact_dir,
        )

        execution_config = ExecutionConfig(mode="document_mode")
        runner = Runner(
            pipeline=pipeline,
            runs_root=artifact_dir,
            verbose=args.verbose,
            execution_config=execution_config,
            max_workers=args.max_workers,
            enable_checkpoints=not args.no_checkpoints,
            save_chunk_checkpoints=not args.no_chunk_checkpoints,
        )

        start = time.time()
        final_state = runner.run(state)
        elapsed = time.time() - start

        prediction = state_to_canonical_prediction(
            final_state,
            source_entities=source_entities_from_record(record),
            allowed_relations=list(getattr(args, "allowed_relation_specs", []) or []),
            constrained=bool(getattr(args, "force_relation_vocabulary", False)),
        )
        if getattr(args, "docred_direct_constrained_extraction", False):
            try:
                direct_prediction = run_docred_direct_constrained_extraction(
                    record=record,
                    backend=backend,
                    args=args,
                    artifact_dir=artifact_dir,
                )
                mode = str(getattr(args, "docred_direct_output_mode", "replace") or "replace").lower()
                if mode == "supplement":
                    prediction = merge_canonical_predictions(prediction, direct_prediction)
                else:
                    prediction = direct_prediction
            except Exception as exc:
                fallback = str(getattr(args, "docred_direct_fallback_on_error", "native") or "native").lower()
                error_payload = {
                    "document_id": doc_id,
                    "title": title_from_record(record, doc_id),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "fallback": fallback,
                }
                Path(artifact_dir).mkdir(parents=True, exist_ok=True)
                (Path(artifact_dir) / "docred_direct_constrained_extraction.error.json").write_text(
                    json.dumps(error_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if fallback == "fail":
                    raise
                if fallback == "empty":
                    prediction = {
                        "entities": canonical_entities_from_source(source_entities_from_record(record)),
                        "relations": [],
                        "projection_diagnostics": {
                            "mode": "docred_direct_constrained_extraction",
                            "direct_extraction_error": True,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "fallback": fallback,
                            "source_entity_count": len(source_entities_from_record(record)),
                            "allowed_relation_count": len(list(getattr(args, "allowed_relation_specs", []) or [])),
                            "raw_relation_items": 0,
                            "accepted_relations": 0,
                            "rejected_relations": 0,
                        },
                    }
                else:
                    diagnostics = prediction.setdefault("projection_diagnostics", {})
                    if isinstance(diagnostics, dict):
                        diagnostics.update(
                            {
                                "direct_extraction_error": True,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "fallback": fallback,
                            }
                        )
        result = {
            "document_id": doc_id,
            "title": title_from_record(record, doc_id),
            "type": record.get("type") or record.get("split"),
            "method": "neoolaf",
            "parsed_ok": True,
            "prediction": prediction,
            "raw_counts": raw_counts_from_state(final_state, prediction),
            "artifact_dir": artifact_dir,
            "runtime_seconds": elapsed,
            "llm_call_policy": "full_pipeline_document_run",
        }
        return index, result
    except Exception as exc:
        traceback_text = traceback.format_exc()
        write_document_error_report(
            artifact_dir,
            doc_id=doc_id,
            error=exc,
            traceback_text=traceback_text,
        )
        return index, make_error_result(
            record,
            index,
            method="neoolaf",
            error=exc,
            artifact_dir=artifact_dir,
            traceback_text=traceback_text,
        )


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoOLAF on a RAGTree JSONL dataset.")

    parser.add_argument("--dataset-jsonl-path", required=True)
    parser.add_argument("--ontology-path", required=True)
    parser.add_argument("--output-jsonl-path", required=True)
    parser.add_argument("--backend-name", default="openrouter")
    parser.add_argument("--host", default="https://openrouter.ai/api")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--type-filter", default="all")
    parser.add_argument("--user-guidance-path", default=None)
    parser.add_argument("--few-shot-from-dataset", action="store_true")
    parser.add_argument("--few-shot-source-type", default="all")
    parser.add_argument("--few-shot-k", type=int, default=0)

    # Constrained DocRED/RAGTree benchmark export. These options do not modify
    # NeoOLAF native KG/ontology artifacts. They only constrain guidance and the
    # benchmark-facing canonical JSONL output.
    parser.add_argument("--relation-vocab-source", default="auto", choices=["auto", "dataset", "ontology", "json", "union"], help="Allowed relation vocabulary source for constrained output.")
    parser.add_argument("--relation-vocab-json", default=None, help="Optional JSON/JSONL allowed relation vocabulary.")
    parser.add_argument("--relation-vocab-dataset-path", default=None, help="Dataset JSONL used only to extract the relation label set.")
    parser.add_argument("--relation-vocab-ontology-path", default=None, help="Reference ontology used only to extract relation properties.")
    parser.add_argument("--relation-vocab-output-path", default=None, help="Write the resolved allowed relation vocabulary here.")
    parser.add_argument("--force-relation-vocabulary", action="store_true", help="Force canonical output to use only allowed relation labels.")
    parser.add_argument("--source-entity-anchoring", action="store_true", help="Expose source entity IDs/labels and require source entities in constrained output.")
    parser.add_argument("--docred-direct-constrained-extraction", action="store_true", help="Run an extra direct DocRED-constrained LLM extraction call for the final benchmark-facing canonical output.")
    parser.add_argument("--docred-direct-output-mode", default="replace", choices=["replace", "supplement"], help="How to combine direct DocRED extraction with the native NeoOLAF projection.")
    parser.add_argument("--docred-direct-max-entities", type=int, default=None, help="Optional cap on source entities shown to the direct DocRED extractor.")
    parser.add_argument("--docred-direct-max-relations", type=int, default=None, help="Optional cap on allowed relations shown to the direct DocRED extractor.")
    parser.add_argument("--docred-direct-temperature", type=float, default=0.0, help="Temperature for the direct DocRED-constrained extraction call.")
    parser.add_argument("--docred-direct-retries", type=int, default=3, help="Retry the direct DocRED extraction call when OpenRouter returns empty content or transient invalid responses.")
    parser.add_argument("--docred-direct-retry-sleep", type=float, default=2.0, help="Seconds to sleep between direct DocRED extraction retries.")
    parser.add_argument("--docred-direct-fallback-on-error", default="native", choices=["native", "empty", "fail"], help="What to do if the optional direct DocRED extraction fails after retries.")
    parser.add_argument("--docred-direct-focus-relation-ids", default=None, help="Optional comma-separated relation IDs to show to the direct extractor, e.g. P17,P27,P69. Do not derive this from test-document gold pairs for final evaluation.")
    parser.add_argument("--docred-direct-disable-hints", action="store_true", help="Disable gold-free DocRED relation disambiguation hints in the direct extraction prompt.")
    parser.add_argument("--docred-relation-family-filter", action="store_true", help="Gold-free pre-prompt relation subset inferred from entity types and trigger words.")
    parser.add_argument("--docred-calibrate-relations", action="store_true", help="Apply deterministic relabel/reject calibration for common DocRED relation confusions.")
    parser.add_argument("--docred-verification-pass", action="store_true", help="Alias for --docred-calibrate-relations and --docred-strict-type-constraints; kept for notebook/readability.")
    parser.add_argument("--docred-zero-relation-family-probes", action="store_true", help="If direct extraction returns zero relations, run targeted family probes.")
    parser.add_argument("--docred-zero-relation-probe-max-families", type=int, default=3, help="Maximum number of targeted relation family probes after zero-relation extraction.")
    parser.add_argument("--docred-strict-type-constraints", action="store_true", help="Reject common relation/type mismatches after extraction.")
    parser.add_argument("--output-format", default="canonical", choices=["canonical"])
    parser.add_argument("--artifacts-root", default="./runs/neoolaf_artifacts")

    # No-chunk benchmark mode is represented by one very large chunk.
    parser.add_argument("--chunk-size", type=int, default=10_000_000)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=1)

    # Existing intra-document limits and workers.
    parser.add_argument("--max-expressions", type=int, default=None)
    parser.add_argument("--max-relation-mentions", type=int, default=None)
    parser.add_argument("--max-concept-inputs", type=int, default=None)
    parser.add_argument("--max-relation-inputs", type=int, default=None)
    parser.add_argument("--max-concept-pairs", type=int, default=None)
    parser.add_argument("--max-relation-pairs", type=int, default=None)
    parser.add_argument("--max-relation-schema-inputs", type=int, default=None)
    parser.add_argument("--max-subclass-inputs", type=int, default=None)
    parser.add_argument("--max-schema-inputs", type=int, default=None)
    parser.add_argument("--max-description-inputs", type=int, default=None)
    parser.add_argument("--max-triples", type=int, default=None)
    parser.add_argument("--max-inferred-triples", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=1, help="Intra-document/chunk workers kept for compatibility.")

    # New document-level parallelism.
    parser.add_argument(
        "--document-workers",
        type=int,
        default=1,
        help="Number of documents to process in parallel. Default preserves old sequential behavior.",
    )

    # Diagnostics/progress.
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--show-error-traceback", action="store_true", help="Print full traceback for document errors.")
    parser.add_argument("--error-log-jsonl-path", default=None, help="Optional JSONL file for document-level errors.")
    parser.add_argument("--summary-output-path", default=None, help="Optional JSON summary of the benchmark run.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first document-level error.")

    # Runtime controls.
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--openrouter-reasoning-effort",
        default="minimal",
        choices=["xhigh", "high", "medium", "low", "minimal", "none", ""],
        help="OpenRouter reasoning effort for reasoning models. Use minimal/none to avoid empty final content on gpt-oss providers.",
    )
    parser.add_argument(
        "--openrouter-exclude-reasoning",
        action="store_true",
        default=True,
        help="Request OpenRouter to exclude reasoning traces from the response. Enabled by default.",
    )
    parser.add_argument(
        "--openrouter-include-reasoning",
        dest="openrouter_exclude_reasoning",
        action="store_false",
        help="Debug option: allow reasoning traces to be returned by OpenRouter.",
    )
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search in enrichment for speed/reproducibility.")
    parser.add_argument(
        "--disable-wikipedia-lookups",
        action="store_true",
        help="Use offline Wikipedia/Wikidata source objects in Layer 2 without changing NeoOLAF source.",
    )
    parser.add_argument(
        "--offline-ontology-only",
        action="store_true",
        help="Benchmark mode: disable web enrichment and block Wikipedia/Wikimedia lookups.",
    )
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--no-chunk-checkpoints", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """Write canonical JSONL output atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def default_error_log_path(output_jsonl_path: str | Path) -> Path:
    """Default error JSONL path derived from the prediction output path."""
    path = Path(output_jsonl_path)
    return path.with_name(path.stem + ".errors.jsonl")


def default_summary_path(output_jsonl_path: str | Path) -> Path:
    """Default run summary JSON path derived from the prediction output path."""
    path = Path(output_jsonl_path)
    return path.with_name(path.stem + ".run_summary.json")


def relation_count_from_result(result: Dict[str, Any]) -> int:
    """Return number of canonical predicted relations for one result row."""
    return len(((result.get("prediction") or {}).get("relations") or []))


def build_run_summary(
    *,
    args: argparse.Namespace,
    final_rows: List[Dict[str, Any]],
    elapsed: float,
) -> Dict[str, Any]:
    """Build a compact dataset-level run summary with error diagnostics."""
    parsed_ok = sum(1 for row in final_rows if row.get("parsed_ok"))
    error_rows = [row for row in final_rows if not row.get("parsed_ok")]
    zero_relation_docs = [
        row.get("document_id")
        for row in final_rows
        if row.get("parsed_ok") and relation_count_from_result(row) == 0
    ]
    return {
        "dataset_jsonl_path": args.dataset_jsonl_path,
        "ontology_path": args.ontology_path,
        "output_jsonl_path": args.output_jsonl_path,
        "model_name": args.model_name,
        "type_filter": args.type_filter,
        "documents": len(final_rows),
        "parsed_ok": parsed_ok,
        "failed": len(error_rows),
        "relations": sum(relation_count_from_result(row) for row in final_rows),
        "elapsed_seconds": elapsed,
        "document_workers": args.document_workers,
        "max_workers": args.max_workers,
        "error_type_counts": dict(Counter(str(row.get("error_type", "Error")) for row in error_rows)),
        "zero_relation_docs_count": len(zero_relation_docs),
        "zero_relation_doc_ids_preview": zero_relation_docs[:20],
        "errors_preview": [
            {
                "document_id": row.get("document_id"),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message") or row.get("error"),
                "artifact_dir": row.get("artifact_dir"),
                "artifact_error_files": row.get("artifact_error_files", [])[:3],
            }
            for row in error_rows[:20]
>>>>>>> d746806 (docred - relation and openrouter patch)
        ],
    }


<<<<<<< HEAD
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
=======
def main() -> None:
    args = parse_args()
    if getattr(args, "docred_verification_pass", False):
        args.docred_calibrate_relations = True
        args.docred_strict_type_constraints = True
    if getattr(args, "offline_ontology_only", False):
        args.no_web_search = True
        args.disable_wikipedia_lookups = True
    env_disable_wiki = os.environ.get("NEOOLAF_DISABLE_WIKIPEDIA", "").strip().lower() in {"1", "true", "yes", "on"}
    if env_disable_wiki:
        args.disable_wikipedia_lookups = True
    if args.disable_wikipedia_lookups:
        print("[NeoOLAF benchmark] Wikipedia/Wikidata enrichment disabled by offline source objects.")
    if args.no_web_search:
        print("[NeoOLAF benchmark] Web-search enrichment disabled (--no-web-search).")
    start = time.time()
>>>>>>> d746806 (docred - relation and openrouter patch)

        try:
            doc = dict(doc)
            doc["document_id"] = document_id

<<<<<<< HEAD
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
=======
    records_all = load_jsonl(args.dataset_jsonl_path)
    records = filter_records(records_all, args.type_filter)
    if args.max_docs is not None:
        records = records[: args.max_docs]

    if not records:
        raise SystemExit("No records selected. Check --dataset-jsonl-path and --type-filter.")

    args.allowed_relation_specs = load_allowed_relation_specs(args)
    if args.allowed_relation_specs:
        print(
            f"[NeoOLAF benchmark] allowed_relations={len(args.allowed_relation_specs)} "
            f"source={args.relation_vocab_source} force={args.force_relation_vocabulary}"
        )
        preview = [rel.get("canonical") for rel in args.allowed_relation_specs[:10]]
        print(f"[NeoOLAF benchmark] allowed_relations_preview={preview}")
    elif args.force_relation_vocabulary:
        print(
            "[NeoOLAF benchmark][warning] --force-relation-vocabulary was set, "
            "but no allowed relations were loaded. Canonical relations will be rejected."
        )

    guidance = load_user_guidance(args.user_guidance_path)
    if args.force_relation_vocabulary:
        guidance = inject_relation_constraints_into_guidance(guidance, args.allowed_relation_specs)
    if args.docred_direct_constrained_extraction:
        print(
            "[NeoOLAF benchmark] docred_direct_constrained_extraction=True "
            f"mode={args.docred_direct_output_mode} "
            f"focus_relation_ids={args.docred_direct_focus_relation_ids} "
            f"hints={not args.docred_direct_disable_hints} "
            f"retries={args.docred_direct_retries} "
            f"fallback={args.docred_direct_fallback_on_error}"
        )
    if args.few_shot_from_dataset:
        guidance = add_few_shot_examples_from_dataset(
            guidance,
            records_all,
            source_type=args.few_shot_source_type,
            k=args.few_shot_k,
        )

    seed_ontology = SeedOntologyLoader().load(args.ontology_path)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")

    print(
        "[NeoOLAF benchmark] "
        f"documents={len(records)} document_workers={args.document_workers} "
        f"max_workers={args.max_workers} model={args.model_name}"
    )

    results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    workers = max(1, int(args.document_workers or 1))
    progress = make_progress(len(records), "NeoOLAF documents", disable=args.no_tqdm)

    def handle_result(completed_no: int, out_idx: int, result: Dict[str, Any]) -> None:
        """Store and log one document result."""
        results[out_idx] = result
        progress.update(1)
        relation_count = relation_count_from_result(result)
        runtime = result.get("runtime_seconds")
        runtime_txt = f" time={runtime:.2f}s" if isinstance(runtime, (int, float)) else ""
        if result.get("parsed_ok"):
            msg = (
                f"[{completed_no}/{len(records)}] {result['document_id']} ok "
                f"relations={relation_count}{runtime_txt}"
>>>>>>> d746806 (docred - relation and openrouter patch)
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