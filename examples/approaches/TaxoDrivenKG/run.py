"""Main entry point for the XQuality TaxoDrivenKG adaptation."""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from consts import PATH
from utils import load_json_file, save_json_file
from utils_LLM import InfoExtractor, TextChunker
from taxonomy import OntologyRetriever
from serialization import save_ttl_outputs
from backends.openai_compatible import OpenAICompatibleBackend
from backends.ollama_backend import OllamaBackend


def load_translated_text_from_state(state_json_path: str | Path) -> Tuple[str, List[dict]]:
    """Load NeoOLAF layer00 state JSON and concatenate only translated_text blocks."""
    state = load_json_file(state_json_path)
    blocks = state.get("content_blocks", [])
    translated_blocks: List[dict] = []
    merged_parts: List[str] = []

    for block in blocks:
        translated_text = (block.get("translated_text") or "").strip()
        if translated_text:
            translated_blocks.append(block)
            merged_parts.append(translated_text)

    full_text = "\n\n".join(merged_parts)
    return full_text, translated_blocks


def build_backend(args: argparse.Namespace):
    """Create the requested backend."""
    if args.backend == "ollama":
        return OllamaBackend(host=args.host)

    api_key = args.api_key
    if not api_key:
        if args.backend == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
        elif args.backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
        else:
            api_key = os.getenv("OPENAI_API_KEY", "dummy")

    extra_headers = None
    if args.backend == "openrouter":
        extra_headers = {}
        if args.referer:
            extra_headers["HTTP-Referer"] = args.referer
        if args.title:
            extra_headers["X-Title"] = args.title

    return OpenAICompatibleBackend(
        base_url=args.host,
        api_key=api_key,
        extra_headers=extra_headers,
    )


def main() -> None:
    """CLI runner."""
    parser = argparse.ArgumentParser(
        description="Run TaxoDrivenKG-style extraction on NeoOLAF translated_text blocks."
    )
    parser.add_argument("--state_json", type=str, default=PATH["inputs"]["state_json"], help="Path to the NeoOLAF layer00 state JSON.")
    parser.add_argument("--ontology", type=str, default=PATH["inputs"]["ontology"], help="Path to the OWL ontology.")
    parser.add_argument("--env_path", type=str, default=PATH["inputs"]["env_file"], help="Path to the .env file.")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "openai", "openrouter", "ollama"], help="Backend type.")
    parser.add_argument("--host", type=str, default="http://localhost:8000/v1", help="Backend base URL or Ollama host.")
    parser.add_argument("--api_key", type=str, default="", help="API key for OpenAI-compatible backends. If empty, load from .env.")
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-20b", help="Model name to send to the backend.")
    parser.add_argument("--experiment", type=str, default="base", help="Prompt mode: base, 0_shot, no_rag, etc.")
    parser.add_argument("--chunk_tokens", type=int, default=600, help="Chunk size in tokens.")
    parser.add_argument("--max_taxonomy_hits", type=int, default=40, help="Max ontology candidates injected per chunk.")
    parser.add_argument("--output_dir", type=str, default=PATH["outputs"]["base"], help="Directory where JSON outputs are written.")
    parser.add_argument("--prompts_dir", type=str, default=PATH["outputs"]["prompts"], help="Directory where prompts are written.")
    parser.add_argument("--conversations_dir", type=str, default=PATH["outputs"]["conversations"], help="Directory where conversations are written.")
    parser.add_argument("--ttl_dir", type=str, default=PATH["outputs"]["ttl"], help="Directory where TTL outputs are written.")
    parser.add_argument("--referer", type=str, default="", help="Optional HTTP-Referer for OpenRouter.")
    parser.add_argument("--title", type=str, default="TaxoDrivenKG-XQuality", help="Optional X-Title for OpenRouter.")
    args = parser.parse_args()

    load_dotenv(args.env_path)

    output_dir = Path(args.output_dir)
    prompts_dir = Path(args.prompts_dir)
    conversations_dir = Path(args.conversations_dir)
    ttl_dir = Path(args.ttl_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    conversations_dir.mkdir(parents=True, exist_ok=True)
    ttl_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.datetime.now()
    print(f"Time now: {start_time}")
    print(f"Loading state JSON: {args.state_json}")

    full_text, translated_blocks = load_translated_text_from_state(args.state_json)
    print(f"Translated blocks loaded: {len(translated_blocks)}")
    print(f"Merged translated chars: {len(full_text)}")

    backend = build_backend(args)
    model = InfoExtractor(backend=backend, model_name=args.model_name, exp=args.experiment, n_shot=3)
    chunker = TextChunker(text_max_tokens=args.chunk_tokens)
    retriever = OntologyRetriever(args.ontology)

    text_chunks, start_idxs = chunker.get_text_chunks(full_text)
    outputs: Dict[str, dict] = {}
    conversations: Dict[str, list] = {}
    prompts: Dict[str, str] = {}

    for chunk_i, (text, start_idx) in enumerate(zip(text_chunks, start_idxs), start=1):
        end_idx = start_idx + len(text)
        print(f"Chunk [{chunk_i}/{len(text_chunks)}] Retrieving ontology candidates")
        retrieved_nodes = retriever.retrieve(text, max_hits=args.max_taxonomy_hits)

        print(f"Chunk [{chunk_i}/{len(text_chunks)}] Running backend")
        output, conversation = model.run(text, retrieved_nodes)

        span_key = str((start_idx, end_idx))
        outputs[span_key] = output
        conversations[span_key] = conversation
        prompts[span_key] = conversation[0]["content"]

    state_name = Path(args.state_json).stem

    output_json_path = output_dir / f"{state_name}.json"
    prompts_json_path = prompts_dir / f"{state_name}.json"
    conversations_json_path = conversations_dir / f"{state_name}.json"

    save_json_file(output_json_path, outputs)
    save_json_file(prompts_json_path, prompts)
    save_json_file(conversations_json_path, conversations)

    ontology_ttl_path, kg_ttl_path = save_ttl_outputs(
        outputs=outputs,
        seed_ontology_path=args.ontology,
        ttl_dir=ttl_dir,
        state_name=state_name,
    )

    print(f"Saved outputs to: {output_json_path}")
    print(f"Saved prompts to: {prompts_json_path}")
    print(f"Saved conversations to: {conversations_json_path}")
    print(f"Saved ontology TTL to: {ontology_ttl_path}")
    print(f"Saved KG TTL to: {kg_ttl_path}")
    print(f"Time elapsed: {datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    main()
    sys.exit(0)