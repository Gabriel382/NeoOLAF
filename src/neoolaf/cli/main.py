from __future__ import annotations

"""Minimal CLI for NeoOLAF ablation runs."""

import argparse
from pathlib import Path
from typing import Any

from neoolaf.core.artifact_store import ArtifactStore
from neoolaf.core.layer_factory import LLM_LAYER_INDEXES, build_default_layers
from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner
from neoolaf.domain.documents import Document
from neoolaf.resources.llm_backends.chat_compat import ChatCompatBackend
from neoolaf.grounding.rag.factory import build_rag_backend
from neoolaf.profiles.profile_loader import load_document_profile


def _parse_skip_layers(value: str | None) -> list[int | str]:
    if not value:
        return []

    result: list[int | str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError:
            result.append(item)
    return result


def _needs_llm(from_layer: int, to_layer: int | None, skip_layers: list[int | str]) -> bool:
    to = 12 if to_layer is None else to_layer
    skipped = set(skip_layers)
    for idx in range(from_layer, to + 1):
        if idx in skipped:
            continue
        if idx in LLM_LAYER_INDEXES:
            return True
    return False


def _build_litellm_backend(model: str) -> Any:
    """
    Build the user's LiteLLM backend if available.

    This assumes step 2 has already created
    `neoolaf.resources.llm_backends.litellm_backend.LiteLLMBackend`.
    """
    try:
        from neoolaf.resources.llm_backends.litellm_backend import LiteLLMBackend

        try:
            backend = LiteLLMBackend(model=model)
        except TypeError:
            backend = LiteLLMBackend()
        return ChatCompatBackend(backend)
    except Exception as exc:
        raise RuntimeError(
            "Could not build LiteLLMBackend. Make sure step 2 exists at "
            "neoolaf.resources.llm_backends.litellm_backend.LiteLLMBackend. "
            f"Original error: {exc}"
        ) from exc




def _build_preprocessing_translator(name: str):
    """Build a translation backend for optional Layer 0 translation."""
    if name == "none":
        return None
    if name == "deep":
        try:
            from neoolaf.resources.translation.deep_translator_backend import DeepTranslatorBackend

            return DeepTranslatorBackend()
        except Exception as exc:
            raise RuntimeError(
                "Could not build DeepTranslatorBackend. Install deep-translator or "
                "run without --translate-preprocessing."
            ) from exc
    raise ValueError(f"Unknown preprocessing translator: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeoOLAF ablation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run all or part of the NeoOLAF pipeline")
    run_parser.add_argument("--input-pdf", help="Input PDF path. Required unless --resume-from is used.")
    run_parser.add_argument("--resume-from", help="Path to a previous layer state.json or checkpoint.")
    run_parser.add_argument("--run-dir", required=True, help="Run output directory.")
    run_parser.add_argument("--model", default="openrouter/openai/gpt-oss-20b")
    run_parser.add_argument("--from-layer", type=int, default=0)
    run_parser.add_argument("--to-layer", type=int, default=None)
    run_parser.add_argument("--skip-layers", default="", help="Comma-separated layer indexes or names")
    run_parser.add_argument("--chunk-size", type=int, default=1500)
    run_parser.add_argument("--overlap", type=int, default=200)
    run_parser.add_argument("--max-chunks-layer01", type=int, default=None, help="Optional chunk limit for Layer 1 debugging/ablation")
    run_parser.add_argument("--verbose", action="store_true")
    run_parser.add_argument("--rag-backend", default="agentic", choices=["agentic", "none", "ragtree"], help="RAG backend for layer-compatible retrieval stubs")
    run_parser.add_argument("--profile", default="generic", help="Document profile name from configs/document_profiles or a profile file path")
    run_parser.add_argument("--profile-path", default=None, help="Explicit JSON/YAML document profile path")
    run_parser.add_argument("--translate-preprocessing", action="store_true", help="Optionally translate content during Layer 0 preprocessing and cache it in the state. Disabled by default to avoid extra calls/tokens.")
    run_parser.add_argument("--preprocessing-translator", default="deep", choices=["deep", "none"], help="Translator backend used only when --translate-preprocessing is enabled.")
    run_parser.add_argument("--source-language", default=None, help="Optional source language code for preprocessing translation, e.g. fr. If omitted, Layer 0 may auto-detect.")
    run_parser.add_argument("--target-language", default=None, help="Optional target language code for preprocessing translation. Defaults to the profile language target or en.")

    return parser


def run_command(args: argparse.Namespace) -> None:
    skip_layers = _parse_skip_layers(args.skip_layers)
    document_profile = load_document_profile(args.profile, args.profile_path)
    profile_config = document_profile.to_state_dict()

    if args.resume_from:
        state = Runner.load_state_artifact(args.resume_from)
        state.llm_model = args.model
        state.artifact_dir = args.run_dir
        # Attach/override the profile at resume time so the selected layers can
        # immediately use the requested profile.
        state.profile_name = document_profile.name
        state.profile_config = profile_config
    else:
        if not args.input_pdf:
            raise ValueError("--input-pdf is required unless --resume-from is provided.")
        source_path = str(Path(args.input_pdf))
        state = PipelineState(
            document=Document(
                doc_id=Path(source_path).stem,
                source_path=source_path,
                raw_text="",
            ),
            llm_model=args.model,
            artifact_dir=args.run_dir,
            profile_name=document_profile.name,
            profile_config=profile_config,
        )

    llm_backend = None
    if _needs_llm(args.from_layer, args.to_layer, skip_layers):
        llm_backend = _build_litellm_backend(args.model)

    rag_backend = build_rag_backend(args.rag_backend)

    language_cfg = profile_config.get("language", {}) if isinstance(profile_config, dict) else {}
    target_language = args.target_language or language_cfg.get("target") or "en"
    translator = None
    if args.translate_preprocessing:
        translator = _build_preprocessing_translator(args.preprocessing_translator)
        if translator is None:
            raise ValueError("--translate-preprocessing requires a preprocessing translator other than 'none'.")

    layers = build_default_layers(
        llm_backend=llm_backend,
        rag_backend=rag_backend,
        verbose=args.verbose,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_chunks_layer01=args.max_chunks_layer01,
        profile_config=profile_config,
        translate_preprocessing=bool(args.translate_preprocessing),
        translator=translator,
        source_language=args.source_language,
        target_language=target_language,
    )

    pipeline = Pipeline(layers=layers, verbose=args.verbose)
    runner = Runner(pipeline=pipeline, verbose=args.verbose)

    ArtifactStore.write_run_config(
        args.run_dir,
        {
            "input_pdf": args.input_pdf,
            "resume_from": args.resume_from,
            "model": args.model,
            "from_layer": args.from_layer,
            "to_layer": args.to_layer,
            "skip_layers": skip_layers,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "max_chunks_layer01": args.max_chunks_layer01,
            "rag_backend": args.rag_backend,
            "profile": args.profile,
            "profile_path": args.profile_path,
            "resolved_profile_name": document_profile.name,
            "profile_config": profile_config,
            "translate_preprocessing": bool(args.translate_preprocessing),
            "preprocessing_translator": args.preprocessing_translator,
            "source_language": args.source_language,
            "target_language": target_language,
        },
    )

    final_state = runner.run(
        state,
        from_layer=args.from_layer,
        to_layer=args.to_layer,
        skip_layers=skip_layers,
        resume_from=None,
        run_dir=args.run_dir,
    )

    print(f"[NeoOLAF] Finished. Final state saved under: {final_state.artifact_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
