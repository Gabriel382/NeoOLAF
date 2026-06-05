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
from neoolaf.core.execution_plan import ExecutionPlan
from neoolaf.agents.orchestrator import LayerOrchestrator
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
    run_parser.add_argument("--orchestration-mode", default="pipeline", choices=["pipeline", "agentic"], help="Simple orchestrator mode label. Both modes currently use the same safe runner; 'agentic' is reserved for future feedback policies.")
    run_parser.add_argument("--max-concurrency-layer01", type=int, default=None, help="Bounded parallel LLM calls for independent Layer 1 units. Defaults to profile setting or 1.")
    run_parser.add_argument("--retry-failed-calls", type=int, default=None, help="Retry count for failed Layer 1 unit calls. Defaults to profile setting or 0.")
    run_parser.add_argument("--retry-sleep-seconds", type=float, default=None, help="Sleep between failed Layer 1 retries. Defaults to profile setting or 2.0.")
    run_parser.add_argument("--rag-layer01", action="store_true", help="Enable small optional RAG guidance for Layer 1. Disabled by default.")
    run_parser.add_argument("--rag-top-k", type=int, default=None, help="Top-k snippets for optional Layer 1 RAG guidance. Use 0 to disable. Defaults to profile setting or 0.")
    run_parser.add_argument("--rag-max-chars", type=int, default=None, help="Maximum characters injected from optional Layer 1 RAG guidance. Defaults to profile setting or 0.")
    run_parser.add_argument("--structured-output-layer01", default="auto", choices=["auto", "on", "off"], help="Override profile setting for optional Pydantic structured-output validation on Layer 1.")
    run_parser.add_argument("--litellm-response-format-layer01", action="store_true", help="Ask LiteLLM to use provider structured response_format for Layer 1 when supported.")
    run_parser.add_argument("--strict-structured-output-layer01", action="store_true", help="Fail a Layer 1 unit when Pydantic validation fails instead of falling back.")
    run_parser.add_argument("--layer01-failed-chunks-file", default=None, help="Optional failed_chunks.json from a previous Layer 1 run. When provided, Layer 1 only reruns the listed chunks.")
    run_parser.add_argument("--max-concurrency-layer02", type=int, default=None, help="Bounded parallel LLM calls for independent Layer 2 expression enrichment. Defaults to profile setting or 1.")
    run_parser.add_argument("--max-concurrency-layer03", type=int, default=None, help="Bounded parallel LLM calls for independent Layer 3 local typing. Defaults to profile setting or 1.")
    run_parser.add_argument("--max-concurrency-layer04", type=int, default=None, help="Bounded parallel relation assertion extraction for Layer 4 when the selected strategy needs it. The ontology-aware record strategy is deterministic and normally does not need LLM calls.")
    run_parser.add_argument("--max-concurrency-layer05", type=int, default=None, help="Accepted for Layer 5 CLI/orchestrator consistency. The ontology-aware assertion-to-triples strategy is deterministic and does not call the LLM.")
    run_parser.add_argument("--max-expressions-layer02", type=int, default=None, help="Optional expression limit for Layer 2 debugging/ablation.")
    run_parser.add_argument("--max-expressions-layer03", type=int, default=None, help="Optional expression limit for Layer 3 debugging/ablation.")
    run_parser.add_argument("--layer02-failed-expressions-file", default=None, help="Optional failed_expressions.json from a previous Layer 2 run. When provided, Layer 2 only reruns the listed expressions.")
    run_parser.add_argument("--layer03-failed-items-file", default=None, help="Optional failed_items.json from a previous Layer 3 run. When provided, Layer 3 only reruns the listed items.")

    return parser


def _apply_structured_output_cli_overrides(profile_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply optional CLI overrides for Layer 1 structured output.

    Profiles remain the source of truth by default. These switches only make it
    easy to run ablations without editing the profile JSON.
    """
    layer_name = "layer01_linguistic_expression_extraction"
    layer_cfg = profile_config.setdefault("layers", {}).setdefault(layer_name, {})
    structured = layer_cfg.setdefault("structured_output", {})

    if getattr(args, "structured_output_layer01", "auto") == "on":
        structured["enabled"] = True
    elif getattr(args, "structured_output_layer01", "auto") == "off":
        structured["enabled"] = False

    if getattr(args, "litellm_response_format_layer01", False):
        structured["use_litellm_response_format"] = True
    if getattr(args, "strict_structured_output_layer01", False):
        structured["strict_validation"] = True
    return profile_config


def run_command(args: argparse.Namespace) -> None:
    skip_layers = _parse_skip_layers(args.skip_layers)
    document_profile = load_document_profile(args.profile, args.profile_path)
    profile_config = document_profile.to_state_dict()
    profile_config = _apply_structured_output_cli_overrides(profile_config, args)

    orchestration_cfg = profile_config.get("orchestration", {}) if isinstance(profile_config, dict) else {}
    layer01_cfg = (
        profile_config.get("layers", {}).get("layer01_linguistic_expression_extraction", {})
        if isinstance(profile_config, dict)
        else {}
    )
    layer02_cfg = (
        profile_config.get("layers", {}).get("layer02_candidate_enrichment", {})
        if isinstance(profile_config, dict)
        else {}
    )
    layer03_cfg = (
        profile_config.get("layers", {}).get("layer03_candidate_typing_resolution", {})
        if isinstance(profile_config, dict)
        else {}
    )
    layer04_cfg = (
        profile_config.get("layers", {}).get("layer04_candidate_relation_extraction", {})
        if isinstance(profile_config, dict)
        else {}
    )
    layer05_cfg = (
        profile_config.get("layers", {}).get("layer05_candidate_triple_generation", {})
        if isinstance(profile_config, dict)
        else {}
    )
    max_concurrency_layer01 = int(
        args.max_concurrency_layer01
        if args.max_concurrency_layer01 is not None
        else orchestration_cfg.get("max_concurrency_layer01", layer01_cfg.get("max_concurrency", 1))
    )
    max_concurrency_layer02 = int(
        args.max_concurrency_layer02
        if args.max_concurrency_layer02 is not None
        else orchestration_cfg.get("max_concurrency_layer02", layer02_cfg.get("max_concurrency", 1))
    )
    max_concurrency_layer03 = int(
        args.max_concurrency_layer03
        if args.max_concurrency_layer03 is not None
        else orchestration_cfg.get("max_concurrency_layer03", layer03_cfg.get("max_concurrency", 1))
    )
    max_concurrency_layer04 = int(
        args.max_concurrency_layer04
        if args.max_concurrency_layer04 is not None
        else orchestration_cfg.get("max_concurrency_layer04", layer04_cfg.get("max_concurrency", 1))
    )
    max_concurrency_layer05 = int(
        args.max_concurrency_layer05
        if args.max_concurrency_layer05 is not None
        else orchestration_cfg.get("max_concurrency_layer05", layer05_cfg.get("max_concurrency", 1))
    )
    retry_failed_calls = int(
        args.retry_failed_calls
        if args.retry_failed_calls is not None
        else orchestration_cfg.get("retry_failed_calls", layer01_cfg.get("retry_failed_calls", 0))
    )
    max_expressions_layer02 = (
        args.max_expressions_layer02
        if args.max_expressions_layer02 is not None
        else layer02_cfg.get("max_expressions")
    )
    max_expressions_layer03 = (
        args.max_expressions_layer03
        if args.max_expressions_layer03 is not None
        else layer03_cfg.get("max_expressions")
    )
    retry_sleep_seconds = float(
        args.retry_sleep_seconds
        if args.retry_sleep_seconds is not None
        else orchestration_cfg.get("retry_sleep_seconds", layer01_cfg.get("retry_sleep_seconds", 2.0))
    )
    rag_top_k = int(
        args.rag_top_k
        if args.rag_top_k is not None
        else layer01_cfg.get("rag_top_k", 0)
    )
    rag_max_chars = int(
        args.rag_max_chars
        if args.rag_max_chars is not None
        else layer01_cfg.get("rag_max_chars", 0)
    )
    rag_layer01_enabled = bool(args.rag_layer01 or layer01_cfg.get("rag_enabled", False))

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
        max_concurrency_layer01=max_concurrency_layer01,
        max_concurrency_layer02=max_concurrency_layer02,
        max_concurrency_layer03=max_concurrency_layer03,
        max_concurrency_layer04=max_concurrency_layer04,
        max_concurrency_layer05=max_concurrency_layer05,
        retry_failed_calls=retry_failed_calls,
        retry_sleep_seconds=retry_sleep_seconds,
        rag_layer01_enabled=rag_layer01_enabled,
        rag_top_k_layer01=rag_top_k,
        rag_max_chars_layer01=rag_max_chars,
        failed_chunks_file_layer01=args.layer01_failed_chunks_file,
        failed_expressions_file_layer02=args.layer02_failed_expressions_file,
        failed_items_file_layer03=args.layer03_failed_items_file,
        max_expressions_layer02=max_expressions_layer02,
        max_expressions_layer03=max_expressions_layer03,
        profile_config=profile_config,
        translate_preprocessing=bool(args.translate_preprocessing),
        translator=translator,
        source_language=args.source_language,
        target_language=target_language,
    )

    pipeline = Pipeline(layers=layers, verbose=args.verbose)
    runner = Runner(pipeline=pipeline, verbose=args.verbose)
    execution_plan = ExecutionPlan(
        from_layer=args.from_layer,
        to_layer=args.to_layer,
        skip_layers=skip_layers,
        mode=args.orchestration_mode,
        max_concurrency_layer01=max_concurrency_layer01,
        max_concurrency_layer02=max_concurrency_layer02,
        max_concurrency_layer03=max_concurrency_layer03,
        max_concurrency_layer04=max_concurrency_layer04,
        max_concurrency_layer05=max_concurrency_layer05,
        retry_failed_calls=retry_failed_calls,
        retry_sleep_seconds=retry_sleep_seconds,
        rag_backend=args.rag_backend,
        rag_layer01_enabled=rag_layer01_enabled,
        rag_top_k=rag_top_k,
        rag_max_chars=rag_max_chars,
    )
    orchestrator = LayerOrchestrator(runner=runner, plan=execution_plan, verbose=args.verbose)

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
            "orchestration_mode": args.orchestration_mode,
            "execution_plan": execution_plan.to_dict(),
            "max_concurrency_layer01": max_concurrency_layer01,
            "max_concurrency_layer02": max_concurrency_layer02,
            "max_concurrency_layer03": max_concurrency_layer03,
            "max_concurrency_layer04": max_concurrency_layer04,
            "max_concurrency_layer05": max_concurrency_layer05,
            "retry_failed_calls": retry_failed_calls,
            "retry_sleep_seconds": retry_sleep_seconds,
            "rag_layer01_enabled": rag_layer01_enabled,
            "rag_top_k": rag_top_k,
            "rag_max_chars": rag_max_chars,
            "structured_output_layer01": profile_config.get("layers", {}).get("layer01_linguistic_expression_extraction", {}).get("structured_output", {}),
            "layer01_failed_chunks_file": args.layer01_failed_chunks_file,
            "layer02_failed_expressions_file": args.layer02_failed_expressions_file,
            "layer03_failed_items_file": args.layer03_failed_items_file,
            "max_expressions_layer02": max_expressions_layer02,
            "max_expressions_layer03": max_expressions_layer03,
        },
    )

    final_state = orchestrator.run(
        state,
        run_dir=args.run_dir,
        resume_from=None,
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
