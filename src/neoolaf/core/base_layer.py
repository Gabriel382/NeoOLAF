from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import time

from neoolaf.core.artifact_store import ArtifactStore
from neoolaf.core.prompt_capture import PromptCaptureBackend, summarize_prompt_records
from neoolaf.core.pipeline_state import PipelineState


class BaseLayer(ABC):
    """
    Abstract base class for all NeoOLAF layers.

    Every layer must:
    - define a unique name
    - implement `_run`
    - optionally save intermediate results
    - receive and return a PipelineState

    For ablation, each layer now saves a restartable `state.json` and a compact
    `output.json` under `<artifact_dir>/<layer_name>/`.
    """

    name: str = "base_layer"

    def __init__(self, save_intermediate: bool = True, verbose: bool = False) -> None:
        self.save_intermediate = save_intermediate
        self.verbose = verbose

    def run(self, state: PipelineState) -> PipelineState:
        start_time = time.time()

        state.log({"layer": self.name, "status": "started", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.prompt_records = []
        original_backend = getattr(self, "ollama_backend", None)
        capture_backend = None
        if original_backend is not None:
            capture_backend = PromptCaptureBackend(original_backend, layer_name=self.name)
            self.ollama_backend = capture_backend

        if self.verbose:
            print(f"\n[NeoOLAF] Starting layer: {self.name}")

        try:
            state = self._run(state)
        finally:
            if capture_backend is not None:
                self.prompt_records = capture_backend.records
                self.ollama_backend = original_backend

        elapsed = time.time() - start_time
        state.log(
            {
                "layer": self.name,
                "status": "completed",
                "elapsed_seconds": round(elapsed, 3),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        if self.save_intermediate and state.artifact_dir is not None:
            self._save_state(state, elapsed_seconds=elapsed)

        if self.verbose:
            print(f"[NeoOLAF] Finished layer: {self.name} in {elapsed:.2f}s")

        return state

    @abstractmethod
    def _run(self, state: PipelineState) -> PipelineState:
        raise NotImplementedError

    def _save_state(self, state: PipelineState, elapsed_seconds: float | None = None) -> None:
        """
        Save ablation-ready artifacts for this layer.

        New canonical files:
            <artifact_dir>/<layer_name>/state.json
            <artifact_dir>/<layer_name>/output.json
            <artifact_dir>/<layer_name>/metadata.json

        Backward-compatible file kept:
            <artifact_dir>/<layer_name>/<document_name>.json
        """
        artifact_store = ArtifactStore(state.artifact_dir)
        payload = self.build_artifact_payload(state)
        metadata = self.build_metadata_payload(state, elapsed_seconds=elapsed_seconds)

        artifact_store.save_layer_artifacts(
            layer_name=self.name,
            state=state,
            output_payload=payload,
            metadata=metadata,
        )

        prompt_records = getattr(self, "prompt_records", [])
        if prompt_records:
            prompts_dir = artifact_store.layer_dir(self.name) / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            prompt_stats = summarize_prompt_records(prompt_records)
            artifact_store.save_json(self.name, "prompt_stats.json", {
                **prompt_stats,
                "records": [record.to_dict() for record in prompt_records],
            })
            for record in prompt_records:
                prefix = f"prompt_{record.call_index:03d}"
                (prompts_dir / f"{prefix}_system.txt").write_text(record.system_prompt, encoding="utf-8")
                (prompts_dir / f"{prefix}_user.txt").write_text(record.user_prompt, encoding="utf-8")
                (prompts_dir / f"{prefix}_full.txt").write_text(record.full_prompt, encoding="utf-8")

        # Backward compatibility with the previous artifact convention.
        layer_dir = Path(state.artifact_dir) / self.name
        document_name = Path(state.document.source_path).stem or state.document.doc_id
        artifact_store.save_json(self.name, f"{document_name}.json", payload)

    def build_artifact_payload(self, state: PipelineState) -> dict[str, Any]:
        """
        Build a compact layer-specific payload.

        Layers can override this.  The default payload is intentionally useful
        for ablation: it exposes output counters for every layer-level field.
        """
        return {
            "layer": self.name,
            "document_id": state.document.doc_id,
            "counts": self._state_counts(state),
            "logs": state.logs,
        }

    def build_metadata_payload(
        self,
        state: PipelineState,
        elapsed_seconds: float | None = None,
    ) -> dict[str, Any]:
        prompt_stats = summarize_prompt_records(getattr(self, "prompt_records", []))
        return {
            "document_id": state.document.doc_id,
            "source_path": state.document.source_path,
            "llm_model": state.llm_model,
            "elapsed_seconds": None if elapsed_seconds is None else round(elapsed_seconds, 3),
            "counts": self._state_counts(state),
            "prompt_stats": prompt_stats,
        }

    @staticmethod
    def _state_counts(state: PipelineState) -> dict[str, int]:
        """Return compact counts for all major layer outputs."""
        count_fields = [
            "linguistic_expressions",
            "enriched_expressions",
            "entity_candidates",
            "relation_candidates",
            "attribute_candidates",
            "event_candidates",
            "candidate_relation_assertions",
            "candidate_triples",
            "concept_candidates",
            "ontology_relation_candidates",
            "concept_hierarchy_links",
            "relation_hierarchy_links",
            "axiom_schema_candidates",
            "general_axiom_candidates",
            "completion_candidates",
        ]
        counts: dict[str, int] = {}
        for field_name in count_fields:
            value = getattr(state, field_name, None)
            counts[field_name] = len(value) if value is not None else 0
        counts["validation_issues"] = (
            len(state.validation_report.issues)
            if state.validation_report is not None and hasattr(state.validation_report, "issues")
            else 0
        )
        counts["reasoning_inferred_triples"] = (
            len(state.reasoning_report.inferred_triples)
            if state.reasoning_report is not None and hasattr(state.reasoning_report, "inferred_triples")
            else 0
        )
        return counts
