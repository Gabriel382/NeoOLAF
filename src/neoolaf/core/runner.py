from __future__ import annotations

# Standard library imports
import copy
import gzip
import json
import os
import time
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.execution_config import ExecutionConfig
from neoolaf.core.state_merger import StateMerger


class Runner:
    """
    Execute a NeoOLAF pipeline in either:
    - document mode
    - chunk iterative mode

    This version includes:
    - real checkpoint saving/loading
    - atomic checkpoint writes
    - manifest tracking of latest checkpoint
    - optional parallel chunk execution
    """

    def __init__(
        self,
        pipeline: Pipeline,
        runs_root: str = "runs",
        verbose: bool = False,
        execution_config: ExecutionConfig | None = None,
        max_workers: int = 1,
        enable_checkpoints: bool = True,
        save_chunk_checkpoints: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.runs_root = Path(runs_root)
        self.verbose = verbose
        self.execution_config = execution_config or ExecutionConfig()
        self.state_merger = StateMerger()
        self.max_workers = max_workers
        self.enable_checkpoints = enable_checkpoints
        self.save_chunk_checkpoints = save_chunk_checkpoints

    # ---------------------------------------------------------
    # Run directory / checkpoint paths
    # ---------------------------------------------------------
    def prepare_run_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_root / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _checkpoint_dir(self, state: PipelineState) -> Path:
        ckpt_dir = Path(state.artifact_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def _manifest_path(self, state: PipelineState) -> Path:
        return self._checkpoint_dir(state) / "manifest.json"

    # ---------------------------------------------------------
    # Checkpoint save/load
    # ---------------------------------------------------------
    def save_checkpoint(self, state: PipelineState, name: str) -> Path:
        """
        Save a full PipelineState checkpoint atomically as gzipped pickle.
        Also updates manifest.json and verifies reloadability immediately.
        """
        if not self.enable_checkpoints:
            return Path("")

        ckpt_dir = self._checkpoint_dir(state)
        final_path = ckpt_dir / f"{name}.pkl.gz"
        temp_path = ckpt_dir / f"{name}.tmp.pkl.gz"

        payload = {
            "checkpoint_name": name,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": state,
        }

        # Atomic write
        with gzip.open(temp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        os.replace(temp_path, final_path)

        # Immediate verification
        _ = self.load_checkpoint(final_path)

        manifest = {
            "latest_checkpoint": str(final_path),
            "checkpoint_name": name,
            "saved_at": payload["saved_at"],
        }
        with open(self._manifest_path(state), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if self.verbose:
            print(f"[NeoOLAF] Saved checkpoint: {final_path}")

        return final_path

    @staticmethod
    def load_checkpoint(path: str | Path) -> PipelineState:
        """
        Load a PipelineState checkpoint from gzipped pickle.
        """
        path = Path(path)
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and "state" in payload:
            return payload["state"]

        if isinstance(payload, PipelineState):
            return payload

        raise ValueError(f"Checkpoint at {path} does not contain a valid PipelineState.")

    @staticmethod
    def load_latest_checkpoint(run_dir: str | Path) -> PipelineState:
        """
        Load the latest checkpoint from manifest.json inside a run directory.
        """
        run_dir = Path(run_dir)
        manifest_path = run_dir / "checkpoints" / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        checkpoint_path = manifest.get("latest_checkpoint")
        if not checkpoint_path:
            raise ValueError(f"No latest_checkpoint entry found in {manifest_path}")

        return Runner.load_checkpoint(checkpoint_path)

    # ---------------------------------------------------------
    # Public run
    # ---------------------------------------------------------
    def run(self, state: PipelineState) -> PipelineState:
        if state.artifact_dir is None:
            state.artifact_dir = str(self.prepare_run_dir())

        if self.verbose:
            print(f"[NeoOLAF] Run directory: {state.artifact_dir}")

        start_time = time.time()

        if (
            self.execution_config.mode == "chunk_iterative_mode"
            and self.execution_config.chunk_loop_enabled
        ):
            state = self._run_chunk_iterative_mode(state)
        else:
            state = self.pipeline.run(state)
            self.save_checkpoint(state, "after_full_pipeline")

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"[NeoOLAF] Total run time: {elapsed:.2f}s")

        return state

    # ---------------------------------------------------------
    # Chunk iterative mode
    # ---------------------------------------------------------
    def _run_chunk_iterative_mode(self, state: PipelineState) -> PipelineState:
        if self.verbose:
            print("[NeoOLAF] Execution mode: chunk_iterative_mode")

        preprocessing_layers = []
        chunk_layers = []
        global_layers = []

        for layer in self.pipeline.layers:
            if layer.name == "layer00_preprocessing":
                preprocessing_layers.append(layer)
            elif layer.name in self.execution_config.chunk_layer_names:
                chunk_layers.append(layer)
            elif layer.name in self.execution_config.global_layer_names:
                global_layers.append(layer)
            else:
                global_layers.append(layer)

        # ---------------------------------------------------------
        # 1. Global preprocessing
        # ---------------------------------------------------------
        preprocessing_pipeline = Pipeline(
            layers=preprocessing_layers,
            verbose=self.pipeline.verbose,
            continue_from_last=self.pipeline.continue_from_last,
        )
        preprocessed_state = preprocessing_pipeline.run(state)
        self.save_checkpoint(preprocessed_state, "after_layer00_preprocessing")

        chunks = list(preprocessed_state.document.chunks)
        if self.execution_config.max_chunks is not None:
            chunks = chunks[: self.execution_config.max_chunks]

        if self.verbose:
            print(f"[NeoOLAF] Chunk iterative mode will process {len(chunks)} chunks")
            print(f"[NeoOLAF] Parallel workers: {self.max_workers}")

        # ---------------------------------------------------------
        # 2. Chunk layers
        # ---------------------------------------------------------
        if self.max_workers <= 1:
            chunk_states = []
            for idx, chunk in enumerate(chunks, start=1):
                chunk_state = self._run_single_chunk(
                    base_state=preprocessed_state,
                    chunk=chunk,
                    chunk_layers=chunk_layers,
                    chunk_index=idx,
                    total_chunks=len(chunks),
                )
                chunk_states.append(chunk_state)
        else:
            chunk_states = self._run_chunks_in_parallel(
                base_state=preprocessed_state,
                chunks=chunks,
                chunk_layers=chunk_layers,
            )

        # ---------------------------------------------------------
        # 3. Merge
        # ---------------------------------------------------------
        merged_state = self.state_merger.merge_chunk_states(
            base_state=preprocessed_state,
            chunk_states=chunk_states,
        )
        self.save_checkpoint(merged_state, "after_chunk_merge")

        if self.verbose:
            print("[NeoOLAF] Chunk states merged into document-level state")

        # ---------------------------------------------------------
        # 4. Global layers
        # ---------------------------------------------------------
        current_state = merged_state
        if global_layers:
            for layer in global_layers:
                single_pipeline = Pipeline(
                    layers=[layer],
                    verbose=self.pipeline.verbose,
                    continue_from_last=self.pipeline.continue_from_last,
                )
                current_state = single_pipeline.run(current_state)
                self.save_checkpoint(current_state, f"after_{layer.name}")

        return current_state

    # ---------------------------------------------------------
    # Parallel chunk execution
    # ---------------------------------------------------------
    def _run_chunks_in_parallel(
        self,
        base_state: PipelineState,
        chunks: list,
        chunk_layers: list,
    ) -> list:
        chunk_results = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self._run_single_chunk,
                    base_state,
                    chunk,
                    chunk_layers,
                    idx + 1,
                    len(chunks),
                ): idx
                for idx, chunk in enumerate(chunks)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                chunk_results[idx] = future.result()

        return chunk_results

    def _run_single_chunk(
        self,
        base_state: PipelineState,
        chunk,
        chunk_layers: list,
        chunk_index: int,
        total_chunks: int,
    ) -> PipelineState:
        if self.verbose:
            print(f"[NeoOLAF] Processing chunk {chunk_index}/{total_chunks}: {chunk.chunk_id}")

        chunk_state = copy.deepcopy(base_state)
        chunk_state.document.chunks = [chunk]

        if chunk_state.artifact_dir is not None:
            chunk_dir = Path(chunk_state.artifact_dir) / "chunks" / chunk.chunk_id
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_state.artifact_dir = str(chunk_dir)

        chunk_state.linguistic_expressions = []
        chunk_state.enriched_expressions = []
        chunk_state.entity_candidates = []
        chunk_state.relation_candidates = []
        chunk_state.attribute_candidates = []
        chunk_state.event_candidates = []
        chunk_state.candidate_relation_assertions = []
        chunk_state.candidate_triples = []

        chunk_pipeline = Pipeline(
            layers=chunk_layers,
            verbose=self.pipeline.verbose,
            continue_from_last=self.pipeline.continue_from_last,
        )
        chunk_state = chunk_pipeline.run(chunk_state)

        if self.enable_checkpoints and self.save_chunk_checkpoints:
            self.save_checkpoint(chunk_state, f"after_{chunk.chunk_id}")

        return chunk_state