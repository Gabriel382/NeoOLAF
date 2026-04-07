from __future__ import annotations

# Standard library imports
import copy
import time
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

    This version supports parallel chunk execution in chunk_iterative_mode.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        runs_root: str = "runs",
        verbose: bool = False,
        execution_config: ExecutionConfig | None = None,
        max_workers: int = 1,
    ) -> None:
        """
        Initialize the runner.

        Args:
            pipeline:
                The pipeline instance to execute.
            runs_root:
                Root directory where execution artifacts are stored.
            verbose:
                Whether to print runner-level progress.
            execution_config:
                Optional execution configuration.
            max_workers:
                Number of workers for parallel chunk execution.
                Use 1 for sequential mode.
        """
        self.pipeline = pipeline
        self.runs_root = Path(runs_root)
        self.verbose = verbose
        self.execution_config = execution_config or ExecutionConfig()
        self.state_merger = StateMerger()
        self.max_workers = max_workers

    def prepare_run_dir(self) -> Path:
        """
        Create a run directory.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_root / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the pipeline according to the configured execution mode.
        """
        if state.artifact_dir is None:
            state.artifact_dir = str(self.prepare_run_dir())

        if self.verbose:
            print(f"[NeoOLAF] Run directory: {state.artifact_dir}")

        start_time = time.time()

        # ---------------------------------------------------------
        # Dispatch by execution mode
        # ---------------------------------------------------------
        if (
            self.execution_config.mode == "chunk_iterative_mode"
            and self.execution_config.chunk_loop_enabled
        ):
            state = self._run_chunk_iterative_mode(state)
        else:
            state = self.pipeline.run(state)

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"[NeoOLAF] Total run time: {elapsed:.2f}s")

        return state

    def _run_chunk_iterative_mode(self, state: PipelineState) -> PipelineState:
        """
        Execute the pipeline in chunk iterative mode.

        Strategy:
        1. run preprocessing globally to obtain chunks
        2. run selected layers independently per chunk
        3. merge chunk-level states
        4. run remaining layers globally
        """
        if self.verbose:
            print("[NeoOLAF] Execution mode: chunk_iterative_mode")

        # Split pipeline into:
        # - preprocessing layers
        # - chunk layers
        # - global layers
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
                # Fallback:
                # if not explicitly configured, run globally
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

        chunks = list(preprocessed_state.document.chunks)
        if self.execution_config.max_chunks is not None:
            chunks = chunks[: self.execution_config.max_chunks]

        if self.verbose:
            print(f"[NeoOLAF] Chunk iterative mode will process {len(chunks)} chunks")
            print(f"[NeoOLAF] Parallel workers: {self.max_workers}")

        # ---------------------------------------------------------
        # 2. Run selected chunk layers per chunk
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
        # 3. Merge chunk-level states into one document-level state
        # ---------------------------------------------------------
        merged_state = self.state_merger.merge_chunk_states(
            base_state=preprocessed_state,
            chunk_states=chunk_states,
        )

        if self.verbose:
            print("[NeoOLAF] Chunk states merged into document-level state")

        # ---------------------------------------------------------
        # 4. Run global layers after aggregation
        # ---------------------------------------------------------
        if global_layers:
            global_pipeline = Pipeline(
                layers=global_layers,
                verbose=self.pipeline.verbose,
                continue_from_last=self.pipeline.continue_from_last,
            )
            merged_state = global_pipeline.run(merged_state)

        return merged_state

    def _run_chunks_in_parallel(
        self,
        base_state: PipelineState,
        chunks: list,
        chunk_layers: list,
    ) -> list:
        """
        Run chunk subpipelines in parallel and preserve original chunk order.
        """
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
        """
        Run the chunk-layer subpipeline for one chunk.
        """
        if self.verbose:
            print(
                f"[NeoOLAF] Processing chunk {chunk_index}/{total_chunks}: {chunk.chunk_id}"
            )

        # Deep copy the preprocessed document-level state
        chunk_state = copy.deepcopy(base_state)

        # Restrict the document chunks to the current chunk only
        chunk_state.document.chunks = [chunk]

        # Give each chunk its own artifact subdirectory to avoid write collisions
        if chunk_state.artifact_dir is not None:
            chunk_dir = Path(chunk_state.artifact_dir) / "chunks" / chunk.chunk_id
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_state.artifact_dir = str(chunk_dir)

        # Reset chunk-level outputs so they do not carry over
        chunk_state.linguistic_expressions = []
        chunk_state.enriched_expressions = []
        chunk_state.entity_candidates = []
        chunk_state.relation_candidates = []
        chunk_state.attribute_candidates = []
        chunk_state.event_candidates = []
        chunk_state.candidate_relation_assertions = []
        chunk_state.candidate_triples = []

        # Run the chunk-layer subpipeline
        chunk_pipeline = Pipeline(
            layers=chunk_layers,
            verbose=self.pipeline.verbose,
            continue_from_last=self.pipeline.continue_from_last,
        )
        chunk_state = chunk_pipeline.run(chunk_state)

        return chunk_state