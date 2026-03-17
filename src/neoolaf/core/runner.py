"""
High-level runner responsible for preparing and executing a pipeline.
This is the natural place to manage run directories and experiment execution.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
# Standard library imports
import time

class Runner:
    """
    High-level orchestrator for a NeoOLAF pipeline execution.
    """

    def __init__(self, pipeline: Pipeline, runs_root: str = "runs", verbose: bool = False) -> None:
        """
        Args:
            pipeline:
                The pipeline instance to execute.
            runs_root:
                Root directory where execution artifacts are stored.
            verbose:
                If True, print runner-level information.
        """
        self.pipeline = pipeline
        self.runs_root = Path(runs_root)
        self.verbose = verbose

    def prepare_run_dir(self) -> Path:
        """
        Create a timestamped run directory.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_root / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the pipeline and attach an artifact directory if missing.
        """
        if state.artifact_dir is None:
            state.artifact_dir = str(self.prepare_run_dir())

        if self.verbose:
            print(f"[NeoOLAF] Run directory: {state.artifact_dir}")

        start_time = time.time()
        state = self.pipeline.run(state)
        elapsed = time.time() - start_time

        if self.verbose:
            print(f"[NeoOLAF] Total run time: {elapsed:.2f}s")

        return state