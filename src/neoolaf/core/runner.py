"""
High-level runner responsible for preparing and executing a pipeline.
This is the natural place to manage run directories and experiment execution.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState


class Runner:
    """
    High-level orchestrator for a NeoOLAF pipeline execution.
    """

    def __init__(self, pipeline: Pipeline, runs_root: str = "runs") -> None:
        """
        Args:
            pipeline:
                The pipeline instance to execute.
            runs_root:
                Root directory where execution artifacts are stored.
        """
        self.pipeline = pipeline
        self.runs_root = Path(runs_root)

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

        return self.pipeline.run(state)