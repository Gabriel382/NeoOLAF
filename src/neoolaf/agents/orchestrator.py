from __future__ import annotations

"""Minimal NeoOLAF layer orchestrator.

The orchestrator currently wraps the existing Runner/Pipeline stack.  Its role is
not to add a complex multi-agent policy yet, but to make orchestration explicit:
loading an execution plan, keeping a shared artifact store, running selected
layers, and exposing a place where feedback/retry policies can be added later.
"""

import json
import time
from pathlib import Path
from typing import Any

from neoolaf.core.execution_plan import ExecutionPlan
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner


class LayerOrchestrator:
    """Simple orchestrator around Runner for ablation and agentic evolution."""

    def __init__(self, runner: Runner, plan: ExecutionPlan, verbose: bool = False) -> None:
        self.runner = runner
        self.plan = plan
        self.verbose = verbose

    def run(
        self,
        state: PipelineState,
        *,
        run_dir: str | Path,
        resume_from: str | Path | None = None,
    ) -> PipelineState:
        """Execute the configured plan and save a small orchestration manifest."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_manifest(run_dir, status="started")

        started = time.time()
        if self.verbose:
            print("[NeoOLAF][Orchestrator] Starting execution plan")
            print(json.dumps(self.plan.to_dict(), indent=2, ensure_ascii=False))

        final_state = self.runner.run(
            state,
            from_layer=self.plan.from_layer,
            to_layer=self.plan.to_layer,
            skip_layers=self.plan.skip_layers,
            resume_from=resume_from,
            run_dir=run_dir,
        )

        elapsed = time.time() - started
        self._save_manifest(run_dir, status="completed", elapsed_seconds=round(elapsed, 3))
        if self.verbose:
            print(f"[NeoOLAF][Orchestrator] Completed in {elapsed:.2f}s")
        return final_state

    def _save_manifest(
        self,
        run_dir: Path,
        *,
        status: str,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Persist a lightweight orchestration manifest for reproducibility."""
        manifest: dict[str, Any] = {
            "status": status,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_plan": self.plan.to_dict(),
        }
        if elapsed_seconds is not None:
            manifest["elapsed_seconds"] = elapsed_seconds
        (run_dir / "orchestration_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
