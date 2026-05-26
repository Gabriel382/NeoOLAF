from __future__ import annotations

"""Optional file-based prompt loader.

NeoOLAF layers can keep their current Python prompt builders while ablation
runs inspect actual prompts through prompt capture.  This helper adds a stable
place for future prompt externalization: if a prompt file exists under the
project-level `prompts/` directory, it can override or document a layer prompt
without changing package code again.
"""

from pathlib import Path
from string import Template
from typing import Any


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def default_prompts_dir() -> Path:
    return project_root_from_here() / "prompts"


def load_prompt_template(name: str, *, prompts_dir: str | Path | None = None, fallback: str = "") -> str:
    """Load a prompt template by name, returning fallback if it does not exist."""
    base = Path(prompts_dir) if prompts_dir is not None else default_prompts_dir()
    path = base / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def render_prompt_template(template: str, **kwargs: Any) -> str:
    """Render a simple `$variable` prompt template."""
    safe_kwargs = {key: "" if value is None else str(value) for key, value in kwargs.items()}
    return Template(template).safe_substitute(**safe_kwargs)


def prompt_size_stats(prompt: str) -> dict[str, int]:
    """Return character and rough token counts for a prompt."""
    return {
        "prompt_chars": len(prompt),
        "estimated_tokens": max(1, len(prompt) // 4),
    }
