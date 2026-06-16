from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neoolaf.profiles.document_profile import DocumentProfile


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def default_profiles_dir() -> Path:
    return project_root_from_here() / "configs" / "document_profiles"


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                f"Profile {path} is YAML but PyYAML is not installed. "
                "Either install pyyaml or use a .json profile."
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported profile extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Profile must contain a JSON/YAML object: {path}")
    return data


def resolve_profile_path(profile: str | None = None, profile_path: str | Path | None = None) -> Path:
    """Resolve a profile name or path.

    Names are searched under ``configs/document_profiles`` with .json, .yaml,
    then .yml extensions.
    """
    if profile_path is not None:
        path = Path(profile_path)
        if not path.exists():
            raise FileNotFoundError(f"Profile file not found: {path}")
        return path

    name = profile or "generic"
    maybe_path = Path(name)
    if maybe_path.exists():
        return maybe_path

    base = default_profiles_dir()
    for suffix in (".json", ".yaml", ".yml"):
        path = base / f"{name}{suffix}"
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find document profile '{name}'. Looked in {base} for .json/.yaml/.yml."
    )


def load_document_profile(profile: str | None = None, profile_path: str | Path | None = None) -> DocumentProfile:
    path = resolve_profile_path(profile=profile, profile_path=profile_path)
    data = _load_json_or_yaml(path)
    return DocumentProfile.from_dict(data, source_path=str(path))
