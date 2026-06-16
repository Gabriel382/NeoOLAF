from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentProfile:
    """Configuration object describing a document-specific extraction strategy.

    The profile is intentionally permissive: all fields remain available through
    ``config`` so future document types can add parameters without changing this
    dataclass every time.
    """

    name: str = "generic"
    config: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str | None = None) -> "DocumentProfile":
        name = str(data.get("profile_name") or data.get("name") or "generic")
        return cls(name=name, config=data, source_path=source_path)

    def get(self, path: str, default: Any = None) -> Any:
        """Read a dotted profile path, e.g. ``chunking.preferred_unit``."""
        node: Any = self.config
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def prompt_path(self, layer_name: str, default: str | None = None) -> str | None:
        """Return the configured prompt path for a layer, if any."""
        return self.get(f"prompts.{layer_name}", default)

    def layer_strategy(self, layer_name: str, default: str = "generic") -> str:
        """Return the strategy name configured for a layer."""
        return str(self.get(f"layers.{layer_name}.strategy", default))

    def allowed_relations(self) -> list[str]:
        return [str(item) for item in self.get("relations.allowed", [])]

    def field_to_relation(self) -> dict[str, str]:
        mapping = self.get("field_to_relation", {})
        return {str(k): str(v) for k, v in mapping.items()} if isinstance(mapping, dict) else {}

    def field_aliases(self) -> dict[str, list[str]]:
        aliases = self.get("table_extraction.field_aliases", {})
        if not isinstance(aliases, dict):
            return {}
        return {str(k): [str(x) for x in (v or [])] for k, v in aliases.items()}

    def preferred_extraction_unit(self) -> str:
        return str(self.get("chunking.preferred_unit_for_extraction", "chunk"))

    def to_state_dict(self) -> dict[str, Any]:
        payload = dict(self.config)
        payload.setdefault("profile_name", self.name)
        if self.source_path:
            payload.setdefault("_profile_source_path", self.source_path)
        return payload

    @property
    def root_dir(self) -> Path | None:
        if not self.source_path:
            return None
        return Path(self.source_path).resolve().parent.parent.parent
