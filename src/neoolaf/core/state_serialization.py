from __future__ import annotations

"""
JSON-safe serialization helpers for NeoOLAF pipeline states.

This module is intentionally lightweight and dependency-free.  It stores the
fully qualified class name of each dataclass so that a saved layer state can be
loaded again in a later run.
"""

from dataclasses import fields, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any
import json


_CLASS_KEY = "__neoolaf_class__"


def _class_path(obj: Any) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_class(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and common Python objects into JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            _CLASS_KEY: _class_path(value),
            **{field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)},
        }

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, Path):
        return str(value)

    # Keep primitive values as-is.
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Last-resort fallback. This prevents JSON export failures for unexpected
    # small helper objects while making the lossy conversion explicit.
    return str(value)


def from_jsonable(value: Any) -> Any:
    """Rebuild dataclasses saved by :func:`to_jsonable`."""
    if isinstance(value, list):
        return [from_jsonable(item) for item in value]

    if isinstance(value, dict):
        class_name = value.get(_CLASS_KEY)
        if class_name:
            cls = _import_class(class_name)
            kwargs = {
                key: from_jsonable(item)
                for key, item in value.items()
                if key != _CLASS_KEY
            }
            return cls(**kwargs)

        return {key: from_jsonable(item) for key, item in value.items()}

    return value


def dump_json(path: str | Path, data: Any) -> None:
    """Write a JSON file with UTF-8 encoding and stable indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> Any:
    """Read a JSON file and rebuild dataclass objects when possible."""
    path = Path(path)
    return from_jsonable(json.loads(path.read_text(encoding="utf-8")))
