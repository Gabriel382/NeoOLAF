"""Utility helpers kept close in spirit to the original repository."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

text_template = "<heading>{}</heading>\n{}\n"


def load_json_file(file_path: str | Path) -> Dict[str, Any]:
    """Load and return a JSON document."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(file_path: str | Path, data: Any) -> None:
    """Save JSON with UTF-8 and indentation."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def construct_regex(keyword: str) -> str:
    """Construct a permissive regex from a keyword, close to the original code."""
    parts = re.split(r"[^a-zA-Z0-9]", keyword)
    parts = [p for p in parts if len(p) > 0]
    out = "[^a-zA-Z0-9]*".join(parts)
    return r"\b" + out + r"\b"


def remove_overlapping_mentions(mentions: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Remove overlapping mentions by preferring longer spans."""
    mentions.sort(key=lambda x: x[1] - x[0], reverse=True)
    output: List[Tuple[int, int, str]] = []
    for mention in mentions:
        if not any(mention[0] <= kept[1] and mention[1] >= kept[0] for kept in output):
            output.append(mention)
    return output


def remove_common_keys(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Remove all items from dict1 whose key also appears in dict2."""
    return {key: dict1[key] for key in dict1 if key not in dict2}
