"""Very light post-processing stage, inspired by the original repository."""

from __future__ import annotations

import argparse
from utils import load_json_file, save_json_file


def main() -> None:
    """Merge and deduplicate records across chunks."""
    parser = argparse.ArgumentParser(description="Post-process TaxoDrivenKG-style output.")
    parser.add_argument("input_json", type=str, help="Path to the chunk-level output JSON.")
    parser.add_argument("output_json", type=str, help="Path to save the merged JSON.")
    args = parser.parse_args()
    preds = load_json_file(args.input_json)
    merged = {"entities": [], "relationships": []}
    seen_entities = set()
    seen_relationships = set()
    for _, chunk_pred in preds.items():
        for entity in chunk_pred.get("entities", []):
            key = (entity.get("name", "").lower(), entity.get("label", "").lower())
            if key not in seen_entities:
                seen_entities.add(key)
                merged["entities"].append(entity)
        for relation in chunk_pred.get("relationships", []):
            key = (relation.get("source", "").lower(), relation.get("target", "").lower(), relation.get("relation", "").lower())
            if key not in seen_relationships:
                seen_relationships.add(key)
                merged["relationships"].append(relation)
    save_json_file(args.output_json, merged)
    print(f"Saved merged output to {args.output_json}")


if __name__ == "__main__":
    main()
