from __future__ import annotations

# Standard library imports
from typing import Dict, List, Any

# Third-party imports
import requests


class WikidataSource:
    """
    Wikidata wrapper with two-step retrieval:
    1. search entities
    2. fetch full entity data for aliases, labels, and descriptions
    """

    def __init__(self, language: str = "en", timeout: int = 30) -> None:
        self.language = language
        self.timeout = timeout
        self.api_url = "https://www.wikidata.org/w/api.php"

    def search(self, term: str, limit: int = 3) -> Dict:
        """
        Search Wikidata and enrich top hits with aliases and labels.
        """
        entity_ids = self._search_entity_ids(term, limit=limit)
        full_results: List[Dict[str, Any]] = []

        for entity_id in entity_ids:
            entity_data = self._fetch_entity_data(entity_id)
            if entity_data is not None:
                full_results.append(entity_data)

        # Aggregate lexical material across the returned entities
        aliases: List[str] = []
        labels: List[str] = []
        descriptions: List[str] = []

        for item in full_results:
            aliases.extend(item.get("aliases", []))
            labels.extend(item.get("labels", []))
            if item.get("description"):
                descriptions.append(item["description"])

        return {
            "source": "wikidata",
            "term": term,
            "results": full_results,
            "aliases": self._dedup(aliases),
            "labels": self._dedup(labels),
            "descriptions": self._dedup(descriptions),
        }

    def _search_entity_ids(self, term: str, limit: int = 3) -> List[str]:
        """
        Search entity IDs using Wikidata wbsearchentities.
        """
        params = {
            "action": "wbsearchentities",
            "search": term,
            "language": self.language,
            "format": "json",
            "limit": limit,
        }

        try:
            response = requests.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return [item["id"] for item in data.get("search", []) if "id" in item]
        except Exception:
            return []

    def _fetch_entity_data(self, entity_id: str) -> Dict[str, Any] | None:
        """
        Fetch full entity data, including aliases, labels, and descriptions.
        """
        params = {
            "action": "wbgetentities",
            "ids": entity_id,
            "languages": self.language,
            "format": "json",
            "props": "labels|descriptions|aliases",
        }

        try:
            response = requests.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            entity = data.get("entities", {}).get(entity_id, {})
            if not entity:
                return None

            labels = []
            aliases = []
            description = None

            label_obj = entity.get("labels", {}).get(self.language)
            if label_obj and "value" in label_obj:
                labels.append(label_obj["value"])

            alias_objs = entity.get("aliases", {}).get(self.language, [])
            for alias_obj in alias_objs:
                value = alias_obj.get("value")
                if value:
                    aliases.append(value)

            desc_obj = entity.get("descriptions", {}).get(self.language)
            if desc_obj and "value" in desc_obj:
                description = desc_obj["value"]

            return {
                "id": entity_id,
                "labels": self._dedup(labels),
                "aliases": self._dedup(aliases),
                "description": description,
                "url": f"https://www.wikidata.org/wiki/{entity_id}",
            }
        except Exception:
            return None

    def _dedup(self, items: List[str]) -> List[str]:
        """
        Deduplicate strings while preserving order.
        """
        cleaned = [x.strip() for x in items if x and x.strip()]
        return list(dict.fromkeys(cleaned))