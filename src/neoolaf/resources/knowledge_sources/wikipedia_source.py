from __future__ import annotations

# Standard library imports
from typing import Dict, List

# Third-party imports
import wikipediaapi


class WikipediaSource:
    """
    Wikipedia wrapper for enrichment.
    Uses wikipedia-api for more stable page access.
    """

    def __init__(self, language: str = "en") -> None:
        """
        Initialize the Wikipedia API client.
        """
        self.language = language
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent="NeoOLAF/0.1 (research prototype)"
        )

    def search(self, term: str) -> Dict:
        """
        Retrieve a Wikipedia page directly from the term and return structured evidence.
        """
        page = self.wiki.page(term)

        if not page.exists():
            return {
                "source": "wikipedia",
                "term": term,
                "found": False,
                "aliases": [],
                "summary": "",
                "url": None,
            }

        # Section titles and redirects are not easily exposed as aliases here,
        # but the title itself can still serve as lexical evidence.
        aliases: List[str] = [page.title]

        summary = page.summary[:1200] if page.summary else ""

        return {
            "source": "wikipedia",
            "term": term,
            "found": True,
            "title": page.title,
            "aliases": self._dedup(aliases),
            "summary": summary,
            "url": page.fullurl,
        }

    def _dedup(self, items: List[str]) -> List[str]:
        """
        Deduplicate strings while preserving order.
        """
        cleaned = [x.strip() for x in items if x and x.strip()]
        return list(dict.fromkeys(cleaned))