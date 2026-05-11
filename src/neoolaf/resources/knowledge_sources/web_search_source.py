from __future__ import annotations

# Standard library imports
from typing import Dict, List

# Third-party imports
from ddgs import DDGS


class WebSearchSource:
    """
    Lightweight web search wrapper using the ddgs package.
    No API key is required.
    """

    def search(self, term: str, max_results: int = 3) -> Dict:
        """
        Search the web for a term and return a few snippets.
        """
        results: List[Dict] = []

        try:
            with DDGS() as ddgs:
                for item in ddgs.text(term, max_results=max_results):
                    results.append(
                        {
                            "title": item.get("title"),
                            "body": item.get("body"),
                            "href": item.get("href"),
                        }
                    )
        except Exception:
            pass

        return {
            "source": "web",
            "term": term,
            "results": results,
        }