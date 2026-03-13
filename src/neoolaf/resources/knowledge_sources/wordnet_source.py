from __future__ import annotations

# Standard library imports
from typing import Dict, List

# Third-party imports
from nltk.corpus import wordnet as wn


class WordNetSource:
    """
    WordNet wrapper used for lexical enrichment.
    """

    def search(self, term: str, max_synsets: int = 5) -> Dict:
        """
        Search WordNet for a term and return aliases, synonyms, lexical variants, and definitions.
        """
        synsets = wn.synsets(term)

        aliases: List[str] = []
        synonyms: List[str] = []
        lexical_variants: List[str] = []
        definitions: List[str] = []

        for syn in synsets[:max_synsets]:
            definitions.append(syn.definition())

            # Lemma names are the strongest lexical material from WordNet
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace("_", " ")
                aliases.append(lemma_name)
                synonyms.append(lemma_name)

            # Hypernyms / related forms can provide light ontology hints later if needed
            for lemma in syn.lemmas():
                lexical_variants.append(lemma.name().replace("_", " "))

        aliases = self._dedup(aliases)
        synonyms = self._dedup(synonyms)
        lexical_variants = self._dedup(lexical_variants)
        definitions = self._dedup(definitions)

        return {
            "source": "wordnet",
            "term": term,
            "aliases": aliases,
            "synonyms": synonyms,
            "lexical_variants": lexical_variants,
            "definitions": definitions,
        }

    def _dedup(self, items: List[str]) -> List[str]:
        """
        Deduplicate while preserving order and removing empty strings.
        """
        cleaned = [x.strip() for x in items if x and x.strip()]
        return list(dict.fromkeys(cleaned))