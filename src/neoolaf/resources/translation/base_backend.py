from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTranslationBackend(ABC):
    """
    Contract all translation backends must follow.

    Subclasses must implement translate() to convert text from a source
    language to a target language.
    """

    @property
    def name(self) -> str:
        """Return the backend class name used in logs."""
        return self.__class__.__name__

    @abstractmethod
    def translate(
        self,
        text: str,
        source_language: str | None = None,
        target_language: str = "en",
    ) -> str:
        """
        Translate text from source_language to target_language.

        Args:
            text:
                Input text to translate.
            source_language:
                Source language code (e.g. "fr", "de", "it").
                If None, the backend should auto-detect or default.
            target_language:
                Target language code, default "en".

        Returns:
            The translated text.
        """
