from __future__ import annotations


class SimpleTranslator:
    """
    Minimal translator interface.

    For now this is only a placeholder so that the preprocessing
    layer can optionally call a translator backend later.
    """

    def translate(
        self,
        text: str,
        source_language: str | None = None,
        target_language: str = "en",
    ) -> str:
        """
        Translate text from source_language to target_language.

        This method must be replaced by a real backend later.
        """
        raise NotImplementedError("Translation backend not implemented yet.")