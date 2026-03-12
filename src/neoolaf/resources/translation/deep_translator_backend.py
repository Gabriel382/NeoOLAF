from __future__ import annotations

# Standard library imports
import re
from typing import List

# External translation backend
from deep_translator import GoogleTranslator


class DeepTranslatorBackend:
    """
    Simple translation backend using deep-translator.

    This backend is:
    - free to use
    - easy to integrate
    - suitable for quick preprocessing experiments

    Important:
    deep-translator / GoogleTranslator cannot safely handle very long texts
    in a single request, so this backend splits the input into smaller parts.
    """

    def __init__(self, max_chars: int = 4500) -> None:
        """
        Initialize the translator backend.

        Args:
            max_chars:
                Maximum number of characters per translation segment.
                Kept below 5000 to avoid backend length errors.
        """
        self.max_chars = max_chars

    def translate(
        self,
        text: str,
        source_language: str | None = None,
        target_language: str = "en",
    ) -> str:
        """
        Translate text into the target language.

        Args:
            text:
                Input text to translate.
            source_language:
                Optional source language code such as 'fr', 'pt', 'de'.
                If None, automatic detection is used.
            target_language:
                Target language code, default is English.

        Returns:
            The translated text.
        """
        # Guard against empty input
        if not text or not text.strip():
            return text

        # Resolve source language
        src = source_language if source_language is not None else "auto"

        # Build translator
        translator = GoogleTranslator(source=src, target=target_language)

        # Split the text into safe segments
        segments = self._split_text(text, max_chars=self.max_chars)

        # Translate each segment independently
        translated_segments: List[str] = []
        for segment in segments:
            translated_segment = translator.translate(segment)
            translated_segments.append(translated_segment)

        # Merge translated segments back into a single text
        return "\n".join(translated_segments)

    def _split_text(self, text: str, max_chars: int) -> List[str]:
        """
        Split text into translation-safe segments.

        Strategy:
        1. split by paragraph boundaries when possible
        2. if a paragraph is still too long, split by sentences
        3. if a sentence is still too long, split by raw character windows

        Args:
            text:
                Full text to split.
            max_chars:
                Maximum length per segment.

        Returns:
            List of segments safe for translation.
        """
        # Normalize paragraph boundaries first
        paragraphs = re.split(r"\n\s*\n", text)

        segments: List[str] = []
        current = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If the paragraph fits, try to append it to the current segment
            if len(paragraph) <= max_chars:
                tentative = f"{current}\n\n{paragraph}".strip() if current else paragraph
                if len(tentative) <= max_chars:
                    current = tentative
                else:
                    if current:
                        segments.append(current)
                    current = paragraph
            else:
                # Flush current if needed before handling the large paragraph
                if current:
                    segments.append(current)
                    current = ""

                # Split the large paragraph further
                paragraph_segments = self._split_large_paragraph(paragraph, max_chars)
                segments.extend(paragraph_segments)

        # Flush remaining current segment
        if current:
            segments.append(current)

        return segments

    def _split_large_paragraph(self, paragraph: str, max_chars: int) -> List[str]:
        """
        Split a paragraph that is too long.

        First try sentence splitting. If sentences are still too long,
        fall back to raw character slicing.
        """
        sentence_candidates = re.split(r"(?<=[.!?])\s+", paragraph)

        segments: List[str] = []
        current = ""

        for sentence in sentence_candidates:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) <= max_chars:
                tentative = f"{current} {sentence}".strip() if current else sentence
                if len(tentative) <= max_chars:
                    current = tentative
                else:
                    if current:
                        segments.append(current)
                    current = sentence
            else:
                # Flush accumulated sentence group
                if current:
                    segments.append(current)
                    current = ""

                # Hard split oversized sentence
                hard_splits = self._hard_split(sentence, max_chars)
                segments.extend(hard_splits)

        if current:
            segments.append(current)

        return segments

    def _hard_split(self, text: str, max_chars: int) -> List[str]:
        """
        Last-resort split for very long text spans.

        This simply slices the text into fixed-width character windows.
        """
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]