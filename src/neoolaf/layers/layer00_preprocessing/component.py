from __future__ import annotations

from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.preprocessing.normalization import normalize_text
from neoolaf.preprocessing.chunking import chunk_text


class PreprocessingLayer(BaseLayer):
    """
    Layer 00: preprocessing

    Responsibilities:
    - normalize raw text
    - optionally translate text
    - create chunks
    """

    name = "layer00_preprocessing"

    def __init__(
        self,
        chunk_size: int = 1500,
        overlap: int = 200,
        translate: bool = False,
        translator=None,
        source_language: str | None = None,
        target_language: str = "en",
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            chunk_size:
                Maximum character length of each chunk.
            overlap:
                Overlap between consecutive chunks.
            translate:
                Whether to apply translation after cleaning.
            translator:
                Translator backend with a `.translate(...)` method.
            source_language:
                Optional source language hint.
            target_language:
                Target language used if translation is enabled.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Wheter to show logs or not.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.translate = translate
        self.translator = translator
        self.source_language = source_language
        self.target_language = target_language

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Execute preprocessing.
        """
        # Normalize the raw extracted text
        cleaned = normalize_text(state.document.raw_text)
        state.document.cleaned_text = cleaned

        # By default, the cleaned text is used for chunking
        text_for_chunking = cleaned

        # Optional translation step
        if self.translate:
            if self.translator is None:
                raise ValueError("Translation was requested but no translator backend was provided.")

            translated = self.translator.translate(
                cleaned,
                source_language=self.source_language,
                target_language=self.target_language,
            )

            # Store translated version in the document
            state.document.translated_text = translated
            text_for_chunking = translated
            state.log("[layer00_preprocessing] translation applied")

        # Chunk the final text version used downstream
        chunks = chunk_text(
            text_for_chunking,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        state.document.chunks = chunks
        state.log(f"[layer00_preprocessing] produced {len(chunks)} chunks")

        return state

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize relevant preprocessing outputs.
        """
        return {
            "layer": self.name,
            "doc_id": state.document.doc_id,
            "cleaned_text_preview": (state.document.cleaned_text or "")[:1000],
            "translated_text_preview": (state.document.translated_text or "")[:1000],
            "num_chunks": len(state.document.chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "text_preview": c.text[:300],
                }
                for c in state.document.chunks
            ],
        }