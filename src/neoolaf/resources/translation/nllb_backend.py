from __future__ import annotations

import re
from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from langdetect import detect

from neoolaf.resources.translation.base_backend import BaseTranslationBackend


def detect_language(text: str) -> str | None:
    """Auto-detect source language from text using langdetect."""
    try:
        sample = text[:2000]
        return detect(sample)
    except Exception:
        return None


LANG_MAP = {
    "auto": None,
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "es": "spa_Latn",
}

def _resolve_lang(code: str | None) -> str | None:
    if code is None:
        return None
    code = code.strip().lower()
    if code in LANG_MAP:
        return LANG_MAP[code]
    if "_" in code and len(code) >= 7:
        return code
    return None


class NLLB200TranslatorBackend(BaseTranslationBackend):
    """
    Translation backend using Meta's NLLB-200 model via Hugging Face.
    Supports 200 languages with a single model. Runs locally on GPU or CPU.
    """
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str | None = None,
        max_length: int = 512,
        max_chars: int = 4000,
    ) -> None:
        
        self.model_name = model_name
        self.max_length = max_length
        self.max_chars = max_chars

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def translate(
        self,
        text: str,
        source_language: str | None = None,
        target_language: str = "en",
    ) -> str:
 
        if not text or not text.strip():
            return text

        if source_language is None:
            source_language = _detect_language(text)

        src_lang = _resolve_lang(source_language) or "eng_Latn"
        tgt_lang = _resolve_lang(target_language) or "eng_Latn"

        if src_lang == tgt_lang:
            return text

        self.tokenizer.src_lang = src_lang

        segments = self._split_text(text)
        translated_parts: List[str] = []

        for segment in segments:
            try:
                translated_parts.append(self._translate_segment(segment, tgt_lang))
            except Exception:
                translated_parts.append(segment)

        return "\n".join(translated_parts)

    def _translate_segment(self, text: str, tgt_lang: str) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True
        ).to(self.device)

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=self.max_length,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _split_text(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        segments: List[str] = []
        current = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) <= self.max_chars:
                tentative = f"{current}\n\n{paragraph}".strip() if current else paragraph
                if len(tentative) <= self.max_chars:
                    current = tentative
                else:
                    if current:
                        segments.append(current)
                    current = paragraph
            else:
                if current:
                    segments.append(current)
                    current = ""
                for chunk in self._split_long(paragraph):
                    segments.append(chunk)

        if current:
            segments.append(current)
        return segments

    def _split_long(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        segments: List[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= self.max_chars:
                tentative = f"{current} {sentence}".strip() if current else sentence
                if len(tentative) <= self.max_chars:
                    current = tentative
                else:
                    if current:
                        segments.append(current)
                    current = sentence
            else:
                if current:
                    segments.append(current)
                    current = ""
                for i in range(0, len(sentence), self.max_chars):
                    segments.append(sentence[i : i + self.max_chars])

        if current:
            segments.append(current)
        return segments
