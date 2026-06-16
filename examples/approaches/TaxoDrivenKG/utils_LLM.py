"""Chunking and extraction logic adapted closely from the original TaxoDrivenKG repository."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import tiktoken

from consts import PATH
from prompt_templates import PROMPT_TEMPLATE, PROMPT_TEMPLATE_NO_RAG, PROMPT_TEMPLATE_ZERO_SHOT
from utils import load_json_file
from backends.base import BaseChatBackend


delimiters = {
    "section_delimiter": "-",
    "tuple_delimiter": "<|>",
    "completion_delimiter": "<|COMPLETE|>",
    "record_delimiter": "##",
}


class TextChunker:
    """Token chunker kept very close to the original implementation."""

    def __init__(self, text_max_tokens: int = 600):
        self.text_max_tokens = text_max_tokens

    def _chunk_text(self, text: str, max_tokens: int, tokenizer: Optional[tiktoken.Encoding] = None, chunk_overlap: int = 100) -> Tuple[List[str], List[int]]:
        """Chunk text by token length with overlap."""
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        splits: List[str] = []
        input_ids = tokenizer.encode(text)
        start_idx = 0
        start_chars: List[int] = []
        cur_idx = min(start_idx + max_tokens, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            splits.append(tokenizer.decode(chunk_ids))
            if start_idx == 0:
                start_chars.append(0)
            else:
                start_chars.append(len(tokenizer.decode(input_ids[:start_idx])))
            start_idx += max_tokens - chunk_overlap
            cur_idx = min(start_idx + max_tokens, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
        return splits, start_chars

    def get_text_chunks(self, text: str) -> Tuple[List[str], List[int]]:
        """Chunk raw text into overlapping token windows."""
        return self._chunk_text(text, max_tokens=self.text_max_tokens, chunk_overlap=100)


class InfoExtractor:
    """TaxoDriven-style extractor with pluggable backends."""

    def __init__(self, backend: BaseChatBackend, model_name: str, exp: str = "base", n_shot: int = 3, few_shot_path: str | None = None):
        self.backend = backend
        self.model_name = model_name
        if exp == "0_shot":
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE_ZERO_SHOT
        elif exp == "no_rag":
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE_NO_RAG
        else:
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE
        if "_shot" in exp:
            n_shot = int(exp.split("_")[0])
        examples_path = few_shot_path or PATH["few_shot"]
        self.formatted_examples = ""
        try:
            examples = load_json_file(examples_path)
        except FileNotFoundError:
            examples = []
        for i, example in enumerate(examples[:n_shot]):
            self.formatted_examples += f"\nExample {i+1}:\n{example}"

    def parse_response(self, response: str, with_description: bool = True) -> Dict[str, List[Dict[str, str]]]:
        """Parse tuple-formatted records into structured dicts."""
        out: Dict[str, List[Dict[str, str]]] = {"entities": [], "relationships": []}
        start_index = response.find('("')
        if start_index > 0:
            response = response[start_index:]
        response = response.split(delimiters["completion_delimiter"])[0]
        response = response.split(delimiters["record_delimiter"])
        response = [r.lstrip("\n").rstrip("\n").lstrip("(").rstrip(")") for r in response]
        rows = [re.split(r"<\s*\|\s*>", r) for r in response if r.strip()]
        for row in rows:
            if not row:
                continue
            tag = row[0].strip().lower()
            if "entity" in tag:
                if with_description and len(row) == 4:
                    out["entities"].append({"name": row[1].strip(), "label": row[2].strip(), "description": row[3].strip()})
                elif (not with_description) and len(row) == 3:
                    out["entities"].append({"name": row[1].strip(), "label": row[2].strip()})
            elif "relationship" in tag:
                if len(row) == 4:
                    out["relationships"].append({"source": row[1].strip(), "target": row[2].strip(), "relation": row[3].strip()})
        return out

    def run(self, text: str, retrieved_nodes: Dict[str, List[str | float | None]]) -> Tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
        """Run extraction on one chunk."""
        potential_entities = ", ".join(list(retrieved_nodes.keys()))
        prompt = self.PROMPT_TEMPLATE.format(
            **delimiters,
            formatted_examples=self.formatted_examples,
            input_text=text.replace("{", "").replace("}", ""),
            potential_entities=potential_entities,
        ).format(**delimiters)
        conversation = [{"role": "user", "content": prompt}]
        pred_content = self.backend.chat(messages=conversation, model_name=self.model_name, temperature=0.0, max_tokens=2048)
        conversation.append({"role": "assistant", "content": pred_content})
        return self.parse_response(pred_content), conversation
