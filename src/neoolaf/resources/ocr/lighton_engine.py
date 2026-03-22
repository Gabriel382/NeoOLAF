from __future__ import annotations

import gc
import re

import torch
from PIL import Image

from neoolaf.resources.ocr.base_engine import BaseOCREngine


class LightOnOCREngine(BaseOCREngine):
    """
    OCR engine powered by LightOnOCR-2-1B.
    """
    MODEL_ID = "lightonai/LightOnOCR-2-1B"
    MAX_PX_SIDE = 1540

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

    def _load(self):
        """Download and initialize the model on first use."""
        if self._model is not None:
            return
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        from transformers import (
            LightOnOcrForConditionalGeneration,
            LightOnOcrProcessor,
        )

        self._processor = LightOnOcrProcessor.from_pretrained(self.MODEL_ID)
        self._model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self._dtype,
            device_map="auto",
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def release(self):
        self._model = None
        self._processor = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _resize_for_model(image: Image.Image) -> Image.Image:
        w, h = image.size
        longest = max(w, h)
        if longest <= LightOnOCREngine.MAX_PX_SIDE:
            return image
        scale = LightOnOCREngine.MAX_PX_SIDE / longest
        return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    @staticmethod
    def _markdown_table_to_html(md_table: str) -> str:
        lines = [l.strip() for l in md_table.strip().splitlines() if l.strip()]
        rows = [l for l in lines if not re.match(r"^\|[-| :]+\|$", l)]
        if not rows:
            return ""
        html = ["<table>"]
        for r_idx, row in enumerate(rows):
            cells = [c.strip() for c in row.strip("|").split("|") if c.strip()]
            tag = "th" if r_idx == 0 else "td"
            html.append(
                "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
            )
        html.append("</table>")
        return "\n".join(html)

    @staticmethod
    def _parse_tables(text: str) -> tuple:
        tables = []

        html_pattern = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
        for match in html_pattern.finditer(text):
            tables.append({"bbox": None, "html": match.group(0)})
        plain_text = html_pattern.sub("", text).strip()

        md_pattern = re.compile(
            r"(\|.+\|\n(?:\|[-| :]+\|\n)(?:\|.+\|\n?)+)", re.MULTILINE
        )
        for match in md_pattern.finditer(plain_text):
            html = LightOnOCREngine._markdown_table_to_html(match.group(1))
            if html:
                tables.append({"bbox": None, "html": html})
        plain_text = md_pattern.sub("", plain_text).strip()

        return plain_text, tables

    def ocr_page(self, image: Image.Image) -> dict:
        self._load()

        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self._resize_for_model(image)

        conversation = [
            {"role": "user", "content": [{"type": "image", "image": image}]}
        ]

        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(device=self._device, dtype=self._dtype)
            if v.is_floating_point()
            else v.to(self._device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=4096, do_sample=False
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0, prompt_len:]
        raw_text = self._processor.decode(generated, skip_special_tokens=True)

        plain_text, tables = self._parse_tables(raw_text)

        return {"text": plain_text, "tables": tables, "raw": raw_text}
