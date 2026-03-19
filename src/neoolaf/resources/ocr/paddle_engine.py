from __future__ import annotations

import tempfile

from PIL import Image

from neoolaf.resources.ocr.base_engine import BaseOCREngine


class PaddleOCREngine(BaseOCREngine):
    """
    OCR engine powered by PaddleOCR-VL-1.5.
    """

    def __init__(
        self,
        use_doc_orientation_classify: bool = True,
        use_doc_unwarping: bool = False,
        use_layout_detection: bool = True,
        device: str = "gpu:0",
    ):
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_layout_detection = use_layout_detection
        self.device = device
        self._pipeline = None

    def _load(self):
        """Initialize PaddleOCRVL on first use (lazy loading)."""
        if self._pipeline is not None:
            return
        self.free_gpu()
        from paddleocr import PaddleOCRVL

        self._pipeline = PaddleOCRVL(
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_doc_unwarping=self.use_doc_unwarping,
            use_layout_detection=self.use_layout_detection,
            device=self.device,
        )

    def ocr_page(self, image: Image.Image) -> dict:
        """
        Run PaddleOCR-VL on a single page.

        The image is saved to a temp file because PaddleOCRVL expects
        a file path, not a PIL Image object.

        Returns:
            dict with keys: text (str), tables (list of dicts), raw (list)
        """
        self._load()

        if image.mode != "RGB":
            image = image.convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, "PNG")

        output = list(self._pipeline.predict(tmp_path))

        text_parts = []
        table_parts = []
        for res in output:
            for block in res["parsing_res_list"]:
                content = block.content.strip()
                if not content:
                    continue
                if block.label == "table":
                    table_parts.append({"bbox": block.bbox, "html": content})
                else:
                    text_parts.append(content)

        return {
            "text": "\n".join(text_parts),
            "tables": table_parts,
            "raw": output,
        }
