from __future__ import annotations

import gc
from abc import ABC, abstractmethod

from PIL import Image


class BaseOCREngine(ABC):
    """
    Contract all OCR engines must follow.
    Subclasses must implement ocr_page() to extract text and tables from a page image.
    """

    @property
    def name(self) -> str:
        """Return the engine class name used in logs."""
        return self.__class__.__name__

    def release(self):
        """
        Unload the internal pipeline and release all GPU memory.
        Safe to call even if the engine was never loaded.
        """
        if getattr(self, "_pipeline", None) is not None:
            self._pipeline = None
            gc.collect()
            try:
                import paddle
                paddle.device.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def free_gpu():
        """
        Force garbage collection and flush the GPU memory cache.
        Called before loading a model to maximize available VRAM.
        """
        gc.collect()
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

    @abstractmethod
    def ocr_page(self, image: Image.Image) -> dict:
        """
        Process a single page image and return extracted content.

        Returns:
            dict with keys:
                - text   (str)  : plain text extracted from the page
                - tables (list) : list of dicts with 'bbox' and 'html' keys
                - raw    (any)  : raw model output for debugging
        """
