from __future__ import annotations

from typing import List, Sequence

import easyocr


class OCRRunner:
    """Small wrapper around EasyOCR so the rest of the app stays clean."""

    def __init__(self, languages: Sequence[str] | None = None, gpu: bool = False):
        self.languages = list(languages or ["en"])
        self.gpu = gpu
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)

    def extract_text(self, image) -> List[str]:
        """
        Return OCR lines in reading order.

        detail=0 keeps the output simple: a list of detected text fragments.
        """
        results = self.reader.readtext(
            image,
            detail=0,
            paragraph=False,
            min_size=8,
            text_threshold=0.4,
            low_text=0.3,
            link_threshold=0.4,
        )
        return [str(item).strip() for item in results if str(item).strip()]
