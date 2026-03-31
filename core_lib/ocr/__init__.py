"""OCR module for core-lib.

Provides OCR capabilities using dots-ocr (VLM) as primary provider
and vision-capable LLM providers as fallback.
"""

from .models import OcrResult, OcrPageResult, LayoutElement
from .dots_ocr_client import DotsOcrClient
from .ocr_service import OcrService

__all__ = [
    "OcrResult",
    "OcrPageResult",
    "LayoutElement",
    "DotsOcrClient",
    "OcrService",
]
