"""OCR module for core-lib.

Provides OCR capabilities using vision-capable LLM providers as the primary
mechanism, with dots-ocr (VLM) available as an optional fallback provider.
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
