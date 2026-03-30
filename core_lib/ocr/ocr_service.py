"""OCR service with vision-LLM primary provider and optional dots-ocr fallback.

Processes images primarily through a vision-capable LLM (e.g. Gemini,
Configured via ``llm_providers.yaml`` with ``usage: [vision, ocr]``).
dots-ocr is an optional secondary provider activated by setting
``DOTS_OCR_BASE_URL``.  Uses Redis/Valkey cache for image
deduplication — identical images are only OCR'd once.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from .. import get_module_logger
from ..config.ocr_settings import OcrSettings
from .dots_ocr_client import DotsOcrClient, DEFAULT_MODE
from .models import LayoutElement, OcrPageResult, OcrResult

logger = get_module_logger()


class OcrService:
    """Orchestrates OCR processing with caching and provider selection.

    Primary provider: vision-capable LLM (passed as ``vision_llm_client``,
    typically configured via ``llm_providers.yaml`` with
    ``usage: [vision, ocr]``).

    Optional secondary provider: dots-ocr — only attempted when
    ``OcrSettings.dots_ocr_base_url`` is set.  Acts as a high-performance
    specialist fallback when the vision LLM is unavailable.

    Usage::

        from core_lib.config import OcrSettings
        from core_lib.ocr import OcrService

        settings = OcrSettings.from_env()
        service = OcrService(settings, vision_llm_client=my_gemini_client)

        result = service.process_images(images, mime_types=["image/png"])
    """

    def __init__(
        self,
        settings: OcrSettings,
        *,
        vision_llm_client: Optional[Any] = None,
        cache_client: Optional[Any] = None,
    ) -> None:
        """
        Args:
            settings: OCR configuration.
            vision_llm_client: LLMClient configured for a vision model
                (e.g. Gemini). This is the **primary** OCR provider.
                When not provided, falls back to dots-ocr (if configured).
            cache_client: Optional cache client (``BaseCache``) for deduplication.
                When provided, OCR results for identical images are cached.
        """
        self._settings = settings
        self._vision_client = vision_llm_client
        self._cache = cache_client
        self._cache_prefix = "ocr:"

        # Lazily initialised dots-ocr client
        self._dots_client: Optional[DotsOcrClient] = None
        self._dots_healthy = True
        self._dots_last_check = 0.0
        self._health_check_interval = 60.0  # seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_images(
        self,
        images: List[bytes],
        *,
        mime_types: Optional[List[str]] = None,
        start_page: int = 1,
        mode: str = DEFAULT_MODE,
    ) -> OcrResult:
        """Process a batch of images sequentially.

        Args:
            images: List of raw image byte buffers.
            mime_types: Parallel list of MIME types (defaults to image/png).
            start_page: Page number offset for the first image.
            mode: dots-ocr prompt mode.

        Returns:
            Aggregated ``OcrResult``.
        """
        result = OcrResult()
        if not images:
            return result

        if mime_types is None:
            mime_types = ["image/png"] * len(images)

        for idx, (img_bytes, mime) in enumerate(zip(images, mime_types)):
            page_num = start_page + idx

            # --- size filter ---
            if not self._passes_filter(img_bytes):
                result.total_images_skipped += 1
                continue

            # --- cache lookup ---
            cache_key = self._cache_key(img_bytes)
            cached = self._cache_get(cache_key)
            if cached is not None:
                cached.page_number = page_num
                result.pages.append(cached)
                result.total_images_cached += 1
                result.total_images_processed += 1
                continue

            # --- OCR ---
            page_result = self._process_single(img_bytes, mime_type=mime, mode=mode)
            page_result.page_number = page_num
            result.pages.append(page_result)
            result.total_images_processed += 1
            result.source = result.source or page_result.source

            # --- cache store ---
            self._cache_set(cache_key, page_result)

        return result

    def process_single_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
        mode: str = DEFAULT_MODE,
        page_number: int = 1,
    ) -> OcrPageResult:
        """Process one image with caching and fallback."""
        if not self._passes_filter(image_bytes):
            return OcrPageResult(page_number=page_number, source="skipped")

        cache_key = self._cache_key(image_bytes)
        cached = self._cache_get(cache_key)
        if cached is not None:
            cached.page_number = page_number
            return cached

        page = self._process_single(image_bytes, mime_type=mime_type, mode=mode)
        page.page_number = page_number
        self._cache_set(cache_key, page)
        return page

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _passes_filter(self, img_bytes: bytes) -> bool:
        """Check image size thresholds (byte count only — dimension check
        is deferred to callers that have decoded pixel data)."""
        return len(img_bytes) >= self._settings.min_image_bytes

    def _process_single(
        self,
        img_bytes: bytes,
        *,
        mime_type: str = "image/png",
        mode: str = DEFAULT_MODE,
    ) -> OcrPageResult:
        """Try vision LLM first, fall back to dots-ocr (if configured)."""
        # Primary: vision-capable LLM
        if self._vision_client is not None:
            result = self._ocr_via_vision_llm(img_bytes, mime_type=mime_type)
            if result.source != "llm-vision-error":
                return result
            logger.warning("Vision LLM OCR failed, trying dots-ocr fallback")

        # Secondary (optional): dots-ocr — only when URL is configured
        if self._is_dots_configured() and self._is_dots_available():
            try:
                client = self._get_dots_client()
                return client.process_image(img_bytes, mode=mode, mime_type=mime_type)
            except Exception as exc:
                logger.warning("dots-ocr fallback also failed: %s", exc)
                self._dots_healthy = False
                self._dots_last_check = time.monotonic()

        logger.error(
            "No OCR provider available (vision LLM not configured or failed, "
            "dots-ocr %s)",
            "also failed" if self._is_dots_configured() else "not configured",
        )
        return OcrPageResult(page_number=0, raw_text="", source="none")

    def _is_dots_configured(self) -> bool:
        """Return True when a dots-ocr URL has been explicitly configured."""
        url = self._settings.dots_ocr_base_url
        return bool(url and url.strip())

    def _is_dots_available(self) -> bool:
        """Check if dots-ocr should be attempted."""
        if self._dots_healthy:
            return True

        # Periodic health re-check
        elapsed = time.monotonic() - self._dots_last_check
        if elapsed >= self._health_check_interval:
            try:
                client = self._get_dots_client()
                healthy = client.health_check()
                self._dots_healthy = healthy
                self._dots_last_check = time.monotonic()
                if healthy:
                    logger.info("dots-ocr is healthy again")
                return healthy
            except Exception:
                self._dots_last_check = time.monotonic()
                return False

        return False

    def _get_dots_client(self) -> DotsOcrClient:
        if self._dots_client is None:
            self._dots_client = DotsOcrClient(self._settings)
        return self._dots_client

    # ------------------------------------------------------------------
    # Vision LLM fallback
    # ------------------------------------------------------------------

    def _ocr_via_vision_llm(
        self,
        img_bytes: bytes,
        *,
        mime_type: str = "image/png",
    ) -> OcrPageResult:
        """Use a vision-capable LLM (e.g. Gemini) to extract text from an image."""
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": (
                            "Extract all text content from this image. "
                            "Preserve the document structure including headings, "
                            "paragraphs, lists, and tables. Output tables as HTML. "
                            "Output formulas as LaTeX. Be thorough and accurate."
                        ),
                    },
                ],
            }
        ]

        try:
            response = self._vision_client.chat(messages=messages)
            raw_text = response.get("content", "") or ""
        except Exception as exc:
            logger.error("Vision LLM OCR failed: %s", exc)
            return OcrPageResult(page_number=0, raw_text="", source="llm-vision-error")

        return OcrPageResult(
            page_number=0,
            elements=[LayoutElement(category="Text", content=raw_text, order=0)] if raw_text else [],
            raw_text=raw_text,
            source="llm-vision",
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(img_bytes: bytes) -> str:
        """Generate a deterministic cache key from image bytes."""
        digest = hashlib.sha256(img_bytes).hexdigest()
        return f"ocr:{digest}"

    def _cache_get(self, key: str) -> Optional[OcrPageResult]:
        if self._cache is None:
            return None
        try:
            raw = self._cache.get(key)
            if raw is None:
                return None
            data = json.loads(raw) if isinstance(raw, str) else raw
            return self._deserialize_page(data)
        except Exception as exc:
            logger.debug("OCR cache read error: %s", exc)
            return None

    def _cache_set(self, key: str, page: OcrPageResult) -> None:
        if self._cache is None:
            return
        try:
            data = self._serialize_page(page)
            self._cache.set(key, json.dumps(data), ttl=self._settings.ocr_cache_ttl)
        except Exception as exc:
            logger.debug("OCR cache write error: %s", exc)

    @staticmethod
    def _serialize_page(page: OcrPageResult) -> Dict[str, Any]:
        return {
            "page_number": page.page_number,
            "raw_text": page.raw_text,
            "source": page.source,
            "elements": [
                {"category": e.category, "content": e.content, "order": e.order}
                for e in page.elements
            ],
        }

    @staticmethod
    def _deserialize_page(data: Dict[str, Any]) -> OcrPageResult:
        return OcrPageResult(
            page_number=data.get("page_number", 0),
            raw_text=data.get("raw_text", ""),
            source=data.get("source", "cached"),
            elements=[
                LayoutElement(
                    category=e.get("category", "Text"),
                    content=e.get("content", ""),
                    order=e.get("order", 0),
                )
                for e in data.get("elements", [])
            ],
        )
