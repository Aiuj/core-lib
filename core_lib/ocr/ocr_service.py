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
import re
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

        # Override the vision LLM's timeout and max output tokens.
        # Vision / OCR requests send large base64 payloads that local models
        # (Ollama) process much slower than text-only requests — the default
        # 60 s provider timeout is often insufficient.
        # The max_tokens cap prevents small models from generating excessively
        # long (or runaway) output with complex prompts like the enriched one.
        if self._vision_client is not None:
            try:
                provider = getattr(self._vision_client, "_provider", None)
                config = getattr(provider, "config", None)
                if config is not None:
                    if hasattr(config, "timeout"):
                        config.timeout = settings.vision_llm_timeout
                    if settings.vision_max_output_tokens > 0 and hasattr(config, "max_tokens"):
                        config.max_tokens = settings.vision_max_output_tokens
            except Exception:
                pass  # Best effort — not all providers expose these attributes

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
        enrich: bool = False,
    ) -> OcrResult:
        """Process a batch of images sequentially.

        Args:
            images: List of raw image byte buffers.
            mime_types: Parallel list of MIME types (defaults to image/png).
            start_page: Page number offset for the first image.
            mode: dots-ocr prompt mode.
            enrich: When True, the vision LLM produces a combined output
                with an image description, search-oriented keywords, and
                the extracted text.  Improves embedding quality for
                search/retrieval at the cost of slightly higher latency.

        Returns:
            Aggregated ``OcrResult``.
        """
        result = OcrResult()
        if not images:
            return result

        if mime_types is None:
            mime_types = ["image/png"] * len(images)

        total = len(images)
        logger.info("OCR: processing %d image(s) (enrich=%s)", total, enrich)

        for idx, (img_bytes, mime) in enumerate(zip(images, mime_types)):
            page_num = start_page + idx

            # --- size filter ---
            if not self._passes_filter(img_bytes):
                result.total_images_skipped += 1
                logger.debug("OCR image %d/%d: skipped (too small)", idx + 1, total)
                continue

            # --- cache lookup ---
            cache_key = self._cache_key(img_bytes, enrich=enrich)
            cached = self._cache_get(cache_key)
            if cached is not None:
                cached.page_number = page_num
                result.pages.append(cached)
                result.total_images_cached += 1
                result.total_images_processed += 1
                logger.debug("OCR image %d/%d: cache hit", idx + 1, total)
                continue

            # --- OCR ---
            logger.info("OCR image %d/%d: processing (%.1f KB)...", idx + 1, total, len(img_bytes) / 1024)
            page_result = self._process_single(img_bytes, mime_type=mime, mode=mode, enrich=enrich)
            page_result.page_number = page_num
            result.pages.append(page_result)
            result.total_images_processed += 1
            result.source = result.source or page_result.source

            # --- cache store ---
            self._cache_set(cache_key, page_result)

        logger.info(
            "OCR: completed — processed=%d, cached=%d, skipped=%d",
            result.total_images_processed, result.total_images_cached, result.total_images_skipped,
        )
        return result

    def process_single_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
        mode: str = DEFAULT_MODE,
        page_number: int = 1,
        enrich: bool = False,
    ) -> OcrPageResult:
        """Process one image with caching and fallback."""
        if not self._passes_filter(image_bytes):
            return OcrPageResult(page_number=page_number, source="skipped")

        cache_key = self._cache_key(image_bytes, enrich=enrich)
        cached = self._cache_get(cache_key)
        if cached is not None:
            cached.page_number = page_number
            return cached

        page = self._process_single(image_bytes, mime_type=mime_type, mode=mode, enrich=enrich)
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
        enrich: bool = False,
    ) -> OcrPageResult:
        """Try vision LLM first, fall back to dots-ocr (if configured)."""
        # Primary: vision-capable LLM
        if self._vision_client is not None:
            result = self._ocr_via_vision_llm(img_bytes, mime_type=mime_type, enrich=enrich)
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

    _OCR_ONLY_PROMPT = (
        "Extract all text content from this image. "
        "Output as clean markdown. "
        "Preserve the document structure including headings, "
        "paragraphs, lists, and tables. Use markdown tables. "
        "Output formulas as LaTeX. Be thorough and accurate. "
        "Do not wrap the output in code fences."
    )

    _ENRICHED_PROMPT = (
        "Analyze this image and summarize it for search indexing. "
        "Return EXACTLY three markdown sections in this order:\n\n"
        "## Description\n"
        "Write 1-2 short sentences describing the main subject of the image.\n\n"
        "## Search Terms\n"
        "List 5-10 relevant keywords as a comma-separated list.\n\n"
        "## Text Content\n"
        "Extract all readable text exactly as it appears. Use markdown tables for tabular data. "
        "Output formulas as LaTeX. If there is no text, write 'None'.\n\n"
        "Do not include any other commentary. Do not wrap the output in code fences."
    )

    def _ocr_via_vision_llm(
        self,
        img_bytes: bytes,
        *,
        mime_type: str = "image/png",
        enrich: bool = False,
    ) -> OcrPageResult:
        """Use a vision-capable LLM (e.g. Gemini) to extract text from an image.

        When *enrich* is True the prompt asks for a visual description and
        search-oriented keywords in addition to the raw OCR text.  This
        produces richer content that improves embedding quality for
        search and retrieval of image-heavy documents.
        """
        # Resize large images and convert PNG→JPEG to shrink the base64
        # payload.  This dramatically speeds up local model inference.
        img_bytes, mime_type = self._optimize_image(img_bytes, mime_type)

        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"

        prompt_text = self._ENRICHED_PROMPT if enrich else self._OCR_ONLY_PROMPT

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ]

        try:
            response = self._vision_client.chat(messages=messages)
            # LLM clients may swallow exceptions internally and return
            # {"content": None, "error": "..."} instead of raising.
            # Detect this so the dots-ocr fallback can be attempted.
            if response.get("error") or response.get("content") is None:
                error_detail = response.get("error", "no content in response")
                logger.error("Vision LLM OCR failed: %s", error_detail)
                return OcrPageResult(page_number=0, raw_text="", source="llm-vision-error")
            raw_text = response.get("content", "") or ""
        except Exception as exc:
            logger.error("Vision LLM OCR failed: %s", exc)
            return OcrPageResult(page_number=0, raw_text="", source="llm-vision-error")

        # Strip markdown code fences that some LLMs add around their output
        # (e.g. ```html\n<table>...</table>\n```) even when not asked to.
        raw_text = self._strip_code_fences(raw_text)

        source = "llm-vision-enriched" if enrich else "llm-vision"
        return OcrPageResult(
            page_number=0,
            elements=[LayoutElement(category="Text", content=raw_text, order=0)] if raw_text else [],
            raw_text=raw_text,
            source=source,
        )

    # ------------------------------------------------------------------
    # Image optimisation
    # ------------------------------------------------------------------

    def _optimize_image(
        self,
        img_bytes: bytes,
        mime_type: str,
    ) -> tuple:
        """Resize and compress an image to reduce vision-LLM processing time.

        1. If the longest side exceeds ``max_image_dimension``, resize
           proportionally (LANCZOS downscale).
        2. Convert non-JPEG images to JPEG at ``ocr_jpeg_quality`` to
           shrink the base64 payload (typically 3-5× smaller than PNG).

        Returns ``(optimised_bytes, mime_type)``; falls back to the
        original if PIL is not available or on any error.
        """
        max_dim = self._settings.max_image_dimension
        jpeg_quality = self._settings.ocr_jpeg_quality

        # Feature disabled — return as-is
        if max_dim <= 0 and mime_type in ("image/jpeg", "image/jpg"):
            return img_bytes, mime_type

        try:
            from PIL import Image
            import io as _io

            img = Image.open(_io.BytesIO(img_bytes))
            w, h = img.size
            original_size = len(img_bytes)
            resized = False

            # --- Resize if needed ----------------------------------------
            if max_dim > 0 and max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                resized = True

            # --- Convert to JPEG -----------------------------------------
            # JPEG doesn't support alpha; convert RGBA → RGB first.
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")

            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            optimised = buf.getvalue()
            new_mime = "image/jpeg"

            if resized or len(optimised) < original_size:
                logger.debug(
                    "Image optimised: %dx%d → %dx%d, %s → JPEG, "
                    "%.1f KB → %.1f KB (%.0f%% reduction)",
                    w, h,
                    img.size[0], img.size[1],
                    mime_type.split("/")[-1].upper(),
                    original_size / 1024,
                    len(optimised) / 1024,
                    (1 - len(optimised) / original_size) * 100 if original_size else 0,
                )
                return optimised, new_mime

            # Conversion didn't help (rare for PNG) — keep original
            return img_bytes, mime_type

        except ImportError:
            logger.debug("PIL not available — skipping image optimisation")
            return img_bytes, mime_type
        except Exception as exc:
            logger.debug("Image optimisation failed, using original: %s", exc)
            return img_bytes, mime_type

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences wrapping the LLM response.

        Vision LLMs sometimes wrap their entire output in a fenced code block
        (e.g. ```html ... ```) even when not asked to.  This strips such
        wrappers so the raw content (HTML, LaTeX, plain text) is stored
        directly rather than inside a markdown formatting artefact.

        If the whole response is one fence, the inner content is returned.
        If the response contains multiple fences mixed with plain text, each
        fence is replaced by its inner content in-place.
        """
        stripped = text.strip()
        # Fast path: entire response is a single code fence
        single = re.match(r'^```[a-zA-Z]*\n(.*?)\n?```\s*$', stripped, re.DOTALL)
        if single:
            return single.group(1).strip()
        # Slow path: replace individual fences embedded in larger text
        return re.sub(r'```[a-zA-Z]*\n(.*?)\n?```', lambda m: m.group(1).strip(), stripped, flags=re.DOTALL)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(img_bytes: bytes, *, enrich: bool = False) -> str:
        """Generate a deterministic cache key from image bytes."""
        digest = hashlib.sha256(img_bytes).hexdigest()
        prefix = "ocr-enriched" if enrich else "ocr"
        return f"{prefix}:{digest}"

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
            self._cache.set(key, data, ttl=self._settings.ocr_cache_ttl)
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
