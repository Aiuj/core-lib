"""Dots-OCR client for VLM-based document OCR.

Communicates with the dots-ocr service (vLLM-hosted VLM) via OpenAI-compatible API.
"""

from __future__ import annotations

import base64
import re
from typing import Dict, List, Optional

from .. import get_module_logger
from ..config.ocr_settings import OcrSettings
from .models import LayoutElement, OcrPageResult

logger = get_module_logger()

# dots-ocr prompt prefix required by the model
_IMG_PREFIX = "<|img|><|imgpad|><|endofimg|>"

# Supported prompt modes matching the dots-ocr model
PROMPTS: Dict[str, str] = {
    "layout_all": (
        "Parse the text in the image. The output is the content in each "
        "layout element in order. The following categories of layout elements "
        "are considered: Title, Section-header, Text, Table, List-item, "
        "Picture, Caption, Footnote, Formula, Page-footer, Page-header."
    ),
    "layout_only": (
        "Parse the layout of the image. The output is the position and category "
        "of each layout element in the image (in JSON format). The following "
        "categories of layout elements are considered: Title, Section-header, "
        "Text, Table, List-item, Picture, Caption, Footnote, Formula, "
        "Page-footer, Page-header."
    ),
    "ocr": "Read all the text in the image.",
    "scene_spotting": (
        "List all the readable text along with its spatial position in the image."
    ),
    "web_parsing": "Parse the content of the web page in the image.",
}

# Default mode for document processing
DEFAULT_MODE = "layout_all"


class DotsOcrClient:
    """Client for the dots-ocr VLM service.

    Uses the OpenAI-compatible API exposed by vLLM to send images and
    receive structured OCR results.
    """

    def __init__(self, settings: OcrSettings) -> None:
        self._settings = settings
        self._base_url = settings.dots_ocr_base_url.rstrip("/")
        self._client = self._create_client()

    def _create_client(self):
        """Create OpenAI client for dots-ocr API."""
        from openai import OpenAI

        return OpenAI(
            api_key=self._settings.dots_ocr_api_key or "not-needed",
            base_url=f"{self._base_url}/v1",
            timeout=self._settings.dots_ocr_timeout,
        )

    def process_image(
        self,
        image_bytes: bytes,
        *,
        mode: str = DEFAULT_MODE,
        mime_type: str = "image/png",
    ) -> OcrPageResult:
        """Process a single image through dots-ocr.

        Args:
            image_bytes: Raw image bytes (PNG, JPEG, etc.)
            mode: OCR prompt mode (layout_all, ocr, etc.)
            mime_type: MIME type of the image.

        Returns:
            OcrPageResult with extracted elements.
        """
        prompt_text = PROMPTS.get(mode, PROMPTS[DEFAULT_MODE])
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": f"{_IMG_PREFIX}{prompt_text}"},
                ],
            }
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._settings.dots_ocr_model,
                messages=messages,
                max_completion_tokens=self._settings.dots_ocr_max_tokens,
                temperature=self._settings.ocr_temperature,
                top_p=0.9,
            )
            raw_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.error("dots-ocr request failed: %s", e)
            raise

        elements = self._parse_layout_output(raw_text) if mode == "layout_all" else []

        return OcrPageResult(
            page_number=0,
            elements=elements,
            raw_text=raw_text,
            source="dots-ocr",
        )

    def health_check(self) -> bool:
        """Check if the dots-ocr service is reachable."""
        import httpx

        try:
            resp = httpx.get(
                f"{self._base_url}/health",
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            try:
                # vLLM may not expose /health; try listing models
                resp = httpx.get(
                    f"{self._base_url}/v1/models",
                    timeout=5,
                )
                return resp.status_code == 200
            except Exception:
                return False

    @staticmethod
    def _parse_layout_output(raw: str) -> List[LayoutElement]:
        """Parse dots-ocr layout_all output into structured elements.

        The model outputs content grouped by category headers like:
            Title:
            Some title text

            Text:
            Body text here

            Table:
            <table>...</table>
        """
        # Known layout categories from dots-ocr
        categories = {
            "title", "section-header", "text", "table", "list-item",
            "picture", "caption", "footnote", "formula",
            "page-footer", "page-header",
        }

        # Build pattern to match category headers
        cat_pattern = re.compile(
            r"^(" + "|".join(re.escape(c) for c in sorted(categories)) + r")\s*:\s*$",
            re.IGNORECASE | re.MULTILINE,
        )

        elements: List[LayoutElement] = []
        # Split by category headers
        parts = cat_pattern.split(raw)

        # parts[0] is text before any category header (if any)
        # Then alternating: category, content, category, content, ...
        if len(parts) < 3:
            # No structured output detected; return raw as single Text element
            text = raw.strip()
            if text:
                elements.append(LayoutElement(category="Text", content=text, order=0))
            return elements

        idx = 1  # skip parts[0] (pre-header text)
        order = 0
        while idx + 1 < len(parts):
            cat_name = parts[idx].strip()
            content = parts[idx + 1].strip()
            if content:
                # Normalize category name to Title-case
                cat_display = cat_name.capitalize()
                for known in categories:
                    if known.lower() == cat_name.lower():
                        cat_display = known.replace("-", "-").title().replace("-", "-")
                        # Fix common ones
                        if known.lower() == "section-header":
                            cat_display = "Section-header"
                        elif known.lower() == "list-item":
                            cat_display = "List-item"
                        elif known.lower() == "page-footer":
                            cat_display = "Page-footer"
                        elif known.lower() == "page-header":
                            cat_display = "Page-header"
                        else:
                            cat_display = known.capitalize()
                        break
                elements.append(LayoutElement(
                    category=cat_display,
                    content=content,
                    order=order,
                ))
                order += 1
            idx += 2

        return elements
