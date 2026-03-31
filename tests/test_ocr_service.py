"""Unit tests for OcrService — focusing on error-response detection in _ocr_via_vision_llm.

These tests validate that when an LLM client swallows an exception and returns
{"content": None, "error": "..."} instead of raising, the OcrService correctly
classifies the result as "llm-vision-error" so the dots-ocr fallback is triggered.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core_lib.config.ocr_settings import OcrSettings
from core_lib.ocr.ocr_service import OcrService
from core_lib.ocr.models import OcrPageResult


@pytest.fixture()
def settings() -> OcrSettings:
    """Minimal OcrSettings with no dots-ocr configured."""
    return OcrSettings(min_image_bytes=1)  # low threshold so test images pass filter


@pytest.fixture()
def tiny_image() -> bytes:
    """A 2-byte dummy image that passes a min_image_bytes=1 filter."""
    return b"\x89P"  # first two bytes of a PNG header — enough for the filter


def _make_service(settings: OcrSettings, vision_response: dict) -> OcrService:
    """Build an OcrService with a mock vision LLM client returning *vision_response*."""
    mock_client = MagicMock()
    mock_client.chat.return_value = vision_response
    return OcrService(settings, vision_llm_client=mock_client)


class TestOcrViaVisionLlm:
    """Tests for _ocr_via_vision_llm error-response detection."""

    def test_successful_response_returns_content(self, settings, tiny_image):
        """When chat() returns valid content, a llm-vision page is produced."""
        service = _make_service(settings, {"content": "Extracted text from slide", "error": None})
        page = service._ocr_via_vision_llm(tiny_image, mime_type="image/png")

        assert page.source == "llm-vision"
        assert page.raw_text == "Extracted text from slide"
        assert len(page.elements) == 1

    def test_error_in_response_dict_returns_llm_vision_error(self, settings, tiny_image):
        """When chat() swallows exception and returns {"content": None, "error": "..."}, 
        _ocr_via_vision_llm must return source="llm-vision-error" so that dots-ocr fallback fires."""
        service = _make_service(settings, {"content": None, "error": "HTTP 503 Service Unavailable"})
        page = service._ocr_via_vision_llm(tiny_image, mime_type="image/png")

        assert page.source == "llm-vision-error", (
            "An error response dict must produce source='llm-vision-error' "
            "so that _process_single() can attempt the dots-ocr fallback."
        )
        assert page.raw_text == ""

    def test_none_content_no_error_field_returns_llm_vision_error(self, settings, tiny_image):
        """When content is None even without an explicit error key, treat as failure."""
        service = _make_service(settings, {"content": None})
        page = service._ocr_via_vision_llm(tiny_image, mime_type="image/png")

        assert page.source == "llm-vision-error"

    def test_empty_string_content_no_error_returns_llm_vision_with_no_elements(self, settings, tiny_image):
        """When content is an empty string (genuine LLM empty output), 
        we still get source='llm-vision' but with no elements (slide truly had no text)."""
        service = _make_service(settings, {"content": "", "error": None})
        page = service._ocr_via_vision_llm(tiny_image, mime_type="image/png")

        assert page.source == "llm-vision"
        assert page.raw_text == ""
        assert page.elements == []

    def test_exception_raised_by_chat_returns_llm_vision_error(self, settings, tiny_image):
        """When chat() actually raises, the except-block must return llm-vision-error."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("Connection refused")
        service = OcrService(settings, vision_llm_client=mock_client)

        page = service._ocr_via_vision_llm(tiny_image, mime_type="image/png")
        assert page.source == "llm-vision-error"


class TestProcessSingleFallback:
    """Tests that _process_single correctly falls through to dots-ocr when vision LLM fails."""

    def test_error_response_triggers_dots_ocr_fallback_path(self, settings, tiny_image):
        """When _ocr_via_vision_llm returns llm-vision-error and dots-ocr is not configured,
        the final result must be source='none' (not 'llm-vision' with empty content)."""
        service = _make_service(settings, {"content": None, "error": "auth error"})
        # dots-ocr is NOT configured (no dots_ocr_base_url), so fallback goes straight to 'none'
        page = service._process_single(tiny_image, mime_type="image/png")

        assert page.source == "none", (
            "With no dots-ocr configured and a failing vision LLM, source must be 'none'"
        )
        assert page.raw_text == ""

    def test_successful_vision_result_not_overridden(self, settings, tiny_image):
        """A successful vision LLM response must be returned directly without attempting dots-ocr."""
        service = _make_service(settings, {"content": "Green energy statistics", "error": None})
        page = service._process_single(tiny_image, mime_type="image/png")

        assert page.source == "llm-vision"
        assert "Green energy" in page.raw_text


class TestStripCodeFences:
    """Tests for the _strip_code_fences helper."""

    def _strip(self, text: str) -> str:
        return OcrService._strip_code_fences(text)

    def test_strips_single_html_fence(self):
        raw = "```html\n<table><tr><td>Hello</td></tr></table>\n```"
        assert self._strip(raw) == "<table><tr><td>Hello</td></tr></table>"

    def test_strips_generic_fence_no_language(self):
        raw = "```\nsome text\n```"
        assert self._strip(raw) == "some text"

    def test_strips_python_fence(self):
        raw = "```python\nprint('hi')\n```"
        assert self._strip(raw) == "print('hi')"

    def test_plain_text_unchanged(self):
        raw = "Just plain text without any fences."
        assert self._strip(raw) == raw

    def test_mixed_content_inline_fences_replaced(self):
        """Fences embedded in larger text are stripped in-place."""
        raw = "Title here\n\n```html\n<table></table>\n```\n\nMore text."
        result = self._strip(raw)
        assert "```" not in result
        assert "<table></table>" in result
        assert "Title here" in result
        assert "More text." in result

    def test_leading_trailing_whitespace_stripped(self):
        raw = "  ```html\n<p>Text</p>\n```  "
        assert self._strip(raw) == "<p>Text</p>"

    def test_realistic_slide_html(self):
        """Mirrors the actual LLM output observed in production."""
        raw = (
            "```html\n"
            "<table border=\"1\">\n"
            "    <tr><td><h1>Decarbonized Future</h1></td></tr>\n"
            "</table>\n"
            "```"
        )
        result = self._strip(raw)
        assert result.startswith("<table")
        assert "```" not in result
        assert "Decarbonized Future" in result

    def test_ocr_via_vision_llm_strips_fences_from_content(self, settings, tiny_image):
        """End-to-end: when the LLM wraps its response in ```html, the stored raw_text is clean."""
        html_in_fence = (
            "```html\n"
            "<table><tr><td>Green Energy Slide</td></tr></table>\n"
            "```"
        )
        service = _make_service(settings, {"content": html_in_fence, "error": None})
        page = service._ocr_via_vision_llm(tiny_image, mime_type="image/png")

        assert page.source == "llm-vision"
        assert "```" not in page.raw_text, "Code fence must be stripped from raw_text"
        assert "<table>" in page.raw_text
        assert "Green Energy Slide" in page.raw_text
