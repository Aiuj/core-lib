"""OCR data models.

Structured output models for dots-ocr and vision LLM results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LayoutElement:
    """A single layout element detected by OCR."""

    category: str  # Title, Section-header, Text, Table, List-item, etc.
    content: str  # Extracted text/HTML/LaTeX content
    order: int = 0  # Reading order within the page


@dataclass
class OcrPageResult:
    """OCR result for a single page or image."""

    page_number: int
    elements: List[LayoutElement] = field(default_factory=list)
    raw_text: str = ""  # Raw OCR output before parsing
    source: str = ""  # "dots-ocr" or "llm-vision"

    def to_markdown(self) -> str:
        """Convert structured elements to markdown text."""
        if not self.elements:
            return self.raw_text

        parts: List[str] = []
        for el in sorted(self.elements, key=lambda e: e.order):
            cat = el.category.lower()
            content = el.content.strip()
            if not content:
                continue

            if cat == "title":
                parts.append(f"# {content}")
            elif cat == "section-header":
                parts.append(f"## {content}")
            elif cat == "table":
                # Tables come as HTML from dots-ocr; preserve as-is
                parts.append(content)
            elif cat == "formula":
                parts.append(f"$$\n{content}\n$$")
            elif cat == "list-item":
                parts.append(f"- {content}")
            elif cat in ("page-header", "page-footer"):
                # Skip headers/footers — usually noise
                continue
            elif cat == "caption":
                parts.append(f"*{content}*")
            else:
                parts.append(content)

        return "\n\n".join(parts)


@dataclass
class OcrResult:
    """Aggregated OCR result for an entire document."""

    pages: List[OcrPageResult] = field(default_factory=list)
    source: str = ""  # Primary provider used
    total_images_processed: int = 0
    total_images_skipped: int = 0  # Below size thresholds
    total_images_cached: int = 0  # Served from cache
    errors: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Convert all pages to a single markdown string."""
        parts: List[str] = []
        for page in sorted(self.pages, key=lambda p: p.page_number):
            md = page.to_markdown()
            if md.strip():
                parts.append(md)
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return not any(p.elements or p.raw_text for p in self.pages)
