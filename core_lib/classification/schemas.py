"""Pydantic models for document classification results."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentClassificationResult(BaseModel):
    """Result of LLM-based document classification with RAG description."""

    category_id: str = Field(
        description="Category key from DOC_CATEGORIES (e.g. 'technical_product_documentation')"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )
    reasoning: str = Field(
        description="One-sentence justification for the chosen category"
    )
    description: str = Field(
        description=(
            "2-4 sentence semantic summary of the document content, "
            "written in the document's own language, optimised for RAG retrieval"
        )
    )
    detection_method: Literal["llm", "default"] = Field(
        default="llm",
        description="How the classification was determined ('llm' or 'default' fallback)",
    )
    alternative_categories: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Up to 2 alternative category candidates with 'category_id' and "
            "'confidence' keys; empty list when classification is highly confident"
        ),
    )
