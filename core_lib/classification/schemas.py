"""Pydantic models for document classification results."""

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TopicProfile(BaseModel):
    """Semantic scope profile shared by questionnaires and knowledge documents."""

    description: str = Field(
        default="",
        description="Retrieval-oriented summary of the subject and questions the source can answer",
    )
    primary_topics: List[str] = Field(default_factory=list)
    product_areas: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    document_category: Optional[str] = None
    language: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    detection_method: Literal["llm", "derived", "provided", "default"] = "llm"
    classifier_version: str = "topic-profile-v1"
    fingerprint: str = ""

    @staticmethod
    def normalize_terms(value: Any, max_terms: int = 20) -> List[str]:
        """Normalize, deduplicate, and bound profile term lists consistently."""
        if not value:
            return []
        if not isinstance(value, list):
            value = [value]
        cleaned: List[str] = []
        seen = set()
        for item in value:
            term = " ".join(str(item).split()).strip()
            key = term.casefold()
            if term and key not in seen:
                cleaned.append(term)
                seen.add(key)
        return cleaned[:max_terms]

    @field_validator("primary_topics", "product_areas", "capabilities", mode="before")
    @classmethod
    def _clean_terms(cls, value: Any) -> List[str]:
        return cls.normalize_terms(value)

    @model_validator(mode="after")
    def _set_fingerprint(self) -> "TopicProfile":
        # Never trust a caller- or model-supplied fingerprint: cache identity
        # must match the normalized semantic fields in this profile.
        self.fingerprint = self.compute_fingerprint()
        return self

    def compute_fingerprint(self) -> str:
        def normalized_scalar(value: Any, default: str = "") -> str:
            normalized = " ".join(str(value or "").split()).casefold()
            return normalized or default

        def normalized_terms(values: List[str]) -> List[str]:
            normalized_values = (normalized_scalar(value) for value in values)
            return sorted({value for value in normalized_values if value})

        payload = {
            "description": normalized_scalar(self.description),
            "primary_topics": normalized_terms(self.primary_topics),
            "product_areas": normalized_terms(self.product_areas),
            "capabilities": normalized_terms(self.capabilities),
            "document_category": normalized_scalar(self.document_category),
            "language": normalized_scalar(self.language, default="unknown"),
            "classifier_version": normalized_scalar(self.classifier_version),
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_reranker_text(self) -> str:
        parts = []
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.primary_topics:
            parts.append("Topics: " + ", ".join(self.primary_topics))
        if self.product_areas:
            parts.append("Product areas: " + ", ".join(self.product_areas))
        if self.capabilities:
            parts.append("Capabilities: " + ", ".join(self.capabilities))
        return "\n".join(parts)


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
    primary_topics: List[str] = Field(default_factory=list)
    product_areas: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    content_structure: Literal[
        "prose", "qa_pairs", "table", "list", "presentation", "mixed", "unknown"
    ] = Field(
        default="unknown",
        description="Dominant structural organization, independent of topic category",
    )
    structure_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in content_structure",
    )
    pairing_pattern: Literal[
        "alternating_blocks", "table_columns", "labeled_fields", "mixed", "unknown"
    ] = Field(
        default="unknown",
        description="How questions and answers are paired when content_structure is qa_pairs or mixed",
    )
    prospect_document_type: Optional[Literal[
        "requirements_specification",
        "project_description",
        "rfx_instructions",
        "evaluation_criteria",
        "technical_requirements",
        "security_requirements",
        "commercial_terms",
        "contractual_terms",
        "implementation_timeline",
        "other",
    ]] = Field(
        default=None,
        description=(
            "Dominant buyer/prospect RFx package type, or null when the document "
            "is not clearly prospect-supplied RFx material"
        ),
    )
    prospect_document_type_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in prospect_document_type",
    )

    def to_topic_profile(self, language: str = "unknown") -> TopicProfile:
        return TopicProfile(
            description=self.description,
            primary_topics=self.primary_topics,
            product_areas=self.product_areas,
            capabilities=self.capabilities,
            document_category=self.category_id,
            language=language or "unknown",
            confidence=self.confidence,
            reasoning=self.reasoning,
            detection_method="llm" if self.detection_method == "llm" else "default",
            classifier_version="document-topic-v1",
        )
