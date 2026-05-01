"""LLM-based document classifier with RAG description generation."""
from __future__ import annotations

from typing import Optional

from core_lib.config.doc_categories import DOC_CATEGORIES
from core_lib.llm import create_fallback_llm_client
from core_lib.tracing import get_module_logger

from .schemas import DocumentClassificationResult

logger = get_module_logger()

# Build the category list once at import time so the prompt stays in sync with DOC_CATEGORIES.
_CATEGORIES_LIST = "\n".join(
    f'  - "{cat["key"]}": {cat["description"]}'
    for cat in DOC_CATEGORIES
)

_SYSTEM_PROMPT = f"""You are a document classification expert. Given a document's filename, \
file type, language, and a short content excerpt, classify the document into exactly one of \
the following categories:

{_CATEGORIES_LIST}

Return a JSON object with these fields:
- category_id: the exact key from the list above that best matches the document
- confidence: a float between 0.0 and 1.0 indicating how confident you are
- reasoning: one sentence explaining why this category was chosen
- description: a 2-4 sentence semantic summary of the document content, written in the \
document's own language, suitable for RAG retrieval — focus on what the document contains \
and what questions it can answer
- alternative_categories: list of up to 2 alternative categories as objects with \
"category_id" and "confidence" keys; use an empty list when highly confident

Use only the category_id values from the list above. Be precise."""


class DocumentClassifier:
    """LLM-based document classifier that also generates RAG-optimised descriptions.

    Uses a cheap/fast LLM tier (intelligence_level=3 by default) to classify
    documents and produce a semantic summary in a single call.

    Example::

        classifier = DocumentClassifier()
        result = classifier.classify(
            filename="Q4_Annual_Report_2024.pdf",
            content_excerpt="Revenue increased by 12%...",
            language="en",
            file_type="pdf",
        )
        print(result.category_id, result.confidence, result.description)
    """

    def __init__(self, intelligence_level: int = 3) -> None:
        """Initialise the classifier.

        Args:
            intelligence_level: LLM tier to use (3 = cheap/fast, suitable for classification).
        """
        self._intelligence_level = intelligence_level
        self._client = None  # Lazy initialisation to avoid startup overhead

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        filename: str,
        content_excerpt: str,
        language: str = "unknown",
        file_type: Optional[str] = None,
    ) -> DocumentClassificationResult:
        """Classify a document and generate a RAG description.

        Args:
            filename: Original document filename (provides type/naming hints).
            content_excerpt: Up to 2 000 characters of document content.
            language: ISO language code or 'unknown'.
            file_type: File extension without dot (e.g. 'pdf', 'docx').

        Returns:
            :class:`DocumentClassificationResult` with category, confidence, and description.
            Returns a safe default (category_id='general', confidence=0.0) on any error.
        """
        try:
            user_message = self._build_user_message(filename, content_excerpt, language, file_type)

            client = self._get_client()
            response = client.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                structured_output=DocumentClassificationResult,
            )

            result = response.get("content")
            if not isinstance(result, DocumentClassificationResult):
                logger.warning(
                    f"Unexpected classification response type: {type(result).__name__}"
                )
                return self._default_result()

            # Validate returned category_id against known keys
            valid_keys = {cat["key"] for cat in DOC_CATEGORIES}
            if result.category_id not in valid_keys:
                logger.warning(
                    f"LLM returned unknown category_id '{result.category_id}'; "
                    "substituting 'general'"
                )
                result = DocumentClassificationResult(
                    category_id="general",
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    description=result.description,
                    detection_method="llm",
                    alternative_categories=result.alternative_categories,
                )

            return result

        except Exception as exc:
            logger.warning(f"Document classification failed for '{filename}': {exc}")
            return self._default_result()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazily create and cache the LLM client."""
        if self._client is None:
            self._client = create_fallback_llm_client(
                intelligence_level=self._intelligence_level,
                usage="classify",
            )
        return self._client

    @staticmethod
    def _build_user_message(
        filename: str,
        content_excerpt: str,
        language: str,
        file_type: Optional[str],
    ) -> str:
        file_info = filename
        if file_type:
            file_info += f" ({file_type.upper()} file)"
        lang_part = f", language: {language}" if language and language != "unknown" else ""
        excerpt = content_excerpt or "(no content available)"
        return f"Document: {file_info}{lang_part}\n\nContent excerpt:\n{excerpt}"

    @staticmethod
    def _default_result() -> DocumentClassificationResult:
        """Return a safe fallback when classification is unavailable."""
        return DocumentClassificationResult(
            category_id="general",
            confidence=0.0,
            reasoning="Classification unavailable",
            description="",
            detection_method="default",
            alternative_categories=[],
        )
