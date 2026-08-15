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
- description: a 2-4 sentence semantic summary of the document content, suitable for RAG \
retrieval — focus on what the document contains and what questions it can answer. \
IMPORTANT: detect the language of the content excerpt and write this description in that \
exact same language. If the content is in English write in English, if French write in \
French, etc.
- alternative_categories: list of up to 2 alternative categories as objects with \
"category_id" and "confidence" keys; use an empty list when highly confident
- primary_topics: precise open-vocabulary subjects covered by the document
- product_areas: named products, modules, optional features, or business processes
- capabilities: concrete capabilities or questions this document can support
- content_structure: the dominant organization of the content, exactly one of
  "prose", "qa_pairs", "table", "list", "presentation", "mixed", or "unknown".
  This is independent of the topic category. Use "qa_pairs" only when the excerpt
  contains at least two actual question/prompt and answer relationships, in any
  language. A question-like heading, rhetorical question, troubleshooting title,
  or prose that discusses questions is not a Q&A pair unless answer content follows
  it. Use "mixed" only when at least two real pairs coexist with substantial non-Q&A
  content. Use "prose" for ordinary guides and policies, even when some headings are
  phrased as questions. Preserve genuine Q&A detection when prompts omit question
  marks (for example numbered questionnaire prompts followed by response blocks).
- structure_confidence: confidence between 0.0 and 1.0 in content_structure
- pairing_pattern: exactly one of "alternating_blocks", "table_columns",
  "labeled_fields", "mixed", or "unknown"
  For "qa_pairs", return the concrete observed pairing pattern rather than
  "unknown". Use "alternating_blocks" for repeated prompt then response blocks,
  "labeled_fields" for Q:/A: or equivalent labels, and "table_columns" only when
  separate question and answer columns are visible.
- prospect_document_type: when the document is clearly material supplied by a
  buyer/prospect as part of an RFx package, classify its dominant purpose as
  exactly one of "requirements_specification", "project_description",
  "rfx_instructions", "evaluation_criteria", "technical_requirements",
  "security_requirements", "commercial_terms", "contractual_terms",
  "implementation_timeline", or "other". Return null for company-authored
  knowledge, product documentation, policies, completed answers, and documents
  that are not clearly buyer/prospect RFx material. A cahier des charges or
  specification dominated by requested capabilities is
  "requirements_specification" even when it also contains background,
  technical, security, evaluation, or timeline sections.
- prospect_document_type_confidence: confidence between 0.0 and 1.0 in the
  prospect_document_type; use 0.0 when prospect_document_type is null

`primary_topics` and `capabilities` are required whenever the excerpt contains
enough information to identify them. Return 1-5 concise terms for each; do not
repeat the description. `product_areas` may be empty only when the document does
not name a product, module, feature, or business process.

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
        document_role: Optional[str] = None,
    ) -> DocumentClassificationResult:
        """Classify a document and generate a RAG description.

        Args:
            filename: Original document filename (provides type/naming hints).
            content_excerpt: Up to 2 000 characters of document content.
            language: ISO language code or 'unknown'.
            file_type: File extension without dot (e.g. 'pdf', 'docx').
            document_role: Optional usage hint such as ``prospect_context``.

        Returns:
            :class:`DocumentClassificationResult` with category, confidence, and description.
            Returns a safe default (category_id='general', confidence=0.0) on any error.
        """
        try:
            user_message = self._build_user_message(
                filename, content_excerpt, language, file_type, document_role
            )

            client = self._get_client()
            response = client.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                structured_output=DocumentClassificationResult,
            )

            result = response.get("content")

            # When structured output succeeds the provider returns content as a
            # model_dump() dict (not the Pydantic instance itself).  Validate it.
            if isinstance(result, dict):
                try:
                    result = DocumentClassificationResult.model_validate(result)
                except Exception as exc:
                    logger.warning(f"Failed to validate structured classification response: {exc}")
                    return self._default_result()
            elif not isinstance(result, DocumentClassificationResult):
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
                    primary_topics=result.primary_topics,
                    product_areas=result.product_areas,
                    capabilities=result.capabilities,
                    content_structure=result.content_structure,
                    structure_confidence=result.structure_confidence,
                    pairing_pattern=result.pairing_pattern,
                    prospect_document_type=result.prospect_document_type,
                    prospect_document_type_confidence=result.prospect_document_type_confidence,
                )

            result = self._enrich_missing_scope_terms(
                client, result, filename, content_excerpt, language, file_type,
                document_role,
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
    def _coerce_result(response) -> Optional[DocumentClassificationResult]:
        """Validate a structured provider response without accepting partial dicts."""
        result = response.get("content") if isinstance(response, dict) else None
        if isinstance(result, dict):
            try:
                return DocumentClassificationResult.model_validate(result)
            except Exception:
                return None
        return result if isinstance(result, DocumentClassificationResult) else None

    def _enrich_missing_scope_terms(
        self,
        client,
        result: DocumentClassificationResult,
        filename: str,
        content_excerpt: str,
        language: str,
        file_type: Optional[str],
        document_role: Optional[str],
    ) -> DocumentClassificationResult:
        """Use one bounded follow-up when a valid classification lacks usable scope."""
        if not result.description or (result.primary_topics and result.capabilities):
            return result

        try:
            response = client.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            self._build_user_message(
                                filename, content_excerpt, language, file_type,
                                document_role,
                            )
                            + "\n\nThe initial classification omitted retrieval scope terms. "
                            "Return the complete JSON again, with at least one concise "
                            "primary_topics value and one capabilities value grounded in the excerpt."
                        ),
                    },
                ],
                structured_output=DocumentClassificationResult,
            )
            enriched = self._coerce_result(response)
            if not enriched:
                return result
            return DocumentClassificationResult(
                category_id=result.category_id,
                confidence=result.confidence,
                reasoning=result.reasoning,
                description=result.description,
                detection_method=result.detection_method,
                alternative_categories=result.alternative_categories,
                primary_topics=result.primary_topics or enriched.primary_topics,
                product_areas=result.product_areas or enriched.product_areas,
                capabilities=result.capabilities or enriched.capabilities,
                content_structure=result.content_structure,
                structure_confidence=result.structure_confidence,
                pairing_pattern=result.pairing_pattern,
                prospect_document_type=result.prospect_document_type,
                prospect_document_type_confidence=result.prospect_document_type_confidence,
            )
        except Exception as exc:
            logger.warning("Failed to enrich document scope for '%s': %s", filename, exc)
            return result

    @staticmethod
    def _build_user_message(
        filename: str,
        content_excerpt: str,
        language: str,
        file_type: Optional[str],
        document_role: Optional[str] = None,
    ) -> str:
        file_info = filename
        if file_type:
            file_info += f" ({file_type.upper()} file)"
        lang_part = f", language: {language}" if language and language != "unknown" else ""
        role_part = f", document role: {document_role}" if document_role else ""
        excerpt = content_excerpt or "(no content available)"
        lang_reminder = (
            f"\n\nWrite the description in: {language}"
            if language and language != "unknown"
            else "\n\nDetect the language of the content excerpt above and write the description in that same language."
        )
        return (
            f"Document: {file_info}{lang_part}{role_part}\n\n"
            f"Content excerpt:\n{excerpt}{lang_reminder}"
        )

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
