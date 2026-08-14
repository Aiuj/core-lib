"""Unit tests for core_lib.classification.

All LLM calls are mocked — no network access required.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from core_lib.classification import DocumentClassifier, DocumentClassificationResult
from core_lib.config.doc_categories import DOC_CATEGORIES


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VALID_CATEGORY_KEY = "business_financial_reports"
VALID_CATEGORY_KEY_2 = "sales_rfx"


def _make_result(**kwargs) -> DocumentClassificationResult:
    """Build a valid DocumentClassificationResult with sensible defaults."""
    defaults = dict(
        category_id=VALID_CATEGORY_KEY,
        confidence=0.88,
        reasoning="Looks like a financial report.",
        description="This document is a quarterly financial report covering revenue...",
        detection_method="llm",
        alternative_categories=[],
    )
    defaults.update(kwargs)
    return DocumentClassificationResult(**defaults)


def _mock_client(result: DocumentClassificationResult) -> MagicMock:
    """Return a mock LLM client whose .chat() returns the given result."""
    client = MagicMock()
    client.chat.return_value = {"content": result}
    return client


# ---------------------------------------------------------------------------
# DocumentClassificationResult schema
# ---------------------------------------------------------------------------

class TestDocumentClassificationResultSchema:
    def test_valid_construction(self):
        result = _make_result()
        assert result.category_id == VALID_CATEGORY_KEY
        assert result.confidence == 0.88
        assert result.detection_method == "llm"
        assert result.alternative_categories == []
        assert result.primary_topics == []
        assert result.content_structure == "unknown"
        assert result.structure_confidence == 0.0
        assert result.pairing_pattern == "unknown"
        assert result.prospect_document_type is None
        assert result.prospect_document_type_confidence == 0.0

    def test_structure_fields_are_topic_independent(self):
        result = _make_result(
            content_structure="qa_pairs",
            structure_confidence=0.94,
            pairing_pattern="alternating_blocks",
        )
        assert result.category_id == VALID_CATEGORY_KEY
        assert result.content_structure == "qa_pairs"
        assert result.structure_confidence == 0.94

    def test_prospect_rfx_type_is_validated(self):
        result = _make_result(
            prospect_document_type="requirements_specification",
            prospect_document_type_confidence=0.96,
        )
        assert result.prospect_document_type == "requirements_specification"
        assert result.prospect_document_type_confidence == 0.96

    def test_invalid_prospect_rfx_type_is_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_result(prospect_document_type="random_document")

    def test_default_detection_method_is_llm(self):
        result = DocumentClassificationResult(
            category_id="sales_rfx",
            confidence=0.5,
            reasoning="r",
            description="d",
        )
        assert result.detection_method == "llm"

    def test_confidence_bounds_lower(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DocumentClassificationResult(
                category_id="sales_rfx",
                confidence=-0.1,
                reasoning="r",
                description="d",
            )

    def test_confidence_bounds_upper(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DocumentClassificationResult(
                category_id="sales_rfx",
                confidence=1.01,
                reasoning="r",
                description="d",
            )

    def test_detection_method_rejects_invalid(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DocumentClassificationResult(
                category_id="sales_rfx",
                confidence=0.5,
                reasoning="r",
                description="d",
                detection_method="rule",  # type: ignore[arg-type]
            )

    def test_alternative_categories_default_empty(self):
        result = DocumentClassificationResult(
            category_id="sales_rfx",
            confidence=0.9,
            reasoning="r",
            description="d",
        )
        assert result.alternative_categories == []

    def test_alternative_categories_stored(self):
        alts = [{"category_id": "sales_product_portfolio", "confidence": 0.3}]
        result = _make_result(alternative_categories=alts)
        assert result.alternative_categories == alts


# ---------------------------------------------------------------------------
# DocumentClassifier — construction and lazy init
# ---------------------------------------------------------------------------

class TestDocumentClassifierInit:
    def test_default_intelligence_level(self):
        clf = DocumentClassifier()
        assert clf._intelligence_level == 3

    def test_custom_intelligence_level(self):
        clf = DocumentClassifier(intelligence_level=7)
        assert clf._intelligence_level == 7

    def test_client_is_none_before_first_call(self):
        clf = DocumentClassifier()
        assert clf._client is None

    def test_client_created_lazily(self):
        clf = DocumentClassifier()
        mock_client = _mock_client(_make_result())
        clf._client = mock_client  # inject directly to avoid create_fallback_llm_client
        # Calling classify should use the injected client without re-creating it
        clf.classify(filename="test.pdf", content_excerpt="Revenue up 12%")
        assert mock_client.chat.call_count == 2  # classification plus bounded scope enrichment

    def test_client_reused_on_second_call(self):
        clf = DocumentClassifier()
        mock_client = _mock_client(_make_result())
        clf._client = mock_client
        clf.classify(filename="a.pdf", content_excerpt="text1")
        clf.classify(filename="b.pdf", content_excerpt="text2")
        assert mock_client.chat.call_count == 4  # same client, classification plus enrichment per call

    def test_top_level_import(self):
        """Ensure the module is importable from the package top level."""
        from core_lib import DocumentClassifier as DC, DocumentClassificationResult as DCR  # noqa
        assert DC is DocumentClassifier
        assert DCR is DocumentClassificationResult


# ---------------------------------------------------------------------------
# DocumentClassifier.classify — happy path
# ---------------------------------------------------------------------------

class TestDocumentClassifierClassifyHappyPath:
    def _classifier_with_result(self, result: DocumentClassificationResult) -> DocumentClassifier:
        clf = DocumentClassifier()
        clf._client = _mock_client(result)
        return clf

    def test_returns_llm_result(self):
        expected = _make_result(category_id="sales_rfx", confidence=0.92)
        clf = self._classifier_with_result(expected)
        result = clf.classify(filename="rfp.xlsx", content_excerpt="Questions about security")
        assert result.category_id == "sales_rfx"
        assert result.confidence == 0.92
        assert result.detection_method == "llm"

    def test_all_fields_propagated(self):
        alts = [{"category_id": "sales_product_portfolio", "confidence": 0.1}]
        expected = _make_result(
            category_id="technical_product_documentation",
            confidence=0.78,
            reasoning="Contains technical API docs.",
            description="A detailed API reference guide.",
            alternative_categories=alts,
            primary_topics=["REST APIs"],
            product_areas=["Developer platform"],
            capabilities=["API integration"],
            prospect_document_type="technical_requirements",
            prospect_document_type_confidence=0.82,
        )
        clf = self._classifier_with_result(expected)
        result = clf.classify(filename="api_guide.pdf", content_excerpt="REST endpoints")
        assert result.reasoning == "Contains technical API docs."
        assert result.description == "A detailed API reference guide."
        assert result.alternative_categories == alts
        assert result.primary_topics == ["REST APIs"]
        assert result.prospect_document_type == "technical_requirements"
        assert result.prospect_document_type_confidence == 0.82

    def test_retries_once_when_scope_terms_are_missing(self):
        initial = _make_result(
            description="Security compliance evidence for customer audits.",
        )
        enriched = _make_result(
            primary_topics=["cybersecurity compliance"],
            capabilities=["ISO 27001 and SOC 2 audit evidence"],
        )
        clf = DocumentClassifier()
        clf._client = MagicMock()
        clf._client.chat.side_effect = [{"content": initial}, {"content": enriched}]

        result = clf.classify(filename="security.pdf", content_excerpt="ISO 27001 and SOC 2")

        assert clf._client.chat.call_count == 2
        assert result.primary_topics == ["cybersecurity compliance"]
        assert result.capabilities == ["ISO 27001 and SOC 2 audit evidence"]

    def test_calls_chat_with_system_and_user_messages(self):
        clf = self._classifier_with_result(_make_result())
        clf.classify(filename="doc.pdf", content_excerpt="some text", language="fr", file_type="pdf")
        call_args = clf._client.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user"]

    def test_structured_output_passed(self):
        clf = self._classifier_with_result(_make_result())
        clf.classify(filename="doc.pdf", content_excerpt="text")
        call_kwargs = clf._client.chat.call_args.kwargs
        assert call_kwargs.get("structured_output") is DocumentClassificationResult

    def test_every_valid_category_accepted(self):
        """classify() must return any valid DOC_CATEGORIES key without substituting."""
        valid_keys = [cat["key"] for cat in DOC_CATEGORIES]
        for key in valid_keys:
            clf = DocumentClassifier()
            clf._client = _mock_client(_make_result(category_id=key))
            result = clf.classify(filename="x.pdf", content_excerpt="content")
            assert result.category_id == key, f"Expected {key!r}, got {result.category_id!r}"


# ---------------------------------------------------------------------------
# DocumentClassifier.classify — validation: unknown category_id
# ---------------------------------------------------------------------------

class TestDocumentClassifierCategoryValidation:
    def test_unknown_category_substituted_with_general(self):
        bad_result = _make_result(category_id="totally_unknown_category", confidence=0.7)
        clf = DocumentClassifier()
        clf._client = _mock_client(bad_result)
        result = clf.classify(filename="x.pdf", content_excerpt="text")
        assert result.category_id == "general"

    def test_unknown_category_preserves_other_fields(self):
        bad_result = _make_result(
            category_id="not_a_real_category",
            confidence=0.65,
            reasoning="some reasoning",
            description="some description",
            prospect_document_type="requirements_specification",
            prospect_document_type_confidence=0.91,
        )
        clf = DocumentClassifier()
        clf._client = _mock_client(bad_result)
        result = clf.classify(filename="x.pdf", content_excerpt="text")
        assert result.category_id == "general"
        assert result.confidence == 0.65
        assert result.reasoning == "some reasoning"
        assert result.description == "some description"
        assert result.detection_method == "llm"  # stays llm, not default
        assert result.prospect_document_type == "requirements_specification"
        assert result.prospect_document_type_confidence == 0.91


# ---------------------------------------------------------------------------
# DocumentClassifier.classify — fallback on errors
# ---------------------------------------------------------------------------

class TestDocumentClassifierFallback:
    def test_llm_exception_returns_default(self):
        clf = DocumentClassifier()
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("LLM unavailable")
        clf._client = mock_client
        result = clf.classify(filename="fail.pdf", content_excerpt="text")
        assert result.category_id == "general"
        assert result.confidence == 0.0
        assert result.detection_method == "default"

    def test_non_model_response_returns_default(self):
        """When chat returns content that is not a DocumentClassificationResult."""
        clf = DocumentClassifier()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"content": {"category_id": "sales_rfx"}}  # dict, not model
        clf._client = mock_client
        result = clf.classify(filename="x.pdf", content_excerpt="text")
        assert result.category_id == "general"
        assert result.detection_method == "default"

    def test_none_response_returns_default(self):
        clf = DocumentClassifier()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"content": None}
        clf._client = mock_client
        result = clf.classify(filename="x.pdf", content_excerpt="text")
        assert result.category_id == "general"
        assert result.detection_method == "default"

    def test_empty_excerpt_still_classifies(self):
        clf = DocumentClassifier()
        clf._client = _mock_client(_make_result())
        result = clf.classify(filename="doc.pdf", content_excerpt="")
        assert result.category_id == VALID_CATEGORY_KEY

    def test_default_result_fields(self):
        result = DocumentClassifier._default_result()
        assert result.category_id == "general"
        assert result.confidence == 0.0
        assert result.reasoning == "Classification unavailable"
        assert result.description == ""
        assert result.detection_method == "default"
        assert result.alternative_categories == []


# ---------------------------------------------------------------------------
# DocumentClassifier._build_user_message
# ---------------------------------------------------------------------------

class TestBuildUserMessage:
    """Tests for the static _build_user_message helper."""

    def _call(self, filename, excerpt, language="unknown", file_type=None):
        return DocumentClassifier._build_user_message(filename, excerpt, language, file_type)

    def test_filename_appears(self):
        msg = self._call("report.pdf", "Revenue up")
        assert "report.pdf" in msg

    def test_file_type_appended_uppercase(self):
        msg = self._call("report.pdf", "Revenue up", file_type="pdf")
        assert "(PDF file)" in msg

    def test_no_file_type_no_parenthesis(self):
        msg = self._call("report.pdf", "Revenue up")
        assert "(" not in msg.split("Content excerpt:")[0]  # no type annotation before excerpt

    def test_language_included_when_known(self):
        msg = self._call("doc.docx", "text", language="fr")
        assert "language: fr" in msg

    def test_language_omitted_when_unknown(self):
        msg = self._call("doc.docx", "text", language="unknown")
        assert "language:" not in msg

    def test_language_omitted_when_empty(self):
        msg = self._call("doc.docx", "text", language="")
        assert "language:" not in msg

    def test_excerpt_included(self):
        msg = self._call("doc.pdf", "Annual revenue was $10M")
        assert "Annual revenue was $10M" in msg

    def test_document_role_hint_is_included(self):
        msg = DocumentClassifier._build_user_message(
            "requirements.pdf",
            "Buyer requirements",
            "en",
            "pdf",
            "prospect_context",
        )
        assert "document role: prospect_context" in msg

    def test_empty_excerpt_replaced_with_placeholder(self):
        msg = self._call("doc.pdf", "")
        assert "(no content available)" in msg

    def test_none_excerpt_replaced_with_placeholder(self):
        msg = self._call("doc.pdf", None)  # type: ignore[arg-type]
        assert "(no content available)" in msg

    def test_full_message_structure(self):
        msg = self._call("spec.xlsx", "Column A: Questions", language="en", file_type="xlsx")
        assert msg.startswith("Document: spec.xlsx (XLSX file), language: en")
        assert "Content excerpt:" in msg
        assert "Column A: Questions" in msg


# ---------------------------------------------------------------------------
# DOC_CATEGORIES structure (classifier integration)
# ---------------------------------------------------------------------------

class TestDocCategoriesIntegration:
    """Ensure the classifier is consistent with DOC_CATEGORIES."""

    def test_all_categories_present(self):
        assert len(DOC_CATEGORIES) == 17

    def test_every_category_has_required_fields(self):
        required = {"key", "label", "description"}
        for cat in DOC_CATEGORIES:
            assert required.issubset(cat.keys()), f"Missing fields in category: {cat}"

    def test_all_keys_are_unique(self):
        keys = [cat["key"] for cat in DOC_CATEGORIES]
        assert len(keys) == len(set(keys))

    def test_system_prompt_contains_all_keys(self):
        from core_lib.classification.classifier import _CATEGORIES_LIST
        for cat in DOC_CATEGORIES:
            assert cat["key"] in _CATEGORIES_LIST, (
                f"Category key '{cat['key']}' missing from classifier prompt"
            )

    def test_valid_keys_include_general(self):
        """The general category is a valid low-specificity classification."""
        keys = {cat["key"] for cat in DOC_CATEGORIES}
        assert "general" in keys
