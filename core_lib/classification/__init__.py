"""Document classification module for core-lib.

Provides LLM-based document classification and RAG description generation.

Example usage::

    from core_lib.classification import DocumentClassifier, DocumentClassificationResult

    classifier = DocumentClassifier()
    result = classifier.classify(
        filename="product_spec.pdf",
        content_excerpt="...",
        language="en",
        file_type="pdf",
    )
    print(result.category_id, result.confidence)
"""

from .classifier import DocumentClassifier
from .questionnaire_classifier import QuestionnaireTopicClassifier
from .schemas import DocumentClassificationResult, TopicProfile

__all__ = [
    "DocumentClassifier",
    "DocumentClassificationResult",
    "QuestionnaireTopicClassifier",
    "TopicProfile",
]
