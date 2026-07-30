"""LLM classifier for worksheet-level questionnaire topic profiles."""
from __future__ import annotations

from typing import Optional, Sequence

from core_lib.llm import create_fallback_llm_client
from core_lib.tracing import get_module_logger

from .schemas import TopicProfile

logger = get_module_logger()

_SYSTEM_PROMPT = """You classify the semantic scope of one RFx questionnaire worksheet.
Return a structured topic profile. Focus on the product modules, business processes, capabilities,
and kinds of questions represented across the complete worksheet. Use open-vocabulary, precise topic
names. Distinguish nearby modules such as purchase requisitions, supplier management, and e-invoicing.
If the questionnaire explicitly names a product or module (for example, CloudSync Enterprise),
product_areas MUST include that exact name; generic categories may be added but must not replace it.
The description must explain what evidence would be relevant for answering this worksheet. Do not infer
that a feature is in scope merely because it is common in the wider product. Write the description and
terms in the worksheet language. document_category is the broad document type and should normally be
sales_rfx; it is metadata, not a source-document equality filter. classifier_version must be
questionnaire-topic-v1 and detection_method must be llm."""


class QuestionnaireTopicClassifier:
    """Create one reusable topic profile from a complete questionnaire worksheet."""

    def __init__(self, intelligence_level: int = 3, max_questions: int = 40) -> None:
        self._intelligence_level = intelligence_level
        self._max_questions = max(3, max_questions)
        self._client = None

    def classify(
        self,
        workbook_name: str,
        sheet_name: str,
        questions: Sequence[str],
        instructions: Optional[str] = None,
        section_headings: Optional[Sequence[str]] = None,
        language: str = "unknown",
    ) -> TopicProfile:
        try:
            sampled = self.sample_questions(questions, self._max_questions)
            message = self._build_user_message(
                workbook_name=workbook_name,
                sheet_name=sheet_name,
                questions=sampled,
                instructions=instructions,
                section_headings=section_headings,
                language=language,
            )
            response = self._get_client().chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                structured_output=TopicProfile,
            )
            content = response.get("content")
            if isinstance(content, dict):
                content = TopicProfile.model_validate(content)
            if not isinstance(content, TopicProfile):
                raise TypeError(f"Unexpected questionnaire classification response: {type(content).__name__}")
            content.detection_method = "llm"
            content.classifier_version = "questionnaire-topic-v1"
            explicit_products = self.extract_explicit_product_names(questions)
            # Reuse the TopicProfile term normalization after merging. Explicit names
            # come first so they are retained when the profile reaches its 20-term cap.
            content.product_areas = TopicProfile.normalize_terms(
                [*explicit_products, *content.product_areas]
            )
            content.fingerprint = content.compute_fingerprint()
            return content
        except Exception as exc:
            logger.warning("Questionnaire topic classification failed for '%s/%s': %s", workbook_name, sheet_name, exc)
            return self.default_profile(workbook_name, sheet_name, language)

    def _get_client(self):
        if self._client is None:
            self._client = create_fallback_llm_client(
                intelligence_level=self._intelligence_level,
                usage="classify",
            )
        return self._client

    @staticmethod
    def sample_questions(questions: Sequence[str], limit: int = 40) -> list[str]:
        cleaned = [" ".join(str(q).split()).strip() for q in questions if str(q).strip()]
        if limit <= 0:
            return []
        if len(cleaned) <= limit:
            return cleaned
        if limit == 1:
            return [cleaned[0]]
        # Even sampling covers the beginning, middle, and end while remaining deterministic.
        indices = {round(i * (len(cleaned) - 1) / (limit - 1)) for i in range(limit)}
        return [cleaned[i] for i in sorted(indices)]

    @staticmethod
    def extract_explicit_product_names(questions: Sequence[str]) -> list[str]:
        """Preserve branded product names that the LLM may generalize away."""
        import re

        patterns = (
            r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]+){0,2}",
            r"\b[A-Z]{2,}(?:\s+[A-Z][a-z][A-Za-z0-9]*){1,2}",
        )
        found: list[str] = []
        seen: set[str] = set()
        for question in questions:
            for pattern in patterns:
                for match in re.findall(pattern, str(question)):
                    value = " ".join(match.split()).strip(" ,.;:?!")
                    key = value.casefold()
                    if value and key not in seen:
                        found.append(value)
                        seen.add(key)
        return found[:10]

    @staticmethod
    def _build_user_message(
        workbook_name: str,
        sheet_name: str,
        questions: Sequence[str],
        instructions: Optional[str],
        section_headings: Optional[Sequence[str]],
        language: str,
    ) -> str:
        parts = [f"Workbook: {workbook_name}", f"Worksheet: {sheet_name}", f"Language: {language or 'unknown'}"]
        if section_headings:
            parts.append("Section headings:\n" + "\n".join(f"- {h}" for h in section_headings if h))
        if instructions:
            parts.append("Instructions:\n" + instructions[:6000])
        parts.append("Representative questions from the complete worksheet:\n" + "\n".join(
            f"{i}. {question[:1000]}" for i, question in enumerate(questions, 1)
        ))
        return "\n\n".join(parts)[:24000]

    @staticmethod
    def default_profile(workbook_name: str, sheet_name: str, language: str = "unknown") -> TopicProfile:
        description = " ".join(v for v in [workbook_name, sheet_name] if v).strip()
        return TopicProfile(
            description=description,
            document_category="sales_rfx",
            language=language or "unknown",
            confidence=0.0,
            reasoning="Questionnaire classification unavailable",
            detection_method="default",
            classifier_version="questionnaire-topic-v1",
        )
