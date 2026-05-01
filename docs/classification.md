# Document Classification (`core_lib.classification`)

`core_lib.classification` provides LLM-based document classification into one of 16 predefined categories, plus simultaneous generation of a RAG-optimised semantic description of the document.

---

## Module Layout

```
core_lib/classification/
├── __init__.py      # Exports DocumentClassifier, DocumentClassificationResult
├── classifier.py    # DocumentClassifier implementation
└── schemas.py       # DocumentClassificationResult Pydantic model
```

---

## Quick Start

```python
from core_lib.classification import DocumentClassifier

classifier = DocumentClassifier()

result = classifier.classify(
    filename="Q4_Annual_Report_2024.pdf",
    content_excerpt="Revenue increased by 12% year-over-year...",
    language="en",
    file_type="pdf",
)

print(result.category_id)    # "business_financial_reports"
print(result.confidence)     # 0.94
print(result.description)    # "This annual financial report covers Q4 2024..."
```

Top-level import also works:

```python
from core_lib import DocumentClassifier, DocumentClassificationResult
```

---

## `DocumentClassifier`

```python
class DocumentClassifier:
    def __init__(self, intelligence_level: int = 3) -> None: ...
    def classify(
        self,
        filename: str,
        content_excerpt: str,
        language: str = "unknown",
        file_type: Optional[str] = None,
    ) -> DocumentClassificationResult: ...
```

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `intelligence_level` | `int` | `3` | LLM tier to use. `3` is cheap/fast — appropriate for classification. Higher values route to more capable (and more expensive) models as configured in `llm_providers.yaml`. |

The LLM client is initialised lazily on the first `classify()` call.

### `classify()`

| Parameter | Type | Description |
|---|---|---|
| `filename` | `str` | Original document filename. Used as a naming and type hint by the LLM. |
| `content_excerpt` | `str` | Up to ~2 000 characters of raw document text. Longer text is truncated by the caller. |
| `language` | `str` | ISO 639-1 language code (`"en"`, `"fr"`, …) or `"unknown"`. Helps the LLM generate the `description` in the correct language. |
| `file_type` | `Optional[str]` | File extension without dot (`"pdf"`, `"docx"`, `"xlsx"`, …). Optional — the LLM can often infer type from the filename. |

**Returns:** `DocumentClassificationResult`

**Error behaviour:** Any exception is caught internally. On failure the method returns a safe default:

```python
DocumentClassificationResult(
    category_id="general",
    confidence=0.0,
    reasoning="Classification unavailable",
    description="",
    detection_method="default",
    alternative_categories=[],
)
```

---

## `DocumentClassificationResult`

Pydantic model (`BaseModel`) representing a classification outcome.

| Field | Type | Description |
|---|---|---|
| `category_id` | `str` | One of the 16 predefined category keys, or `"general"` on fallback. |
| `confidence` | `float` | Score between `0.0` and `1.0`. `0.0` when using default fallback. |
| `reasoning` | `str` | One-sentence justification for the chosen category. |
| `description` | `str` | 2–4 sentence semantic summary written in the document's own language. Optimised for RAG retrieval — describes what the document contains and what questions it can answer. |
| `detection_method` | `Literal["llm", "default"]` | `"llm"` when the LLM produced the result; `"default"` when the fallback was used. |
| `alternative_categories` | `List[Dict[str, Any]]` | Up to 2 runner-up candidates, each with `"category_id"` and `"confidence"` keys. Empty list when highly confident or on fallback. |

---

## Available Categories

The classifier maps documents to one of the 16 entries in `core_lib.config.doc_categories.DOC_CATEGORIES`:

| Key | Description |
|-----|-------------|
| `business_company_overview` | Company profile, history, mission, vision, and organizational structure |
| `business_financial_reports` | Annual reports, financial statements, balance sheets, and audit reports |
| `business_certifications_awards` | Industry certifications, awards, recognitions, and compliance documents |
| `sales_product_portfolio` | Product/service descriptions, technical specifications, and pricing sheets |
| `sales_customer_references` | Customer lists, testimonials, references, and satisfaction data |
| `sales_rfx` | RFP/RFI/RFQ documents that need to be answered |
| `marketing_business_cases` | Business cases, ROI analyses, and investment justifications |
| `marketing_customer_success_stories` | Success stories, project outcomes, and impact reports |
| `marketing_white_papers_innovation` | White papers, innovation reports, and thought-leadership content |
| `marketing_brochures_presentations` | Brochures, promotional materials, and slide decks |
| `technical_product_documentation` | Product manuals, technical guides, API documentation, and release notes |
| `technical_functional_specifications` | Functional specifications, requirements, and system design documents |
| `operations_sla_contracts` | SLA agreements, contracts, terms of service, and partnership agreements |
| `operations_security_compliance` | Security policies, data privacy, GDPR compliance, and audit frameworks |
| `operations_implementation_guides` | Implementation, onboarding, and deployment guides |
| `hr_team_expertise` | Team profiles, CVs, organizational charts, and HR documentation |

The LLM prompt is built dynamically from `DOC_CATEGORIES` at import time, so adding a new category there automatically updates the prompt.

---

## LLM Prompt

The classifier uses a single-turn structured-output call:

- **System prompt:** Lists all 16 categories with descriptions; requests a JSON object with `category_id`, `confidence`, `reasoning`, `description`, and `alternative_categories`.
- **User message:** `"Document: <filename> (<file_type> file), language: <language>\n\nContent excerpt:\n<excerpt>"`

Structured output is requested via `client.chat(..., structured_output=DocumentClassificationResult)`, which uses provider-native JSON mode or function calling depending on the LLM backend.

---

## LLM Provider Requirements

`DocumentClassifier` uses `create_fallback_llm_client(intelligence_level=3, usage="classify")` from `core_lib.llm`. Any configured LLM provider works:

| Provider | Required env var |
|---|---|
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_GENAI_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + related vars |
| Ollama | `OLLAMA_BASE_URL` or `OLLAMA_HOST` |

The tier routed to depends on `llm_providers.yaml` — see [FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md).

---

## Validation

After the LLM responds, the classifier validates that `category_id` is in `DOC_CATEGORIES`. If not, it substitutes `"general"` and logs a warning:

```
[WARNING] LLM returned unknown category_id 'some_key'; substituting 'general'
```

---

## Performance Notes

- **Latency:** typically 0.5–3 seconds depending on provider and excerpt length.
- **Token usage:** ~500–1 000 input tokens per call.
- **Lazy init:** The LLM client is created on the first `classify()` call, not at import time.
- **Skip classification:** Pass an explicit `category_id` in your application logic to bypass the classifier entirely.

---

## Related

- [`core_lib/config/doc_categories.py`](../core_lib/config/doc_categories.py) — Category definitions
- [`core_lib/llm/`](../core_lib/llm/) — LLM client infrastructure
- [FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md) — How provider tiers are selected
- [mcp-doc-qa: auto-category-detection](../../mcp-doc-qa/docs/auto-category-detection.md) — How the classifier is integrated into the ingestion pipeline
