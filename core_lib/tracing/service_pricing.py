"""Pricing data for AI services (LLM, embeddings, OCR).

This module contains pricing information for various AI service providers.
Update these dictionaries when providers change their pricing.

Pricing is per 1,000 tokens unless otherwise noted.
All prices in USD.

Last updated: May 26, 2026
"""

# LLM Pricing per 1K tokens (USD)
# Format: {"model-name": {"input": price_per_1k, "output": price_per_1k}}
LLM_PRICING = {
    # OpenAI models
    # Source: https://developers.openai.com/api/docs/pricing (updated May 26, 2026)
    # Prices per 1M tokens in API docs, converted to per 1K here.
    # GPT-5.5 series (flagship models)
    "gpt-5.5": {"input": 0.005, "output": 0.03},          # $5.00/$30.00 per 1M
    "gpt-5.5-pro": {"input": 0.03, "output": 0.18},        # $30.00/$180.00 per 1M
    # GPT-5.4 series
    "gpt-5.4": {"input": 0.0025, "output": 0.015},         # $2.50/$15.00 per 1M
    "gpt-5.4-mini": {"input": 0.00075, "output": 0.0045},  # $0.75/$4.50 per 1M
    "gpt-5.4-nano": {"input": 0.0002, "output": 0.00125},  # $0.20/$1.25 per 1M
    "gpt-5.4-pro": {"input": 0.03, "output": 0.18},        # $30.00/$180.00 per 1M
    # GPT-5.3 series
    "gpt-5.3-codex": {"input": 0.00175, "output": 0.014},  # $1.75/$14.00 per 1M
    # GPT-4.1 series
    "gpt-4.1": {"input": 0.003, "output": 0.012},
    "gpt-4.1-mini": {"input": 0.0008, "output": 0.0032},
    "gpt-4.1-nano": {"input": 0.0002, "output": 0.0008},
    # o4 series (reasoning models)
    "o4-mini": {"input": 0.004, "output": 0.016},          # $4.00/$16.00 per 1M
    
    # Google Gemini models
    # Source: https://ai.google.dev/gemini-api/docs/pricing (updated May 26, 2026)
    # Prices per 1M tokens in API docs, converted to per 1K here.
    # Preview variants (e.g. gemini-3.1-pro-preview) are resolved automatically
    # by the fuzzy matcher in get_llm_pricing() — no need to list them separately.
    # Gemini 3.5 models
    "gemini-3.5-flash": {"input": 0.0015, "output": 0.009},       # $1.50/$9.00 per 1M
    # Gemini 3.1 models
    "gemini-3.1-pro": {"input": 0.002, "output": 0.012},        # $2.00/$12.00 per 1M (<= 200k prompts)
    "gemini-3.1-flash-lite": {"input": 0.00025, "output": 0.0015},  # $0.25/$1.50 per 1M
    # Gemini 3 models
    "gemini-3-pro": {"input": 0.002, "output": 0.012},           # deprecated/shut down 2026-03-09
    "gemini-3-flash": {"input": 0.0005, "output": 0.003},        # $0.50/$3.00 per 1M
    # Gemini 2.5 models
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},        # $1.25/$10.00 per 1M (<= 200k prompts)
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},     # $0.30/$2.50 per 1M
    "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},  # $0.10/$0.40 per 1M
    # Gemini 2.0 models
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},     # $0.10/$0.40 per 1M
    "gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.0003},  # $0.075/$0.30 per 1M
    # Gemma models (open models, free)
    "gemma-3": {"input": 0.0, "output": 0.0},
    "gemma-3n": {"input": 0.0, "output": 0.0},
    
    # Anthropic Claude models
    # Source: https://platform.claude.com/docs/en/about-claude/pricing (updated May 26, 2026)
    # Prices per 1M tokens in API docs, converted to per 1K here.
    # Claude Opus 4.x series
    "claude-opus-4-7": {"input": 0.005, "output": 0.025},   # $5/$25 per 1M
    "claude-opus-4-6": {"input": 0.005, "output": 0.025},   # $5/$25 per 1M
    "claude-opus-4-5": {"input": 0.005, "output": 0.025},   # $5/$25 per 1M
    "claude-opus-4-1": {"input": 0.015, "output": 0.075},   # $15/$75 per 1M
    # Claude Sonnet 4.x series
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015}, # $3/$15 per 1M
    "claude-sonnet-4-5": {"input": 0.003, "output": 0.015}, # $3/$15 per 1M
    "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
    # Claude Haiku 4.x series
    "claude-haiku-4-5": {"input": 0.001, "output": 0.005},  # $1/$5 per 1M
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
    # Claude Haiku 3.5 (retired, still on Bedrock/Vertex AI)
    "claude-haiku-3-5": {"input": 0.0008, "output": 0.004}, # $0.80/$4 per 1M
    
    # Azure OpenAI (same as OpenAI pricing)
    "azure-gpt-4": {"input": 0.03, "output": 0.06},
    "azure-gpt-4o": {"input": 0.005, "output": 0.015},
    "azure-gpt-35-turbo": {"input": 0.0005, "output": 0.0015},
    
    # Local/Self-hosted (free)
    "ollama": {"input": 0.0, "output": 0.0},

    # OVHcloud AI Endpoints
    # Source: https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/
    # Prices in EUR; converted to USD at ~1.10 USD/EUR. Last updated: May 2026.
    "Qwen3.5-9B": {"input": 0.00011, "output": 0.000165},        # 0.10€/$0.15€ per 1M tokens
    "Qwen3.5-397B-A17B": {"input": 0.00066, "output": 0.00396},  # 0.60€/3.60€ per 1M tokens

    # Mistral AI models
    # Source: https://mistral.ai/pricing#api (updated May 2026)
    # Prices per 1M tokens → converted to per 1K by dividing by 1000.
    # Ministral (edge) models — text + vision + agentic
    "ministral-3b-latest": {"input": 0.0001, "output": 0.0001},        # $0.10/$0.10 per 1M
    "ministral-8b-latest": {"input": 0.00015, "output": 0.00015},      # $0.15/$0.15 per 1M
    "ministral-14b-latest": {"input": 0.0002, "output": 0.0002},       # $0.20/$0.20 per 1M
    # Mistral Small — multimodal, reasoning, lightweight
    "mistral-small-latest": {"input": 0.00015, "output": 0.0006},      # $0.15/$0.60 per 1M
    # Mistral Medium — multimodal, agentic
    "mistral-medium-latest": {"input": 0.0015, "output": 0.0075},      # $1.50/$7.50 per 1M
    # Mistral Large — multimodal, flagship
    "mistral-large-latest": {"input": 0.0005, "output": 0.0015},       # $0.50/$1.50 per 1M
    # Magistral thinking/reasoning models
    "magistral-medium-latest": {"input": 0.002, "output": 0.005},      # $2.00/$5.00 per 1M
    "magistral-small-latest": {"input": 0.0005, "output": 0.0015},     # $0.50/$1.50 per 1M
}


# Embedding Pricing per 1K tokens (USD)
# Format: {"model-name": price_per_1k}
EMBEDDING_PRICING = {
    # OpenAI embeddings
    # Source: https://openai.com/api/pricing/
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    "text-embedding-ada-002": 0.0001,
    
    # Google Gemini embeddings
    # Source: https://ai.google.dev/gemini-api/docs/pricing
    "gemini-embedding-001": 0.00015,  # $0.15 per 1M tokens
    "text-embedding-004": 0.00001,  # Legacy model
    "embedding-001": 0.00001,  # Legacy model
    
    # Cohere embeddings
    # Source: https://cohere.com/pricing
    "embed-english-v3.0": 0.0001,
    "embed-multilingual-v3.0": 0.0001,
    "embed-english-light-v3.0": 0.00001,
    "embed-multilingual-light-v3.0": 0.00001,
    
    # Azure OpenAI embeddings (same as OpenAI)
    "azure-text-embedding-3-small": 0.00002,
    "azure-text-embedding-3-large": 0.00013,
    "azure-text-embedding-ada-002": 0.0001,
    
    # Local/Self-hosted (free)
    "infinity": 0.0,
    "ollama": 0.0,
}


# OCR Pricing
# Format: {"provider/model": {"per_page": price, "per_image": price}}
# Note: Actual pricing varies by volume, region, and features
OCR_PRICING = {
    # Azure Document Intelligence
    # Source: https://azure.microsoft.com/en-us/pricing/details/form-recognizer/
    "azure-di/prebuilt-read": {"per_page": 0.001},  # $1 per 1000 pages
    "azure-di/prebuilt-layout": {"per_page": 0.01},  # $10 per 1000 pages
    "azure-di/prebuilt-document": {"per_page": 0.01},
    "azure-di/prebuilt-invoice": {"per_page": 0.01},
    "azure-di/prebuilt-receipt": {"per_page": 0.01},
    "azure-di/custom": {"per_page": 0.04},  # $40 per 1000 pages
    
    # Google Cloud Vision
    # Source: https://cloud.google.com/vision/pricing
    "google-vision/ocr": {"per_image": 0.0015},  # $1.50 per 1000 images
    "google-vision/document-text": {"per_image": 0.006},  # $6 per 1000 images
    
    # AWS Textract
    # Source: https://aws.amazon.com/textract/pricing/
    "aws-textract/detect-text": {"per_page": 0.0015},
    "aws-textract/analyze-document": {"per_page": 0.05},
    
    # Local/Self-hosted (free)
    "tesseract": {"per_page": 0.0},
    "paddleocr": {"per_page": 0.0},
}


# Helper function to get pricing info
def get_llm_pricing(model: str) -> dict:
    """Get pricing for an LLM model.

    Matching is fuzzy so that preview suffixes, date stamps, and tool-variant
    qualifiers are stripped progressively:

    1. Exact match (e.g. ``gemini-3.1-flash-lite``)
    2. Strip trailing date stamp (e.g. ``-09-2025``) then exact match
    3. Truncate to the first ``-preview`` segment then exact match
       (e.g. ``gemini-3.1-pro-preview-customtools`` → ``gemini-3.1-pro-preview``)
    4. Remove ``-preview`` and everything after, then exact match
       (e.g. ``gemini-3.1-pro-preview`` → ``gemini-3.1-pro``)
    5. Prefix match: the known model key is a prefix of the requested name

    Args:
        model: Model name (case-insensitive)

    Returns:
        Dictionary with 'input' and 'output' prices per 1K tokens,
        or None if model not found
    """
    import re

    model_key = model.lower()

    # 1. Exact match
    if model_key in LLM_PRICING:
        return LLM_PRICING[model_key]

    # 2. Strip trailing date stamp suffix, e.g. "-09-2025" or "-12-2025"
    without_date = re.sub(r'-\d{2}-\d{4}$', '', model_key)
    if without_date != model_key:
        if without_date in LLM_PRICING:
            return LLM_PRICING[without_date]
        model_key = without_date  # use date-stripped form for further steps

    # 3. Truncate to the first "-preview" occurrence (drops qualifiers like "-customtools")
    preview_base_match = re.match(r'^(.*?-preview)', model_key)
    if preview_base_match:
        preview_base = preview_base_match.group(1)
        if preview_base != model_key and preview_base in LLM_PRICING:
            return LLM_PRICING[preview_base]

    # 4. Drop "-preview" and everything after to find the stable model entry
    without_preview = re.sub(r'-preview.*$', '', model_key)
    if without_preview != model_key and without_preview in LLM_PRICING:
        return LLM_PRICING[without_preview]

    # 5. Forward prefix: known model key is a prefix of the requested name
    #    e.g. "gemini-2.5-flash" matches "gemini-2.5-flash-thinking-exp"
    for known_model, pricing in LLM_PRICING.items():
        if model_key.startswith(known_model):
            return pricing

    # 6. Reverse prefix: requested name is a prefix of a known model key
    #    e.g. "gemini-3.1-pro" matches "gemini-3.1-pro-preview"
    #    Use the longest matching known key to pick the most specific entry.
    best_match = None
    for known_model, pricing in LLM_PRICING.items():
        if known_model.startswith(model_key) and (best_match is None or len(known_model) < len(best_match)):
            best_match = known_model
    if best_match is not None:
        return LLM_PRICING[best_match]

    return None


def get_embedding_pricing(model: str) -> float:
    """Get pricing for an embedding model.
    
    Args:
        model: Model name (case-insensitive)
        
    Returns:
        Price per 1K tokens, or None if model not found
    """
    model_key = model.lower()
    
    # Try exact match first
    if model_key in EMBEDDING_PRICING:
        return EMBEDDING_PRICING[model_key]
    
    # Try partial match
    for known_model, price in EMBEDDING_PRICING.items():
        if known_model in model_key or model_key.startswith(known_model.split("-")[0]):
            return price
    
    return None


def get_ocr_pricing(provider: str, model: str) -> dict:
    """Get pricing for an OCR service.
    
    Args:
        provider: Provider name (e.g., "azure-di", "google-vision")
        model: Model/service name
        
    Returns:
        Dictionary with 'per_page' or 'per_image' pricing,
        or None if not found
    """
    key = f"{provider}/{model}".lower()
    return OCR_PRICING.get(key)


__all__ = [
    "LLM_PRICING",
    "EMBEDDING_PRICING",
    "OCR_PRICING",
    "get_llm_pricing",
    "get_embedding_pricing",
    "get_ocr_pricing",
]
