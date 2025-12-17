"""
Comprehensive database of embedding models with their specifications.

This database includes:
- Default vector dimensions
- Context/token limits
- Whether the model supports Matryoshka Representation Learning (MRL)
- Provider information
- Query/passage prefixes for asymmetric retrieval models
"""

from typing import Dict, Optional, Tuple, TypedDict


class ModelSpec(TypedDict, total=False):
    """Type definition for model specifications."""
    dimensions: int
    context_size: int
    supports_matryoshka: bool
    provider: str
    supports_custom_dimensions: bool
    query_prefix: str  # Prefix for queries (e.g., "query: " for E5 models)
    passage_prefix: str  # Prefix for passages/documents (e.g., "passage: " for E5 models)
    notes: str


# Model family prefix patterns for auto-detection
# These are used when a model isn't in the database but matches a known family
MODEL_FAMILY_PREFIXES: Dict[str, Tuple[str, str]] = {
    # E5 models (intfloat) - multilingual and English
    "e5": ("query: ", "passage: "),
    "multilingual-e5": ("query: ", "passage: "),
    # BGE models (BAAI) - query instruction prefix only
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    # Instructor models
    "instructor": ("Represent the query for retrieval: ", "Represent the document for retrieval: "),
    # GTE models (Alibaba)
    "gte": ("query: ", ""),
}


# Comprehensive model database
EMBEDDING_MODELS_DATABASE: Dict[str, ModelSpec] = {
    # OpenAI Models
    "text-embedding-3-small": {
        "dimensions": 1536,
        "context_size": 8191,
        "supports_matryoshka": True,
        "supports_custom_dimensions": True,
        "provider": "openai",
        "notes": "Recommended for most use cases. Supports dimension shortening.",
    },
    "text-embedding-3-large": {
        "dimensions": 3072,
        "context_size": 8191,
        "supports_matryoshka": True,
        "supports_custom_dimensions": True,
        "provider": "openai",
        "notes": "Highest quality OpenAI embeddings. Supports dimension shortening.",
    },
    "text-embedding-ada-002": {
        "dimensions": 1536,
        "context_size": 8191,
        "supports_matryoshka": False,
        "supports_custom_dimensions": False,
        "provider": "openai",
        "notes": "Legacy model. Does not support dimension customization.",
    },
    
    # Nomic Models
    "nomic-embed-text": {
        "dimensions": 768,
        "context_size": 8192,
        "supports_matryoshka": True,
        "provider": "nomic",
        "notes": "Long context, Matryoshka representation learning.",
    },
    
    # Jina AI Models
    "jinaai/jina-embeddings-v2-base-en": {
        "dimensions": 768,
        "context_size": 8192,
        "supports_matryoshka": True,
        "provider": "jina",
        "notes": "English-only, supports MRL.",
    },
    "jina-embeddings-v2-base-en": {
        "dimensions": 768,
        "context_size": 8192,
        "supports_matryoshka": True,
        "provider": "jina",
        "notes": "English-only, supports MRL.",
    },
    "jina-embeddings-v2": {
        "dimensions": 768,
        "context_size": 8192,
        "supports_matryoshka": True,
        "provider": "jina",
        "notes": "Multilingual support with MRL.",
    },
    "jina-embeddings-v3": {
        "dimensions": 1024,
        "context_size": 8192,
        "supports_matryoshka": True,
        "provider": "jina",
        "notes": "Latest version with improved quality.",
    },
    "jina-colbert-v2": {
        "dimensions": 128,
        "context_size": 8192,
        "supports_matryoshka": True,
        "provider": "jina",
        "notes": "Multi-vector embeddings for advanced retrieval.",
    },
    
    # Cohere Models
    "embed-english-v3.0": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": True,
        "supports_custom_dimensions": True,
        "provider": "cohere",
        "notes": "English-only with flexible dimensions.",
    },
    "embed-multilingual-v3.0": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": True,
        "supports_custom_dimensions": True,
        "provider": "cohere",
        "notes": "Multilingual with flexible dimensions.",
    },
    "embed-english-light-v3.0": {
        "dimensions": 384,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "cohere",
        "notes": "Lightweight English model.",
    },
    "embed-multilingual-light-v3.0": {
        "dimensions": 384,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "cohere",
        "notes": "Lightweight multilingual model.",
    },
    
    # Mixedbread AI Models
    "mxbai-embed-large": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "mixedbread",
        "notes": "Shortened model name.",
    },
    
    # Snowflake Arctic Models
    "snowflake/arctic-embed": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "snowflake",
        "notes": "Base Arctic embedding model.",
    },
    "arctic-embed-m": {
        "dimensions": 768,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "snowflake",
        "notes": "Medium-sized Arctic model.",
    },
    "arctic-embed-l": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "snowflake",
        "notes": "Large Arctic model.",
    },
    
    # BGE Models (BAAI General Embedding) - use query instruction prefix
    "bge-large-en-v1.5": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": False,
        "provider": "baai",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "notes": "Standard BGE model. Use query instruction prefix for best results.",
    },
    "bge-large-en-v1.5-mrl": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "baai",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "notes": "BGE with Matryoshka support.",
    },
    "bge-base-en-v1.5": {
        "dimensions": 768,
        "context_size": 512,
        "supports_matryoshka": False,
        "provider": "baai",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "notes": "Base BGE model.",
    },
    "bge-base-en-v1.5-mrl": {
        "dimensions": 768,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "baai",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "notes": "Base BGE with MRL.",
    },
    "bge-m3": {
        "dimensions": 1024,
        "context_size": 8192,
        "supports_matryoshka": True,
        "provider": "baai",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "notes": "Multilingual BGE with long context. Supports 100+ languages.",
    },
    "bge-small-en-v1.5": {
        "dimensions": 384,
        "context_size": 512,
        "supports_matryoshka": False,
        "provider": "baai",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "notes": "Small BGE model.",
    },
    "BAAI/bge-small-en-v1.5": {
        "dimensions": 384,
        "context_size": 512,
        "supports_matryoshka": False,
        "provider": "baai",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "notes": "Full HuggingFace path for small BGE.",
    },
    "BAAI/bge-base-en-v1.5": {
        "dimensions": 768,
        "context_size": 512,
        "supports_matryoshka": False,
        "provider": "baai",
        "notes": "Full HuggingFace path for base BGE.",
    },
    "BAAI/bge-large-en-v1.5": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": False,
        "provider": "baai",
        "notes": "Full HuggingFace path for large BGE.",
    },
    
    # Sentence Transformers
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimensions": 384,
        "context_size": 256,
        "supports_matryoshka": False,
        "provider": "sentence-transformers",
        "notes": "Popular lightweight model.",
    },
    "sentence-transformers/all-MiniLM-L6-v2-mrl": {
        "dimensions": 384,
        "context_size": 256,
        "supports_matryoshka": True,
        "provider": "sentence-transformers",
        "notes": "MRL variant of all-MiniLM.",
    },
    "all-minilm-l6-v2": {
        "dimensions": 384,
        "context_size": 256,
        "supports_matryoshka": False,
        "provider": "sentence-transformers",
        "notes": "Shortened name for all-MiniLM.",
    },
    "all-minilm-l6-v2-mrl": {
        "dimensions": 384,
        "context_size": 256,
        "supports_matryoshka": True,
        "provider": "sentence-transformers",
        "notes": "MRL variant shortened name.",
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "dimensions": 768,
        "context_size": 384,
        "supports_matryoshka": False,
        "provider": "sentence-transformers",
        "notes": "High-quality general purpose model.",
    },
    
    # Google Models
    "text-embedding-004": {
        "dimensions": 768,
        "context_size": 2048,
        "supports_matryoshka": False,
        "provider": "google",
        "notes": "Latest Google embedding model.",
    },
    "textembedding-gecko": {
        "dimensions": 768,
        "context_size": 2048,
        "supports_matryoshka": False,
        "provider": "google",
        "notes": "Legacy Google Vertex AI model.",
    },
    "embeddinggemma": {
        "dimensions": 768,
        "context_size": 2048,
        "supports_matryoshka": True,
        "supports_custom_dimensions": True,
        "provider": "google",
        "notes": "300M parameter multilingual embedding model. Supports 128-768 dimensions via MRL. Trained in 100+ languages. Optimized for on-device use.",
    },    
    # E5 Models (Microsoft) - require "query: " and "passage: " prefixes
    "intfloat/e5-large-v2": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": False,
        "provider": "microsoft",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "notes": "Large E5 model. Use 'query: ' prefix for queries and 'passage: ' for documents.",
    },
    "intfloat/e5-base-v2": {
        "dimensions": 768,
        "context_size": 512,
        "supports_matryoshka": False,
        "provider": "microsoft",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "notes": "Base E5 model. Use 'query: ' prefix for queries and 'passage: ' for documents.",
    },
    "intfloat/e5-small-v2": {
        "dimensions": 384,
        "context_size": 512,
        "supports_matryoshka": False,
        "provider": "microsoft",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "notes": "Small E5 model. Use 'query: ' prefix for queries and 'passage: ' for documents.",
    },
    
    # Multilingual E5 - require "query: " and "passage: " prefixes
    "intfloat/multilingual-e5-large": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "microsoft",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "notes": "Multilingual E5 large model. 100 languages. Use 'query: ' and 'passage: ' prefixes.",
    },
    "intfloat/multilingual-e5-base": {
        "dimensions": 768,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "microsoft",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "notes": "Multilingual E5 base model. 100 languages. Best quality/speed balance for multilingual.",
    },
    "intfloat/multilingual-e5-small": {
        "dimensions": 384,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "microsoft",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "notes": "Multilingual E5 small model. 100 languages. Good for resource-constrained environments.",
    },
    "multilingual-e5-base": {
        "dimensions": 768,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "microsoft",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "notes": "Multilingual E5 base (short name). 100 languages.",
    },
    "multilingual-e5-small": {
        "dimensions": 384,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "microsoft",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "notes": "Multilingual E5 small (short name). 100 languages.",
    },
    "multilingual-e5-large": {
        "dimensions": 1024,
        "context_size": 512,
        "supports_matryoshka": True,
        "provider": "microsoft",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "notes": "Multilingual E5 large (short name). 100 languages.",
    },
    
    # IBM Granite Models
    "granite-embedding": {
        "dimensions": 384,
        "context_size": 512,
        "supports_matryoshka": False,
        "provider": "ibm",
        "notes": "30M parameter dense bi-encoder. Optimized for enterprise use cases. Apache 2.0 license.",
    },
}


def get_model_spec(model_name: str) -> Optional[ModelSpec]:
    """
    Get model specification by name with fuzzy matching.
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        ModelSpec dictionary if found, None otherwise
    """
    if not model_name:
        return None
    
    # Direct match (case-insensitive)
    model_lower = model_name.lower().strip()
    for key, spec in EMBEDDING_MODELS_DATABASE.items():
        if key.lower() == model_lower:
            return spec
    
    # Fuzzy match: check if model name contains or is contained in database keys
    for key, spec in EMBEDDING_MODELS_DATABASE.items():
        key_lower = key.lower()
        if key_lower in model_lower or model_lower in key_lower:
            return spec
    
    return None


def get_model_dimension(model_name: str) -> Optional[int]:
    """
    Get the default dimension for a model.
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        Default dimension if found, None otherwise
    """
    spec = get_model_spec(model_name)
    return spec.get("dimensions") if spec else None


def get_model_context_size(model_name: str) -> Optional[int]:
    """
    Get the context/token limit for a model.
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        Context size if found, None otherwise
    """
    spec = get_model_spec(model_name)
    return spec.get("context_size") if spec else None


def supports_matryoshka(model_name: str) -> bool:
    """
    Check if a model supports Matryoshka Representation Learning.
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        True if model supports MRL, False otherwise
    """
    spec = get_model_spec(model_name)
    return spec.get("supports_matryoshka", False) if spec else False


def get_model_prefixes(model_name: str) -> Tuple[str, str]:
    """
    Get query and passage prefixes for a model.
    
    Some embedding models (like E5, BGE) perform better when text is prefixed
    with specific strings. This function returns the appropriate prefixes
    for the given model.
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        Tuple of (query_prefix, passage_prefix). Empty strings if no prefixes needed.
    """
    if not model_name:
        return ("", "")
    
    # First, try to get prefixes from the model database
    spec = get_model_spec(model_name)
    if spec:
        query_prefix = spec.get("query_prefix", "")
        passage_prefix = spec.get("passage_prefix", "")
        return (query_prefix, passage_prefix)
    
    # Fallback: try to match model family patterns
    model_lower = model_name.lower()
    
    for family_pattern, (query_prefix, passage_prefix) in MODEL_FAMILY_PREFIXES.items():
        if family_pattern.lower() in model_lower:
            return (query_prefix, passage_prefix)
    
    # No prefixes needed for this model
    return ("", "")


def get_query_prefix(model_name: str) -> str:
    """
    Get the query prefix for a model.
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        Query prefix string, or empty string if not needed.
    """
    query_prefix, _ = get_model_prefixes(model_name)
    return query_prefix


def get_passage_prefix(model_name: str) -> str:
    """
    Get the passage/document prefix for a model.
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        Passage prefix string, or empty string if not needed.
    """
    _, passage_prefix = get_model_prefixes(model_name)
    return passage_prefix


def model_requires_prefixes(model_name: str) -> bool:
    """
    Check if a model requires query/passage prefixes for optimal performance.
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        True if the model benefits from prefixes, False otherwise.
    """
    query_prefix, passage_prefix = get_model_prefixes(model_name)
    return bool(query_prefix or passage_prefix)


def apply_query_prefix(model_name: str, text: str) -> str:
    """
    Apply the appropriate query prefix to text for the given model.
    
    Args:
        model_name: Model name or identifier
        text: The query text to prefix
        
    Returns:
        Text with query prefix applied (if applicable).
    """
    prefix = get_query_prefix(model_name)
    if prefix and not text.startswith(prefix):
        return f"{prefix}{text}"
    return text


def apply_passage_prefix(model_name: str, text: str) -> str:
    """
    Apply the appropriate passage/document prefix to text for the given model.
    
    Args:
        model_name: Model name or identifier
        text: The passage/document text to prefix
        
    Returns:
        Text with passage prefix applied (if applicable).
    """
    prefix = get_passage_prefix(model_name)
    if prefix and not text.startswith(prefix):
        return f"{prefix}{text}"
    return text