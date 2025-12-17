"""Embeddings package: generic base and provider-specific implementations.

This package provides a stable public API for generating embeddings with multiple providers.
It includes a provider-agnostic `BaseEmbeddingClient` and implementations for:
- OpenAI (text-embedding-3-small, text-embedding-3-large, etc.)
- Google GenAI (text-embedding-004 with task types)
- Ollama (local models)
- Local HuggingFace models (sentence-transformers, transformers)
- Infinity (high-throughput local embedding server)

Example usage:
    # Simple auto-detection from environment
    from core_lib.embeddings import create_embedding_client
    client = create_embedding_client()
    
    # With specific provider and settings
    client = create_embedding_client(provider="openai", model="text-embedding-3-small")
    
    # Using the factory class directly
    from core_lib.embeddings import EmbeddingFactory
    client = EmbeddingFactory.create(provider="google_genai", task_type="SEMANTIC_SIMILARITY")
    
    # Provider-specific creation
    from core_lib.embeddings import create_openai_client, create_google_genai_client, create_infinity_client
    openai_client = create_openai_client(model="text-embedding-3-large")
    google_client = create_google_genai_client(task_type="CLASSIFICATION")
    infinity_client = create_infinity_client(model="BAAI/bge-small-en-v1.5")
"""
from .base import BaseEmbeddingClient, EmbeddingGenerationError
from .ollama import OllamaEmbeddingClient
from .embeddings_config import EmbeddingsConfig, embeddings_settings, TaskType
from .fallback_client import FallbackEmbeddingClient
from .embedding_utils import (
    normalize_embedding_dimension,
    normalize_embeddings_batch,
    is_matryoshka_model,
    get_best_normalization_method,
)
from .models_database import (
    get_model_spec,
    get_model_dimension,
    get_model_context_size,
    supports_matryoshka,
    get_model_prefixes,
    get_query_prefix,
    get_passage_prefix,
    model_requires_prefixes,
    apply_query_prefix,
    apply_passage_prefix,
)
from .factory import (
    EmbeddingFactory,
    create_embedding_client,
    create_client_from_env,
    create_openai_client,
    create_google_genai_client,
    create_ollama_client,
    create_local_client,
    create_infinity_client,
    get_embedding_client,  # Legacy function
)
from .models import EmbeddingResponse

# Conditionally import providers based on availability
try:
    from .openai_provider import OpenAIEmbeddingClient
    __all_providers__ = ["OpenAIEmbeddingClient"]
except ImportError:
    __all_providers__ = []

try:
    from .google_genai_provider import GoogleGenAIEmbeddingClient
    __all_providers__.append("GoogleGenAIEmbeddingClient")
except ImportError:
    pass

try:
    from .local_provider import LocalEmbeddingClient
    __all_providers__.append("LocalEmbeddingClient")
except ImportError:
    pass

try:
    from .infinity_provider import InfinityEmbeddingClient
    __all_providers__.append("InfinityEmbeddingClient")
except ImportError:
    pass

__all__ = [
    # Base classes and errors
    "BaseEmbeddingClient",
    "EmbeddingGenerationError",
    
    # Always available providers
    "OllamaEmbeddingClient",
    
    # Fallback client for high availability
    "FallbackEmbeddingClient",
    
    # Configuration
    "EmbeddingsConfig",
    "embeddings_settings",
    "TaskType",
    
    # Embedding utilities
    "normalize_embedding_dimension",
    "normalize_embeddings_batch",
    "is_matryoshka_model",
    "get_best_normalization_method",
    
    # Model database
    "get_model_spec",
    "get_model_dimension",
    "get_model_context_size",
    "supports_matryoshka",
    "get_model_prefixes",
    "get_query_prefix",
    "get_passage_prefix",
    "model_requires_prefixes",
    "apply_query_prefix",
    "apply_passage_prefix",
    
    # Factory and convenience functions
    "EmbeddingFactory",
    "create_embedding_client",
    "create_client_from_env",
    "create_openai_client",
    "create_google_genai_client",
    "create_ollama_client",
    "create_local_client",
    "create_infinity_client",
    "get_embedding_client",
] + __all_providers__
