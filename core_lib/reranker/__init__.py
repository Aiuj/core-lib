"""Reranker package: generic base and provider-specific implementations.

This package provides a stable public API for reranking search results with multiple providers.
It includes a provider-agnostic `BaseRerankerClient` and implementations for:
- Infinity (high-throughput local reranking server - default)
- Cohere (cloud-based reranking API)
- Local cross-encoder models (sentence-transformers)

Rerankers improve search quality by re-scoring query-document pairs using
cross-encoder models that consider the full context of both query and document.

Example usage:
    # Simple auto-detection from environment
    from core_lib.reranker import create_reranker_client
    client = create_reranker_client()
    
    # Rerank documents
    results = client.rerank(
        query="What is the capital of France?",
        documents=["Paris is the capital of France.", "London is in England."],
        top_k=5
    )
    
    # With specific provider and settings
    client = create_reranker_client(provider="infinity", model="BAAI/bge-reranker-v2-m3")
    
    # Using the factory class directly
    from core_lib.reranker import RerankerFactory
    client = RerankerFactory.create(provider="cohere")
    
    # Provider-specific creation
    from core_lib.reranker import create_infinity_reranker, create_cohere_reranker
    infinity_client = create_infinity_reranker(model="BAAI/bge-reranker-large")
    cohere_client = create_cohere_reranker(model="rerank-english-v3.0")
"""
from .base import BaseRerankerClient, RerankerError, RerankResult
from .reranker_config import RerankerSettings, reranker_settings

# Import providers with optional dependencies
try:
    from .infinity_provider import InfinityRerankerClient
    _infinity_available = True
except ImportError:
    InfinityRerankerClient = None
    _infinity_available = False

try:
    from .cohere_provider import CohereRerankerClient
    _cohere_available = True
except ImportError:
    CohereRerankerClient = None
    _cohere_available = False

try:
    from .local_provider import LocalRerankerClient
    _local_available = True
except ImportError:
    LocalRerankerClient = None
    _local_available = False

# Fallback client (always available as it doesn't require optional dependencies)
from .fallback_client import FallbackRerankerClient

from .factory import (
    RerankerFactory,
    create_reranker_client,
    create_client_from_env,
    create_infinity_reranker,
    create_cohere_reranker,
    create_local_reranker,
    create_fallback_reranker,
    create_reranker_from_env_with_fallback,
    get_reranker_client,
)

__all__ = [
    # Base classes and models
    "BaseRerankerClient",
    "RerankerError",
    "RerankResult",
    
    # Configuration
    "RerankerSettings",
    "reranker_settings",
    
    # Fallback/Multi-provider client
    "FallbackRerankerClient",
    
    # Factory and convenience functions
    "RerankerFactory",
    "create_reranker_client",
    "create_client_from_env",
    "create_infinity_reranker",
    "create_cohere_reranker",
    "create_local_reranker",
    "create_fallback_reranker",
    "create_reranker_from_env_with_fallback",
    "get_reranker_client",
]

# Add available providers to __all__
if _infinity_available:
    __all__.append("InfinityRerankerClient")
if _cohere_available:
    __all__.append("CohereRerankerClient")
if _local_available:
    __all__.append("LocalRerankerClient")
