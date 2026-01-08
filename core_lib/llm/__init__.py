"""LLM module for core-lib.

This module provides abstractions for working with different LLM providers
including Google Gemini, OpenAI, Azure OpenAI, and local Ollama APIs.

The module now features a simplified factory-based approach for creating LLM clients:

Example:
    # Simplest usage - auto-detect from environment
    from core_lib.llm import create_llm_client
    client = create_llm_client()
    
    # With specific provider and settings
    client = create_llm_client(provider="openai", model="gpt-4", temperature=0.2)
    
    # Using the factory class directly
    from core_lib.llm import LLMFactory
    client = LLMFactory.create(provider="gemini", model="gemini-pro")
"""

from .llm_config import LLMConfig, GeminiConfig, OllamaConfig, OpenAIConfig
from .llm_client import LLMClient
from .factory import (
    LLMFactory,
    create_llm_client,
    create_client_from_env,
    create_gemini_client,
    create_ollama_client,
    create_openai_client,
    create_azure_openai_client,
    create_openai_compatible_client,
)
from .provider_registry import ProviderConfig, ProviderRegistry, substitute_env_vars
from .provider_health import (
    ProviderHealthTracker,
    HealthStatus,
    get_health_tracker,
    reset_health_tracker,
    classify_error,
    DEFAULT_UNHEALTHY_TTL,
    FAILURE_TTL_MAP,
)
from .fallback_client import (
    FallbackLLMClient,
    FallbackResult,
    create_fallback_llm_client,
)
from .json_parser import clean_and_parse_json_response

# LangChain adapter - optional import (requires langchain-core)
try:
    from .langchain_adapter import (
        CoreLibChatModel,
        create_langchain_model,
        create_langchain_model_with_fallback,
    )
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    CoreLibChatModel = None  # type: ignore
    create_langchain_model = None  # type: ignore
    create_langchain_model_with_fallback = None  # type: ignore
    _LANGCHAIN_AVAILABLE = False

__all__ = [
    # Core classes
    "LLMConfig", 
    "GeminiConfig",
    "OllamaConfig",
    "OpenAIConfig",
    "LLMClient",
    
    # Fallback client (automatic provider failover)
    "FallbackLLMClient",
    "FallbackResult",
    "create_fallback_llm_client",
    
    # Provider registry (multi-provider configuration)
    "ProviderConfig",
    "ProviderRegistry",
    "substitute_env_vars",
    
    # Provider health tracking (intelligent fallback)
    "ProviderHealthTracker",
    "HealthStatus",
    "get_health_tracker",
    "reset_health_tracker",
    "classify_error",
    "DEFAULT_UNHEALTHY_TTL",
    "FAILURE_TTL_MAP",
    
    # Factory class
    "LLMFactory",
    
    # Main convenience functions
    "create_llm_client",
    "create_client_from_env",
    
    # Provider-specific functions (backward compatibility)
    "create_gemini_client",
    "create_ollama_client",
    "create_openai_client",
    "create_azure_openai_client",
    "create_openai_compatible_client",
    
    # LangChain adapter (optional - requires langchain-core)
    "CoreLibChatModel",
    "create_langchain_model",
    "create_langchain_model_with_fallback",
    
    # Utilities
    "clean_and_parse_json_response"
]
