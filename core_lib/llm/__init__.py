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
from .provider_registry import ProviderConfig, ProviderRegistry
from .json_parser import clean_and_parse_json_response

__all__ = [
    # Core classes
    "LLMConfig", 
    "GeminiConfig",
    "OllamaConfig",
    "OpenAIConfig",
    "LLMClient",
    
    # Provider registry (multi-provider configuration)
    "ProviderConfig",
    "ProviderRegistry",
    
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
    
    # Utilities
    "clean_and_parse_json_response"
]
