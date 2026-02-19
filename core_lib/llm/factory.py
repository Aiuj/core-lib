"""Factory class for creating LLM clients with simplified configuration."""

import os
import warnings
from typing import Optional, Dict, Any, Union, Type, TYPE_CHECKING
from .llm_config import LLMConfig, GeminiConfig, OllamaConfig, OpenAIConfig
from .llm_client import LLMClient

if TYPE_CHECKING:
    from core_lib.config.llm_settings import LLMSettings


class LLMFactory:
    """Factory class for creating LLM clients with simplified configuration.
    
    This factory provides multiple ways to create LLM clients:
    1. From environment variables only (simplest)
    2. From environment with overrides
    3. From a configuration object
    4. From manual parameters
    """
    
    # Default provider preference order when no provider is specified
    DEFAULT_PROVIDER_ORDER = ["ollama", "openai", "gemini"]
    
    @classmethod
    def create(
        cls,
        provider: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        settings: Optional["LLMSettings"] = None,
        **kwargs
    ) -> LLMClient:
        """Create an LLM client using the most appropriate method.
        
        This is the main entry point that intelligently chooses how to create
        the client based on the provided parameters.
        
        Args:
            provider: Provider name ("gemini", "ollama", "openai", "azure-openai").
                     If None, will try to detect from config or environment.
            config: Pre-configured LLMConfig object. If provided, other params are ignored.
            **kwargs: Configuration overrides (model, temperature, max_tokens, etc.)
            
        Returns:
            Configured LLMClient instance
            
        Example:
            # Simple - uses environment variables
            client = LLMFactory.create()
            
            # With provider specification
            client = LLMFactory.create(provider="openai", model="gpt-4")
            
            # With custom config
            config = OpenAIConfig(api_key="sk-...", model="gpt-4")
            client = LLMFactory.create(config=config)
        """
        if config is not None:
            # Apply any overrides to the provided config
            if kwargs:
                config = cls._apply_overrides_to_config(config, kwargs)
            return LLMClient(config)

        if settings is not None:
            config = settings.to_llm_config()
            if kwargs:
                config = cls._apply_overrides_to_config(config, kwargs)
            return LLMClient(config)
        
        return cls.from_env(provider=provider, **kwargs)
    
    @classmethod
    def from_env(
        cls,
        provider: Optional[str] = None,
        **kwargs
    ) -> LLMClient:
        """Create an LLM client from environment variables with optional overrides.
        
        Args:
            provider: Provider name. If None, uses LLM_PROVIDER env var or auto-detects.
            **kwargs: Configuration overrides (model, temperature, max_tokens, etc.)
            
        Returns:
            Configured LLMClient instance
            
        Example:
            # Uses all env vars
            client = LLMFactory.from_env()
            
            # Override specific settings
            client = LLMFactory.from_env(model="gpt-4", temperature=0.2)
        """
        # Determine provider
        if provider is None:
            provider = cls._detect_provider_from_env()
        
        provider_lc = provider.lower()
        
        # Create base config from environment
        if provider_lc in ("gemini", "google"):
            config = GeminiConfig.from_env()
        elif provider_lc == "ollama":
            config = OllamaConfig.from_env()
        elif provider_lc in ("openai", "azure-openai"):
            config = OpenAIConfig.from_env()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Apply overrides
        if kwargs:
            config = cls._apply_overrides_to_config(config, kwargs)
        
        return LLMClient(config)
    
    @classmethod
    def from_config(
        cls,
        config: LLMConfig,
        **kwargs
    ) -> LLMClient:
        """Create an LLM client from a configuration object with optional overrides.
        
        Args:
            config: Pre-configured LLMConfig object
            **kwargs: Configuration overrides
            
        Returns:
            Configured LLMClient instance
            
        Example:
            config = OpenAIConfig(api_key="sk-...", model="gpt-3.5-turbo")
            client = LLMFactory.from_config(config, temperature=0.8)
        """
        if kwargs:
            config = cls._apply_overrides_to_config(config, kwargs)
        return LLMClient(config)
    
    @classmethod
    def gemini(
        cls,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        **kwargs
    ) -> LLMClient:
        """Create a Gemini LLM client.
        
        Args:
            api_key: Google API key (if None, will try to get from environment)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            thinking_enabled: Whether to enable step-by-step thinking
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured LLMClient instance
        """
        if api_key is None:
            config = GeminiConfig.from_env()
            # Override with provided parameters
            config.model = model
            config.temperature = temperature
            config.max_tokens = max_tokens
            config.thinking_enabled = thinking_enabled
        else:
            config = GeminiConfig(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_enabled=thinking_enabled,
                **kwargs
            )
        
        return LLMClient(config)
    
    @classmethod
    def ollama(
        cls,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        **kwargs
    ) -> LLMClient:
        """Create an Ollama LLM client.
        
        Args:
            model: Model name to use
            base_url: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            thinking_enabled: Whether to enable step-by-step thinking
            **kwargs: Additional configuration parameters (num_ctx, top_p, etc.)
            
        Returns:
            Configured LLMClient instance
        """
        config = OllamaConfig(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enabled=thinking_enabled,
            **kwargs
        )
        
        return LLMClient(config)
    
    @classmethod
    def openai(
        cls,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> LLMClient:
        """Create an OpenAI client (standard OpenAI endpoint).

        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            thinking_enabled: Whether to enable step-by-step thinking
            organization: OpenAI organization ID
            project: OpenAI project ID
            base_url: Custom base URL for OpenAI-compatible endpoints
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured LLMClient instance
        """
        if api_key is None:
            config = OpenAIConfig.from_env()
            config.model = model
            config.temperature = temperature
            config.max_tokens = max_tokens
            config.thinking_enabled = thinking_enabled
            if organization is not None:
                config.organization = organization
            if project is not None:
                config.project = project
            if base_url is not None:
                config.base_url = base_url
        else:
            config = OpenAIConfig(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_enabled=thinking_enabled,
                organization=organization,
                project=project,
                base_url=base_url,
                **kwargs,
            )
        return LLMClient(config)
    
    @classmethod
    def azure_openai(
        cls,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        deployment: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        **kwargs,
    ) -> LLMClient:
        """Create an Azure OpenAI client.

        Args:
            api_key: Azure OpenAI API key (if None, will try to get from environment)
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_version: Azure OpenAI API version
            deployment: Azure deployment name (maps to model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            thinking_enabled: Whether to enable step-by-step thinking
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured LLMClient instance
        """
        if api_key is None or azure_endpoint is None:
            config = OpenAIConfig.from_env()
            if deployment is not None:
                config.model = deployment
        else:
            config = OpenAIConfig(
                api_key=api_key,
                model=deployment or "gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_enabled=thinking_enabled,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version or "2024-08-01-preview",
                **kwargs,
            )
        # Apply common overrides
        config.temperature = temperature
        config.max_tokens = max_tokens
        config.thinking_enabled = thinking_enabled
        return LLMClient(config)
    
    @classmethod
    def openai_compatible(
        cls,
        base_url: str,
        api_key: str = "sk-local",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_enabled: bool = False,
        **kwargs,
    ) -> LLMClient:
        """Create a client for OpenAI-compatible endpoints (e.g., Ollama, LiteLLM).

        Args:
            base_url: Base URL for the OpenAI-compatible endpoint
            api_key: API key (many local gateways ignore this but require a non-empty string)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            thinking_enabled: Whether to enable step-by-step thinking
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured LLMClient instance
        """
        config = OpenAIConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enabled=thinking_enabled,
            **kwargs,
        )
        return LLMClient(config)
    
    @classmethod
    def _detect_provider_from_env(cls) -> str:
        """Auto-detect the provider from environment variables.
        
        Returns:
            Provider name based on available environment variables
            
        Note:
            AI_PROVIDER is deprecated. Use LLM_PROVIDERS_FILE for multi-provider
            configuration with fallback, or LLM_PROVIDER for single provider mode.
        """
        # Check explicit provider setting
        provider = os.environ.get("LLM_PROVIDER")
        
        # Deprecated: AI_PROVIDER (emit warning)
        if not provider:
            ai_provider = os.environ.get("AI_PROVIDER")
            if ai_provider:
                warnings.warn(
                    "AI_PROVIDER is deprecated. Use LLM_PROVIDERS_FILE for multi-provider "
                    "configuration with automatic fallback, or LLM_PROVIDER for single "
                    "provider mode. See core_lib.llm.FallbackLLMClient for details.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                provider = ai_provider
        
        if provider:
            return provider.lower()
        
        # Try to detect based on available API keys
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY"):
            return "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            return "openai"
        elif os.environ.get("AZURE_OPENAI_API_KEY"):
            return "azure-openai"
        elif os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST"):
            return "ollama"
        
        # Default fallback (Ollama typically doesn't need API keys)
        return "ollama"
    
    @classmethod
    def _apply_overrides_to_config(cls, config: LLMConfig, overrides: Dict[str, Any]) -> LLMConfig:
        """Apply configuration overrides to a config object.
        
        Args:
            config: Base configuration object
            overrides: Dictionary of configuration overrides
            
        Returns:
            Updated configuration object
        """
        # Apply common overrides when provided
        for key, value in overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        
        return config


# Convenience functions for backward compatibility and simple usage
def create_llm_client(
    provider: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    settings: Optional["LLMSettings"] = None,
    **kwargs
) -> LLMClient:
    """Create an LLM client using the factory.
    
    This is the main convenience function that should be used for most cases.
    
    Args:
        provider: Provider name ("gemini", "ollama", "openai", "azure-openai")
        config: Pre-configured LLMConfig object
        **kwargs: Configuration overrides
        
    Returns:
        Configured LLMClient instance
        
    Example:
        # Simplest usage - auto-detect from environment
        client = create_llm_client()
        
        # With provider and overrides
        client = create_llm_client(provider="openai", model="gpt-4", temperature=0.2)
        
        # With custom config
        config = GeminiConfig(api_key="...", model="gemini-pro")
        client = create_llm_client(config=config)
    """
    return LLMFactory.create(provider=provider, config=config, settings=settings, **kwargs)


def create_client_from_env(
    provider: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """Create an LLM client from environment variables with optional overrides.
    
    This function provides backward compatibility with the existing utils.py interface.
    
    Args:
        provider: Provider name. If None, auto-detects from environment.
        **kwargs: Configuration overrides
        
    Returns:
        Configured LLMClient instance
    """
    return LLMFactory.from_env(provider=provider, **kwargs)


# Provider-specific convenience functions for backward compatibility
def create_gemini_client(**kwargs) -> LLMClient:
    """Create a Gemini LLM client."""
    return LLMFactory.gemini(**kwargs)


def create_ollama_client(**kwargs) -> LLMClient:
    """Create an Ollama LLM client."""
    return LLMFactory.ollama(**kwargs)


def create_openai_client(**kwargs) -> LLMClient:
    """Create an OpenAI LLM client."""
    return LLMFactory.openai(**kwargs)


def create_azure_openai_client(**kwargs) -> LLMClient:
    """Create an Azure OpenAI LLM client."""
    return LLMFactory.azure_openai(**kwargs)


def create_openai_compatible_client(**kwargs) -> LLMClient:
    """Create an OpenAI-compatible LLM client."""
    return LLMFactory.openai_compatible(**kwargs)