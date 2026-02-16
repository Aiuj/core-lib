"""Factory helpers to create embedding client instances based on configuration."""
from typing import Optional

from .embeddings_config import embeddings_settings
from .base import BaseEmbeddingClient
from .ollama import OllamaEmbeddingClient
from ..config.embeddings_settings import EmbeddingsSettings
from ..config.provider_chain_utils import (
    build_kwargs_from_config,
    build_provider_chain,
    create_client_from_runtime_chain,
    get_multi_url_value,
    load_runtime_settings_if_needed,
)
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()

# Import new providers with optional dependencies
try:
    from .openai_provider import OpenAIEmbeddingClient
    _openai_available = True
except ImportError:
    OpenAIEmbeddingClient = None
    _openai_available = False

try:
    from .google_genai_provider import GoogleGenAIEmbeddingClient
    _google_genai_available = True
except ImportError:
    GoogleGenAIEmbeddingClient = None
    _google_genai_available = False

try:
    from .local_provider import LocalEmbeddingClient
    _local_available = True
except ImportError:
    LocalEmbeddingClient = None
    _local_available = False

try:
    from .infinity_provider import InfinityEmbeddingClient
    _infinity_available = True
except ImportError:
    InfinityEmbeddingClient = None
    _infinity_available = False


class EmbeddingFactory:
    """Factory class for creating embedding clients with various providers."""

    @classmethod
    def create(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        norm_method: Optional[str] = None,
        **kwargs
    ) -> BaseEmbeddingClient:
        """Create an embedding client with the specified provider.
        
        Args:
            provider: Provider name ('openai', 'google_genai', 'ollama', 'local', 'infinity')
            model: Model name
            embedding_dim: Target embedding dimension
            use_l2_norm: Whether to apply L2 normalization
            norm_method: Normalization method ('truncate_or_pad', 'interpolate', 'pca_approximate', or None for auto)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Configured embedding client instance
        """
        if provider is None:
            provider = embeddings_settings.provider.lower()
        else:
            provider = provider.lower()

        if provider == "openai":
            return cls.openai(
                model=model,
                embedding_dim=embedding_dim,
                use_l2_norm=use_l2_norm,
                norm_method=norm_method,
                **kwargs
            )
        elif provider == "google_genai" or provider == "google" or provider == "gemini":
            return cls.google_genai(
                model=model,
                embedding_dim=embedding_dim,
                use_l2_norm=use_l2_norm,
                norm_method=norm_method,
                **kwargs
            )
        elif provider == "ollama":
            return cls.ollama(
                model=model,
                embedding_dim=embedding_dim,
                use_l2_norm=use_l2_norm,
                norm_method=norm_method,
                **kwargs
            )
        elif provider == "local" or provider == "huggingface":
            return cls.local(
                model=model,
                embedding_dim=embedding_dim,
                use_l2_norm=use_l2_norm,
                norm_method=norm_method,
                **kwargs
            )
        elif provider == "infinity":
            return cls.infinity(
                model=model,
                embedding_dim=embedding_dim,
                use_l2_norm=use_l2_norm,
                norm_method=norm_method,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def openai(
        cls,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        norm_method: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        **kwargs
    ) -> BaseEmbeddingClient:
        """Create an OpenAI embedding client.
        
        Args:
            model: Model name (e.g., 'text-embedding-3-small')
            embedding_dim: Target embedding dimension
            use_l2_norm: Whether to apply L2 normalization
            norm_method: Normalization method (None for auto-detection)
            api_key: OpenAI API key
            base_url: Custom base URL
            organization: OpenAI organization ID
            project: OpenAI project ID
            **kwargs: Additional parameters
            
        Returns:
            OpenAI embedding client instance
        """
        if not _openai_available or OpenAIEmbeddingClient is None:
            raise ImportError(
                "OpenAI provider not available. Install with: pip install openai"
            )

        return OpenAIEmbeddingClient(
            model=model,
            embedding_dim=embedding_dim,
            use_l2_norm=use_l2_norm,
            norm_method=norm_method,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            **kwargs
        )

    @classmethod
    def google_genai(
        cls,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        norm_method: Optional[str] = None,
        api_key: Optional[str] = None,
        task_type: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> BaseEmbeddingClient:
        """Create a Google GenAI embedding client.
        
        Args:
            model: Model name (e.g., 'text-embedding-004')
            embedding_dim: Target embedding dimension
            use_l2_norm: Whether to apply L2 normalization
            norm_method: Normalization method (None for auto-detection)
            api_key: Google API key
            task_type: Task type for embeddings
            title: Optional title for the embedding task
            **kwargs: Additional parameters
            
        Returns:
            Google GenAI embedding client instance
        """
        if not _google_genai_available or GoogleGenAIEmbeddingClient is None:
            raise ImportError(
                "Google GenAI provider not available. Install with: pip install google-genai"
            )

        return GoogleGenAIEmbeddingClient(
            model=model,
            embedding_dim=embedding_dim,
            use_l2_norm=use_l2_norm,
            api_key=api_key,
            task_type=task_type,
            title=title,
            **kwargs
        )

    @classmethod
    def ollama(
        cls,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        norm_method: Optional[str] = None,
        **kwargs
    ) -> BaseEmbeddingClient:
        """Create an Ollama embedding client.
        
        Args:
            model: Model name
            embedding_dim: Target embedding dimension
            use_l2_norm: Whether to apply L2 normalization
            norm_method: Normalization method (None for auto-detection)
            **kwargs: Additional parameters
            
        Returns:
            Ollama embedding client instance
        """
        return OllamaEmbeddingClient(
            model=model,
            embedding_dim=embedding_dim,
            use_l2_norm=use_l2_norm,
            norm_method=norm_method,
            **kwargs
        )

    @classmethod
    def local(
        cls,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        norm_method: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        use_sentence_transformers: bool = True,
        **kwargs
    ) -> BaseEmbeddingClient:
        """Create a local embedding client.
        
        Args:
            model: Model name from HuggingFace
            embedding_dim: Target embedding dimension
            use_l2_norm: Whether to apply L2 normalization
            norm_method: Normalization method (None for auto-detection)
            device: Device to run the model on
            cache_dir: Directory to cache downloaded models
            trust_remote_code: Whether to trust remote code
            use_sentence_transformers: Whether to use sentence-transformers library
            **kwargs: Additional parameters
            
        Returns:
            Local embedding client instance
        """
        if not _local_available or LocalEmbeddingClient is None:
            raise ImportError(
                "Local provider not available. Install with: pip install sentence-transformers "
                "or pip install transformers torch"
            )

        return LocalEmbeddingClient(
            model=model,
            embedding_dim=embedding_dim,
            use_l2_norm=use_l2_norm,
            norm_method=norm_method,
            device=device,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            use_sentence_transformers=use_sentence_transformers,
            **kwargs
        )

    @classmethod
    def infinity(
        cls,
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        norm_method: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> BaseEmbeddingClient:
        """Create an Infinity embedding client.
        
        Args:
            model: Model name (e.g., 'BAAI/bge-small-en-v1.5')
            embedding_dim: Target embedding dimension
            use_l2_norm: Whether to apply L2 normalization
            norm_method: Normalization method (None for auto-detection)
            base_url: Base URL of Infinity server (default: http://localhost:7997)
            timeout: Request timeout in seconds
            **kwargs: Additional parameters
            
        Returns:
            Infinity embedding client instance
        """
        if not _infinity_available or InfinityEmbeddingClient is None:
            raise ImportError(
                "Infinity provider not available. Install with: pip install requests"
            )

        return InfinityEmbeddingClient(
            model=model,
            embedding_dim=embedding_dim,
            use_l2_norm=use_l2_norm,
            norm_method=norm_method,
            base_url=base_url,
            timeout=timeout,
            **kwargs
        )

    @classmethod
    def from_config(cls, config: Optional[object] = None) -> BaseEmbeddingClient:
        """Create a client from configuration object.
        
        Args:
            config: Configuration object (defaults to embeddings_settings)
            
        Returns:
            Configured embedding client instance
        """
        if config is None:
            config = embeddings_settings

        provider_kwargs = build_kwargs_from_config(
            config,
            field_specs=[
                ("api_key", "api_key", "truthy"),
                ("base_url", "base_url", "truthy"),
                ("organization", "organization", "truthy"),
                ("project", "project", "truthy"),
                ("task_type", "task_type", "truthy"),
                ("title", "title", "truthy"),
                ("device", "device", "truthy"),
                ("cache_dir", "cache_dir", "truthy"),
                ("trust_remote_code", "trust_remote_code", "exists"),
                ("use_sentence_transformers", "use_sentence_transformers", "exists"),
            ],
        )

        # Preserve existing precedence: google_api_key overrides api_key when set.
        if hasattr(config, "google_api_key") and getattr(config, "google_api_key"):
            provider_kwargs["api_key"] = getattr(config, "google_api_key")

        return cls.create(
            provider=config.provider,
            model=config.model,
            embedding_dim=config.embedding_dimension,
            **provider_kwargs
        )


# Convenience functions following the LLM module pattern
def create_embedding_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    use_l2_norm: bool = False,
    norm_method: Optional[str] = None,
    intelligence_level: Optional[int] = None,
    usage: Optional[str] = None,
    **kwargs
) -> BaseEmbeddingClient:
    """Create an embedding client with auto-detection or specified provider.
    
    Automatically uses FallbackEmbeddingClient when comma-separated URLs are detected
    in environment variables for high availability.
    
    Args:
        provider: Provider name (if None, auto-detect from environment)
        model: Model name
        embedding_dim: Target embedding dimension
        use_l2_norm: Whether to apply L2 normalization
        norm_method: Normalization method (None for auto-detection)
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Configured embedding client instance (FallbackEmbeddingClient if multiple URLs detected)
    """
    # Use fresh settings snapshot when provider is not explicitly forced
    runtime_settings = load_runtime_settings_if_needed(
        provider,
        loader=lambda: EmbeddingsSettings.from_env(
            load_dotenv=False,
            intelligence_level=intelligence_level,
            usage=usage,
        ),
    )

    from .fallback_client import FallbackEmbeddingClient

    client_from_chain = create_client_from_runtime_chain(
        provider=provider,
        runtime_settings=runtime_settings,
        model=model,
        chain_overrides={"embedding_dim": embedding_dim},
        create_fallback=lambda provider_configs: FallbackEmbeddingClient.from_config(
            provider_configs=provider_configs,
            common_model=model,
            common_embedding_dim=embedding_dim,
            common_use_l2_norm=use_l2_norm,
            common_norm_method=norm_method,
        ),
        create_single=lambda provider_name, model_name, single_cfg: EmbeddingFactory.create(
            provider=provider_name,
            model=model_name,
            embedding_dim=single_cfg.pop("embedding_dim", embedding_dim),
            use_l2_norm=use_l2_norm,
            norm_method=norm_method,
            **single_cfg,
            **kwargs,
        ),
    )
    if client_from_chain is not None:
        return client_from_chain

    # Detect if we should use fallback client (comma-separated URLs)
    provider_name = provider or (runtime_settings.provider.lower() if runtime_settings else embeddings_settings.provider.lower())
    url_to_check = get_multi_url_value(
        provider_name=provider_name,
        provider_env_map={
            "infinity": ["INFINITY_BASE_URL"],
            "ollama": ["OLLAMA_URL"],
            "openai": ["OPENAI_BASE_URL"],
        },
        default_env_vars=["EMBEDDING_BASE_URL"],
    )
    
    # If comma-separated URLs detected, use FallbackEmbeddingClient
    if url_to_check:
        from .fallback_client import FallbackEmbeddingClient
        logger.debug(f"Detected comma-separated URLs for provider '{provider_name}', using FallbackEmbeddingClient for HA")
        return FallbackEmbeddingClient.from_env(provider=provider_name)
    
    # Otherwise use single provider
    return EmbeddingFactory.create(
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        use_l2_norm=use_l2_norm,
        norm_method=norm_method,
        **kwargs
    )


def create_client_from_env() -> BaseEmbeddingClient:
    """Create an embedding client from environment configuration.
    
    Returns:
        Configured embedding client instance based on environment variables
    """
    runtime_settings = EmbeddingsSettings.from_env(load_dotenv=False)
    if runtime_settings.provider_configs and len(runtime_settings.provider_configs) > 1:
        from .fallback_client import FallbackEmbeddingClient
        provider_configs = build_provider_chain(runtime_settings.provider_configs)
        return FallbackEmbeddingClient.from_config(provider_configs=provider_configs)
    return EmbeddingFactory.from_config(config=runtime_settings)


def create_openai_client(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    **kwargs
) -> BaseEmbeddingClient:
    """Create an OpenAI embedding client.
    
    Args:
        model: Model name
        api_key: OpenAI API key
        **kwargs: Additional parameters
        
    Returns:
        OpenAI embedding client instance
    """
    return EmbeddingFactory.openai(model=model, api_key=api_key, **kwargs)


def create_google_genai_client(
    model: str = "text-embedding-004",
    api_key: Optional[str] = None,
    task_type: Optional[str] = None,
    **kwargs
) -> BaseEmbeddingClient:
    """Create a Google GenAI embedding client.
    
    Args:
        model: Model name
        api_key: Google API key
        task_type: Task type for embeddings
        **kwargs: Additional parameters
        
    Returns:
        Google GenAI embedding client instance
    """
    return EmbeddingFactory.google_genai(
        model=model, api_key=api_key, task_type=task_type, **kwargs
    )


def create_ollama_client(
    model: str = "nomic-embed-text",
    **kwargs
) -> BaseEmbeddingClient:
    """Create an Ollama embedding client.
    
    Args:
        model: Model name
        **kwargs: Additional parameters
        
    Returns:
        Ollama embedding client instance
    """
    return EmbeddingFactory.ollama(model=model, **kwargs)


def create_local_client(
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    **kwargs
) -> BaseEmbeddingClient:
    """Create a local embedding client.
    
    Args:
        model: Model name from HuggingFace
        device: Device to run the model on
        **kwargs: Additional parameters
        
    Returns:
        Local embedding client instance
    """
    return EmbeddingFactory.local(model=model, device=device, **kwargs)


def create_infinity_client(
    model: str = "BAAI/bge-small-en-v1.5",
    base_url: Optional[str] = None,
    **kwargs
) -> BaseEmbeddingClient:
    """Create an Infinity embedding client.
    
    Args:
        model: Model name
        base_url: Base URL of Infinity server
        **kwargs: Additional parameters
        
    Returns:
        Infinity embedding client instance
    """
    return EmbeddingFactory.infinity(model=model, base_url=base_url, **kwargs)


# Legacy function for backward compatibility - now with enhanced capabilities
def get_embedding_client() -> BaseEmbeddingClient:
    """Convenience wrapper to get an embedding client (auto-detect from environment).

    This will attempt to create a client automatically based on configuration.
    """
    return create_client_from_env()
