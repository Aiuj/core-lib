"""Factory helpers to create reranker client instances based on configuration."""
from typing import Optional, List, Dict, Any

from .reranker_config import reranker_settings
from .base import BaseRerankerClient
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()

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


class RerankerFactory:
    """Factory class for creating reranker clients with various providers."""

    @classmethod
    def create(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseRerankerClient:
        """Create a reranker client with the specified provider.
        
        Args:
            provider: Provider name ('infinity', 'cohere', 'local')
            model: Model name
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Configured reranker client instance
        """
        if provider is None:
            provider = reranker_settings.provider.lower()
        else:
            provider = provider.lower()

        if provider == "infinity":
            return cls.infinity(model=model, **kwargs)
        elif provider == "cohere":
            return cls.cohere(model=model, **kwargs)
        elif provider == "local" or provider == "crossencoder":
            return cls.local(model=model, **kwargs)
        else:
            raise ValueError(f"Unknown reranker provider: {provider}")

    @classmethod
    def infinity(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        **kwargs
    ) -> BaseRerankerClient:
        """Create an Infinity reranker client.
        
        Args:
            model: Model name (e.g., 'BAAI/bge-reranker-v2-m3')
            base_url: Base URL of Infinity server
            timeout: Request timeout in seconds
            token: Authentication token
            **kwargs: Additional parameters
            
        Returns:
            Infinity reranker client instance
        """
        if not _infinity_available or InfinityRerankerClient is None:
            raise ImportError(
                "Infinity provider not available. Install with: pip install requests"
            )

        return InfinityRerankerClient(
            model=model,
            base_url=base_url,
            timeout=timeout,
            token=token,
            **kwargs
        )

    @classmethod
    def cohere(
        cls,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseRerankerClient:
        """Create a Cohere reranker client.
        
        Args:
            model: Model name (e.g., 'rerank-multilingual-v3.0')
            api_key: Cohere API key
            **kwargs: Additional parameters
            
        Returns:
            Cohere reranker client instance
        """
        if not _cohere_available or CohereRerankerClient is None:
            raise ImportError(
                "Cohere provider not available. Install with: pip install cohere"
            )

        return CohereRerankerClient(
            model=model,
            api_key=api_key,
            **kwargs
        )

    @classmethod
    def local(
        cls,
        model: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseRerankerClient:
        """Create a local reranker client.
        
        Args:
            model: Model name from HuggingFace
            device: Device to run on ('cpu', 'cuda', 'auto')
            **kwargs: Additional parameters
            
        Returns:
            Local reranker client instance
        """
        if not _local_available or LocalRerankerClient is None:
            raise ImportError(
                "Local provider not available. Install with: pip install sentence-transformers"
            )

        return LocalRerankerClient(
            model=model,
            device=device,
            **kwargs
        )

    @classmethod
    def from_config(cls, config: Optional[object] = None) -> BaseRerankerClient:
        """Create a client from configuration object.
        
        Args:
            config: Configuration object (defaults to reranker_settings)
            
        Returns:
            Configured reranker client instance
        """
        if config is None:
            config = reranker_settings

        provider_kwargs = {}

        # Add provider-specific kwargs based on config
        if hasattr(config, 'api_key') and config.api_key:
            provider_kwargs['api_key'] = config.api_key
        if hasattr(config, 'infinity_url') and config.infinity_url:
            provider_kwargs['base_url'] = config.infinity_url
        if hasattr(config, 'infinity_timeout') and config.infinity_timeout:
            provider_kwargs['timeout'] = config.infinity_timeout
        if hasattr(config, 'infinity_token') and config.infinity_token:
            provider_kwargs['token'] = config.infinity_token
        if hasattr(config, 'device') and config.device:
            provider_kwargs['device'] = config.device
        if hasattr(config, 'cache_dir') and config.cache_dir:
            provider_kwargs['cache_dir'] = config.cache_dir
        if hasattr(config, 'trust_remote_code'):
            provider_kwargs['trust_remote_code'] = config.trust_remote_code
        if hasattr(config, 'cache_duration_seconds'):
            provider_kwargs['cache_duration_seconds'] = config.cache_duration_seconds

        return cls.create(
            provider=config.provider,
            model=config.model,
            **provider_kwargs
        )


# Convenience functions
def create_reranker_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> BaseRerankerClient:
    """Create a reranker client with auto-detection or specified provider.
    
    Args:
        provider: Provider name (if None, auto-detect from environment)
        model: Model name
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Configured reranker client instance
    """
    return RerankerFactory.create(
        provider=provider,
        model=model,
        **kwargs
    )


def create_client_from_env() -> BaseRerankerClient:
    """Create a reranker client from environment configuration.
    
    Returns:
        Configured reranker client instance based on environment variables
    """
    return RerankerFactory.from_config()


def create_infinity_reranker(
    model: str = "BAAI/bge-reranker-v2-m3",
    base_url: Optional[str] = None,
    **kwargs
) -> BaseRerankerClient:
    """Create an Infinity reranker client.
    
    Args:
        model: Model name
        base_url: Base URL of Infinity server
        **kwargs: Additional parameters
        
    Returns:
        Infinity reranker client instance
    """
    return RerankerFactory.infinity(model=model, base_url=base_url, **kwargs)


def create_cohere_reranker(
    model: str = "rerank-multilingual-v3.0",
    api_key: Optional[str] = None,
    **kwargs
) -> BaseRerankerClient:
    """Create a Cohere reranker client.
    
    Args:
        model: Model name
        api_key: Cohere API key
        **kwargs: Additional parameters
        
    Returns:
        Cohere reranker client instance
    """
    return RerankerFactory.cohere(model=model, api_key=api_key, **kwargs)


def create_local_reranker(
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: Optional[str] = None,
    **kwargs
) -> BaseRerankerClient:
    """Create a local reranker client.
    
    Args:
        model: Model name from HuggingFace
        device: Device to run on
        **kwargs: Additional parameters
        
    Returns:
        Local reranker client instance
    """
    return RerankerFactory.local(model=model, device=device, **kwargs)


# Legacy function for backward compatibility
def get_reranker_client() -> BaseRerankerClient:
    """Convenience wrapper to get a reranker client (auto-detect from environment).

    This will attempt to create a client automatically based on configuration.
    """
    return create_client_from_env()


def create_fallback_reranker(
    configs: List[Dict[str, Any]],
    **fallback_kwargs
) -> 'FallbackRerankerClient':
    """Create a fallback reranker client from provider configuration list.
    
    Args:
        configs: List of provider configurations, each a dict with:
            - provider: Provider name ('infinity', 'cohere', 'local')
            - Additional provider-specific params (base_url, api_key, etc.)
        **fallback_kwargs: Arguments for FallbackRerankerClient constructor
        
    Returns:
        Configured fallback reranker client
        
    Example:
        ```python
        from core_lib.reranker import create_fallback_reranker
        
        client = create_fallback_reranker([
            {"provider": "infinity", "base_url": "http://server1:7997"},
            {"provider": "infinity", "base_url": "http://server2:7997"},
            {"provider": "cohere", "api_key": "..."},
        ], max_retries_per_provider=2)
        ```
    """
    from .fallback_client import FallbackRerankerClient
    return FallbackRerankerClient.from_config(configs, **fallback_kwargs)


def create_reranker_from_env_with_fallback() -> BaseRerankerClient:
    """Create reranker client with automatic multi-URL failover from environment.
    
    Checks if INFINITY_BASE_URL contains comma-separated URLs and creates
    a fallback client automatically if multiple URLs are detected.
    
    Returns:
        Single provider client or fallback client based on environment config
        
    Example:
        ```bash
        # Single URL - creates InfinityRerankerClient
        INFINITY_BASE_URL=http://localhost:7997
        
        # Multiple URLs - creates FallbackRerankerClient
        INFINITY_BASE_URL=http://server1:7997,http://server2:7997,http://server3:7997
        ```
    """
    from .fallback_client import FallbackRerankerClient
    import os
    
    # Check if we have comma-separated URLs for Infinity
    infinity_url = os.getenv('INFINITY_BASE_URL') or reranker_settings.infinity_url
    
    if infinity_url and ',' in infinity_url:
        # Multiple URLs detected - create fallback client
        logger.info(f"Creating fallback reranker with multi-URL support: {infinity_url}")
        
        # Create a provider for each URL
        configs = [
            {
                "provider": "infinity",
                "base_url": url.strip(),
                "model": reranker_settings.model,
                "timeout": reranker_settings.infinity_timeout,
                "token": reranker_settings.infinity_token,
            }
            for url in infinity_url.split(',')
        ]
        
        return FallbackRerankerClient.from_config(
            configs,
            cache_duration_seconds=reranker_settings.cache_duration_seconds,
        )
    
    # Single URL or other provider - use standard creation
    return create_client_from_env()
