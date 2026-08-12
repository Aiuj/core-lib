"""Factory helpers to create reranker client instances based on configuration."""
from typing import Optional, List, Dict, Any

from .reranker_config import reranker_settings
from .base import BaseRerankerClient
from .reranker_config import RerankerSettings
from ..config.provider_chain_utils import (
    build_kwargs_from_config,
    build_provider_chain,
    create_client_from_runtime_chain,
    get_multi_url_value,
    load_runtime_settings_if_needed,
)
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
    from .vllm_provider import VLLMRerankerClient
    _vllm_available = True
except ImportError:
    VLLMRerankerClient = None
    _vllm_available = False

try:
    from .tei_provider import TEIRerankerClient
    _tei_available = True
except ImportError:
    TEIRerankerClient = None
    _tei_available = False

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

try:
    from .deepinfra_provider import DeepInfraRerankerClient
    _deepinfra_available = True
except ImportError:
    DeepInfraRerankerClient = None
    _deepinfra_available = False
try:
    from .cloudflare_provider import CloudflareRerankerClient
    _cloudflare_available = True
except ImportError:
    CloudflareRerankerClient = None
    _cloudflare_available = False


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
        elif provider == "vllm":
            return cls.vllm(model=model, **kwargs)
        elif provider == "tei":
            return cls.tei(model=model, **kwargs)
        elif provider == "deepinfra":
            return cls.deepinfra(model=model, **kwargs)
        elif provider == "cloudflare":
            return cls.cloudflare(model=model, **kwargs)
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
    def tei(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        **kwargs
    ) -> BaseRerankerClient:
        """Create a Hugging Face TEI reranker client."""
        if not _tei_available or TEIRerankerClient is None:
            raise ImportError(
                "TEI reranker provider not available. Install with: pip install requests"
            )
        return TEIRerankerClient(
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
    def vllm(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        **kwargs,
    ) -> BaseRerankerClient:
        """Create a client for vLLM's standard ``POST /rerank`` API."""
        if not _vllm_available or VLLMRerankerClient is None:
            raise ImportError("vLLM reranker provider not available. Install requests")
        return VLLMRerankerClient(
            model=model,
            base_url=base_url,
            timeout=timeout,
            token=token,
            **kwargs,
        )

    @classmethod
    def deepinfra(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        **kwargs,
    ) -> BaseRerankerClient:
        if not _deepinfra_available or DeepInfraRerankerClient is None:
            raise ImportError("DeepInfra reranker provider not available. Install with: pip install requests")
        return DeepInfraRerankerClient(
            model=model, base_url=base_url, timeout=timeout, token=token, **kwargs
        )

    @classmethod
    def cloudflare(cls, model=None, **kwargs) -> BaseRerankerClient:
        if not _cloudflare_available or CloudflareRerankerClient is None:
            raise ImportError("Cloudflare reranker provider not available. Install with: pip install requests")
        return CloudflareRerankerClient(model=model, **kwargs)

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

        provider_kwargs = build_kwargs_from_config(
            config,
            field_specs=[
                ("api_key", "api_key", "truthy"),
                ("infinity_url", "base_url", "truthy"),
                ("infinity_timeout", "timeout", "truthy"),
                ("infinity_token", "token", "truthy"),
                ("infinity_wake_on_lan", "wake_on_lan", "truthy"),
                ("infinity_wakeup_service", "wakeup_service", "truthy"),
                ("cloudflare_account_id", "account_id", "truthy"),
                ("cloudflare_api_token", "api_token", "truthy"),
                ("device", "device", "truthy"),
                ("cache_dir", "cache_dir", "truthy"),
                ("trust_remote_code", "trust_remote_code", "exists"),
                ("cache_duration_seconds", "cache_duration_seconds", "exists"),
            ],
        )

        return cls.create(
            provider=config.provider,
            model=config.model,
            **provider_kwargs
        )


# Convenience functions
def create_reranker_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    intelligence_level: Optional[int] = None,
    usage: Optional[str] = None,
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
    runtime_settings = load_runtime_settings_if_needed(
        provider,
        loader=lambda: RerankerSettings.from_env(
            load_dotenv=False,
            intelligence_level=intelligence_level,
            usage=usage,
        ),
    )

    client_from_chain = create_client_from_runtime_chain(
        provider=provider,
        runtime_settings=runtime_settings,
        model=model,
        create_fallback=create_fallback_reranker,
        create_single=lambda provider_name, model_name, single_cfg: RerankerFactory.create(
            provider=provider_name,
            model=model_name,
            **single_cfg,
            **kwargs,
        ),
    )
    if client_from_chain is not None:
        return client_from_chain

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
    runtime_settings = RerankerSettings.from_env(load_dotenv=False)
    if runtime_settings.provider_configs and len(runtime_settings.provider_configs) > 1:
        provider_configs = build_provider_chain(runtime_settings.provider_configs)
        return create_fallback_reranker(provider_configs)
    return RerankerFactory.from_config(config=runtime_settings)


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


def create_vllm_reranker(
    model: str = "Qwen/Qwen3-Reranker-0.6B",
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseRerankerClient:
    """Create a reranker using a vLLM pooling server."""
    return RerankerFactory.vllm(model=model, base_url=base_url, **kwargs)


def create_tei_reranker(
    model: str = "Alibaba-NLP/gte-multilingual-reranker-base",
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseRerankerClient:
    """Create a reranker using a Hugging Face TEI server."""
    return RerankerFactory.tei(model=model, base_url=base_url, **kwargs)


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

    infinity_url = get_multi_url_value(
        provider_name="infinity",
        provider_env_map={"infinity": ["INFINITY_BASE_URL"]},
        extra_candidates=[reranker_settings.infinity_url],
    )

    if infinity_url:
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
