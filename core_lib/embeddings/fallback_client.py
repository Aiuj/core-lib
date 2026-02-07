"""Fallback embedding client with automatic provider switching on failures.

Provides transparent failover between multiple embedding providers/hosts for
production reliability. Automatically retries failed requests on backup providers.

Features smart health tracking:
- Caches healthy provider index to avoid unnecessary retries
- Automatically switches to backup on failure
- Prevents infinite loops when all providers fail
- Periodic health checks to restore failed providers
- Detects temporary overload (503, timeouts) and recovers automatically
"""

from __future__ import annotations

from typing import List, Union, Optional, Dict, Any
import time
import hashlib

from .base import BaseEmbeddingClient, EmbeddingGenerationError
from .factory import EmbeddingFactory
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()

# Default TTLs for health status caching
HEALTH_STATUS_TTL = 300  # 5 minutes - how long to remember a provider is healthy
FAILURE_STATUS_TTL = 60  # 1 minute - how long to remember a provider failed
OVERLOAD_STATUS_TTL = 30  # 30 seconds - shorter TTL for overload (temporary condition)

# HTTP status codes that indicate temporary overload (not permanent failure)
OVERLOAD_STATUS_CODES = {503, 429}  # Service Unavailable, Too Many Requests


class FallbackEmbeddingClient(BaseEmbeddingClient):
    """Embedding client with automatic fallback to backup providers.
    
    Transparently switches between multiple embedding providers when one fails.
    Supports mixing different providers or multiple instances of the same provider
    on different hosts.
    
    Example:
        ```python
        from core_lib.embeddings import FallbackEmbeddingClient
        
        # Multiple Infinity hosts for redundancy
        client = FallbackEmbeddingClient.from_config([
            {"provider": "infinity", "base_url": "http://infinity1:7997"},
            {"provider": "infinity", "base_url": "http://infinity2:7997"},
            {"provider": "infinity", "base_url": "http://infinity3:7997"},
        ])
        
        # Mixed providers with fallback to cloud
        client = FallbackEmbeddingClient.from_config([
            {"provider": "infinity", "base_url": "http://localhost:7997"},
            {"provider": "ollama", "base_url": "http://localhost:11434"},
            {"provider": "openai", "api_key": "sk-..."},
        ])
        ```
    """
    
    def __init__(
        self,
        providers: List[BaseEmbeddingClient],
        model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        use_l2_norm: bool = True,
        cache_duration_seconds: Optional[int] = None,
        norm_method: Optional[str] = None,
        max_retries_per_provider: int = 1,
        fail_on_all_providers: bool = True,
        health_check_interval: int = 60,
        use_health_cache: bool = True,
    ):
        """Initialize fallback client with multiple providers.
        
        Args:
            providers: List of embedding client instances to use as fallback chain
            model: Common model name (overrides individual provider models)
            embedding_dim: Common embedding dimension (overrides individual settings)
            use_l2_norm: Whether to apply L2 normalization
            cache_duration_seconds: Cache TTL in seconds
            norm_method: Normalization method for dimension changes
            max_retries_per_provider: How many times to retry each provider before moving to next
            fail_on_all_providers: If True, raise error when all providers fail. If False, return None.
            health_check_interval: Seconds between health checks for failed providers
            use_health_cache: Whether to use cache for tracking provider health status
        """
        if not providers:
            raise ValueError("At least one provider must be specified")
        
        # Extract model and dimension from first provider if not specified
        first_provider = providers[0]
        model = model or first_provider.model
        embedding_dim = embedding_dim or first_provider.embedding_dim
        
        super().__init__(
            model=model,
            embedding_dim=embedding_dim,
            use_l2_norm=use_l2_norm,
            cache_duration_seconds=cache_duration_seconds,
            norm_method=norm_method,
        )
        
        self.providers = providers
        self.max_retries_per_provider = max_retries_per_provider
        self.fail_on_all_providers = fail_on_all_providers
        self.current_provider_index = 0
        self.provider_failures: Dict[int, int] = {i: 0 for i in range(len(providers))}
        self.provider_overloads: Dict[int, int] = {i: 0 for i in range(len(providers))}
        self.health_check_interval = health_check_interval
        self.use_health_cache = use_health_cache
        self._last_health_check: Dict[int, float] = {}
        self._cache_instance = None
        
        # Generate unique identifier for this fallback client config
        self._client_id = self._generate_client_id()
        
        logger.debug(
            f"Initialized FallbackEmbeddingClient with {len(providers)} providers: "
            f"model={model}, embedding_dim={embedding_dim}, "
            f"health_cache={'enabled' if use_health_cache else 'disabled'}"
        )
    
    def _generate_client_id(self) -> str:
        """Generate unique ID for this client based on provider configuration."""
        # Create hash from provider URLs/types to identify this specific configuration
        provider_info = []
        for provider in self.providers:
            info = f"{type(provider).__name__}"
            if hasattr(provider, 'base_url'):
                info += f":{provider.base_url}"
            elif hasattr(provider, 'api_key'):
                # Use hash of API key to avoid logging sensitive data
                info += f":key_{hashlib.md5(provider.api_key.encode()).hexdigest()[:8]}"
            provider_info.append(info)
        
        config_str = "|".join(provider_info)
        return f"fallback_{hashlib.md5(config_str.encode()).hexdigest()[:12]}"
    
    def _get_cache(self):
        """Get cache instance if available and caching is enabled."""
        if not self.use_health_cache:
            return None
        
        # Cache the cache instance itself
        if self._cache_instance is None:
            try:
                from core_lib.cache import get_cache
                self._cache_instance = get_cache()
            except Exception as e:
                logger.debug(f"Cache not available for health tracking: {e}")
                self._cache_instance = False  # Mark as unavailable
        
        return self._cache_instance if self._cache_instance else None
    
    def _get_health_cache_key(self, provider_idx: int) -> str:
        """Get cache key for provider health status."""
        return f"embedding:fallback:{self._client_id}:provider:{provider_idx}:healthy"
    
    def _get_overload_cache_key(self, provider_idx: int) -> str:
        """Get cache key for provider overload status."""
        return f"embedding:fallback:{self._client_id}:provider:{provider_idx}:overloaded"
    
    def _get_preferred_provider_key(self) -> str:
        """Get cache key for preferred (currently healthy) provider index."""
        return f"embedding:fallback:{self._client_id}:preferred_provider"
    
    def _mark_provider_healthy(self, provider_idx: int):
        """Mark a provider as healthy in cache."""
        cache = self._get_cache()
        if cache:
            try:
                key = self._get_health_cache_key(provider_idx)
                cache.set(key, "1", ttl=HEALTH_STATUS_TTL)
                # Also update preferred provider
                pref_key = self._get_preferred_provider_key()
                cache.set(pref_key, str(provider_idx), ttl=HEALTH_STATUS_TTL)
                logger.debug(f"Marked provider {provider_idx} as healthy in cache")
            except Exception as e:
                logger.debug(f"Could not cache health status: {e}")
    
    def _mark_provider_unhealthy(self, provider_idx: int):
        """Mark a provider as unhealthy in cache."""
        cache = self._get_cache()
        if cache:
            try:
                key = self._get_health_cache_key(provider_idx)
                cache.delete(key)  # Remove healthy status
                logger.debug(f"Marked provider {provider_idx} as unhealthy in cache")
            except Exception as e:
                logger.debug(f"Could not update health status: {e}")
    
    def _mark_provider_overloaded(self, provider_idx: int):
        """Mark a provider as temporarily overloaded in cache.
        
        Overload is different from failure - it's temporary and should recover quickly.
        Uses shorter TTL than permanent failures.
        """
        cache = self._get_cache()
        if cache:
            try:
                key = self._get_overload_cache_key(provider_idx)
                cache.set(key, "1", ttl=OVERLOAD_STATUS_TTL)
                # Remove from healthy status
                health_key = self._get_health_cache_key(provider_idx)
                cache.delete(health_key)
                logger.info(
                    f"Marked provider {provider_idx} as temporarily overloaded "
                    f"(will recheck in {OVERLOAD_STATUS_TTL}s)"
                )
            except Exception as e:
                logger.debug(f"Could not cache overload status: {e}")
    
    def _is_provider_overloaded_cached(self, provider_idx: int) -> bool:
        """Check if provider is marked as overloaded in cache.
        
        Returns:
            True if known to be overloaded, False otherwise
        """
        cache = self._get_cache()
        if not cache:
            return False
        
        try:
            key = self._get_overload_cache_key(provider_idx)
            value = cache.get(key)
            return value == "1"
        except Exception as e:
            logger.debug(f"Could not check cached overload status: {e}")
            return False
    
    def _is_provider_healthy_cached(self, provider_idx: int) -> Optional[bool]:
        """Check if provider is marked as healthy in cache.
        
        Returns:
            True if healthy, False if known unhealthy, None if unknown
        """
        cache = self._get_cache()
        if not cache:
            return None
        
        try:
            key = self._get_health_cache_key(provider_idx)
            value = cache.get(key)
            return value == "1" if value is not None else None
        except Exception as e:
            logger.debug(f"Could not check cached health status: {e}")
            return None
    
    def _get_preferred_provider(self) -> Optional[int]:
        """Get the currently preferred (last successful) provider from cache."""
        cache = self._get_cache()
        if not cache:
            return None
        
        try:
            key = self._get_preferred_provider_key()
            value = cache.get(key)
            if value is not None:
                idx = int(value)
                if 0 <= idx < len(self.providers):
                    logger.debug(f"Using preferred provider {idx} from cache")
                    return idx
        except Exception as e:
            logger.debug(f"Could not get preferred provider: {e}")
        
        return None
    
    def _should_check_health(self, provider_idx: int) -> bool:
        """Determine if we should perform a health check on this provider.
        
        Used to avoid hammering failed providers - only check periodically.
        Overloaded providers get checked more frequently than failed ones.
        """
        now = time.time()
        last_check = self._last_health_check.get(provider_idx, 0)
        
        # If provider is overloaded, use shorter interval (OVERLOAD_STATUS_TTL)
        if self._is_provider_overloaded_cached(provider_idx):
            return (now - last_check) >= OVERLOAD_STATUS_TTL
        
        # Otherwise use normal health check interval
        return (now - last_check) >= self.health_check_interval
    
    def _is_connection_error(self, error: Exception) -> bool:
        """Determine if an error indicates server is unreachable.
        
        Connection error indicators:
        - ConnectionError, ConnectTimeout
        - Name resolution failed
        - Connection refused
        - Network unreachable
        
        Returns:
            True if server is unreachable, False otherwise
        """
        error_str = str(error).lower()
        
        # Check for connection-related exception types
        # Note: requests.exceptions.ConnectionError includes multiple underlying errors
        if 'connectionerror' in type(error).__name__.lower():
            return True
        
        # Check error message for connection indicators
        connection_indicators = [
            'connection refused',
            'connection reset',
            'network unreachable',
            'no route to host',
            'name resolution failed',
            'nodename nor servname provided',
            'failed to connect',
            'cannot connect',
            'host is unreachable',
        ]
        
        return any(indicator in error_str for indicator in connection_indicators)
    
    def _is_overload_error(self, error: Exception) -> bool:
        """Determine if an error indicates temporary overload vs permanent failure.
        
        Overload indicators:
        - HTTP 503 (Service Unavailable)
        - HTTP 429 (Too Many Requests)
        - Timeout errors (server connected but slow to respond)
        - Connection pool exhausted
        
        Note: This returns False for connection errors (see _is_connection_error)
        
        Returns:
            True if error indicates temporary overload, False for permanent failure
        """
        # First check if this is a connection error - those are NOT overload
        if self._is_connection_error(error):
            return False
        
        error_str = str(error).lower()
        
        # Check for HTTP status codes indicating overload
        if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            if error.response.status_code in OVERLOAD_STATUS_CODES:
                return True
        
        # Check error message for overload indicators
        overload_indicators = [
            '503',  # Service Unavailable
            '429',  # Too Many Requests
            'service unavailable',
            'too many requests',
            'temporarily unavailable',
            'server overloaded',
            'timeout',
            'timed out',
            'pool exhausted',
            'connection pool',
        ]
        
        return any(indicator in error_str for indicator in overload_indicators)
    
    @classmethod
    def from_config(
        cls,
        provider_configs: List[Dict[str, Any]],
        common_model: Optional[str] = None,
        common_embedding_dim: Optional[int] = None,
        common_use_l2_norm: bool = True,
        common_cache_duration: Optional[int] = None,
        common_norm_method: Optional[str] = None,
        max_retries_per_provider: int = 1,
        fail_on_all_providers: bool = True,
    ) -> "FallbackEmbeddingClient":
        """Create fallback client from configuration dictionaries.
        
        Args:
            provider_configs: List of dicts with provider configuration.
                Each dict should have 'provider' key and provider-specific settings.
                Example: [
                    {"provider": "infinity", "base_url": "http://host1:7997", "model": "bge-small"},
                    {"provider": "infinity", "base_url": "http://host2:7997"},
                    {"provider": "openai", "api_key": "sk-..."},
                ]
            common_model: Model to use for all providers (overrides per-provider settings)
            common_embedding_dim: Embedding dimension for all providers
            common_use_l2_norm: L2 normalization setting for all providers
            common_cache_duration: Cache duration for all providers
            common_norm_method: Normalization method for all providers
            max_retries_per_provider: Retries per provider before fallback
            fail_on_all_providers: Whether to raise exception if all providers fail
            
        Returns:
            Configured FallbackEmbeddingClient instance
        """
        providers = []
        
        for i, config in enumerate(provider_configs):
            config = config.copy()  # Don't modify original
            provider_name = config.pop("provider", None)
            
            if not provider_name:
                raise ValueError(f"Provider config at index {i} missing 'provider' key")
            
            # Apply common settings (overriding per-provider settings if specified)
            if common_model:
                config["model"] = common_model
            if common_embedding_dim:
                config["embedding_dim"] = common_embedding_dim
            if common_use_l2_norm is not None:
                config["use_l2_norm"] = common_use_l2_norm
            if common_norm_method:
                config["norm_method"] = common_norm_method
            
            # Create provider instance
            try:
                provider = EmbeddingFactory.create(provider=provider_name, **config)
                providers.append(provider)
                logger.debug(f"Created provider {i}: {provider_name} with config: {config}")
            except Exception as e:
                logger.warning(f"Failed to create provider {i} ({provider_name}): {e}")
                # Continue with other providers
        
        if not providers:
            raise ValueError("Failed to create any providers from configuration")
        
        return cls(
            providers=providers,
            model=common_model,
            embedding_dim=common_embedding_dim,
            use_l2_norm=common_use_l2_norm,
            cache_duration_seconds=common_cache_duration,
            norm_method=common_norm_method,
            max_retries_per_provider=max_retries_per_provider,
            fail_on_all_providers=fail_on_all_providers,
        )
    
    @classmethod
    def from_env(
        cls,
        provider: Optional[str] = None,
        use_infinity: bool = True,
    ) -> "FallbackEmbeddingClient":
        """Create fallback client from environment variables using comma-separated URLs.
        
        Supports two configuration patterns:
        
        1. Provider-specific with comma-separated hosts (recommended):
            EMBEDDING_PROVIDER=infinity
            INFINITY_BASE_URL=http://host1:7997,http://host2:7997,http://host3:7997
            EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
            EMBEDDING_TIMEOUT=30
            
        2. Generic with comma-separated hosts:
            EMBEDDING_PROVIDER=infinity
            EMBEDDING_BASE_URL=http://host1:7997,http://host2:7997
            EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
            
        Args:
            provider: Provider type (infinity, ollama, openai). If None, reads EMBEDDING_PROVIDER.
            use_infinity: If True and no URLs found, defaults to Infinity provider.
            
        Returns:
            Configured FallbackEmbeddingClient instance
        """
        import os
        
        # Get provider from env or parameter
        provider = provider or os.getenv("EMBEDDING_PROVIDER", "infinity" if use_infinity else None)
        if not provider:
            raise ValueError("EMBEDDING_PROVIDER not set and no provider specified")
        
        # Common settings from environment
        common_model = os.getenv("EMBEDDING_MODEL")
        common_dimension = os.getenv("EMBEDDING_DIMENSION")
        if common_dimension:
            common_dimension = int(common_dimension)
        common_timeout = os.getenv("EMBEDDING_TIMEOUT")
        if common_timeout:
            common_timeout = int(common_timeout)
        common_cache_duration = os.getenv("EMBEDDING_CACHE_DURATION_SECONDS")
        if common_cache_duration:
            common_cache_duration = int(common_cache_duration)
        
        # Get provider-specific or generic base URLs (comma-separated)
        base_urls = None
        tokens = None  # Comma-separated tokens matching URLs
        
        if provider == "infinity":
            infinity_url = os.getenv("INFINITY_BASE_URL")
            if infinity_url:
                base_urls = [url.strip() for url in infinity_url.split(",")]
            infinity_token = os.getenv("INFINITY_TOKEN") or os.getenv("EMBEDDING_TOKEN")
            if infinity_token:
                tokens = [t.strip() for t in infinity_token.split(",")]
        elif provider == "ollama":
            ollama_url = os.getenv("OLLAMA_URL")
            if ollama_url:
                base_urls = [url.strip() for url in ollama_url.split(",")]
        elif provider == "openai":
            openai_url = os.getenv("OPENAI_BASE_URL")
            if openai_url:
                base_urls = [url.strip() for url in openai_url.split(",")]
            # OpenAI uses API key
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            if api_key:
                tokens = [t.strip() for t in api_key.split(",")]
        
        # Fallback to generic EMBEDDING_BASE_URL
        if not base_urls:
            embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
            if embedding_base_url:
                base_urls = [url.strip() for url in embedding_base_url.split(",")]
            # Also check for generic token
            if not tokens:
                embedding_token = os.getenv("EMBEDDING_TOKEN")
                if embedding_token:
                    tokens = [t.strip() for t in embedding_token.split(",")]
        
        if not base_urls:
            raise ValueError(
                f"No base URLs found for provider '{provider}'. "
                f"Set EMBEDDING_BASE_URL or provider-specific URL variable with comma-separated hosts."
            )
        
        # Build provider configs for each URL
        provider_configs = []
        for i, base_url in enumerate(base_urls):
            config = {
                "provider": provider,
                "base_url": base_url,
            }
            
            # Add token/API key if available (use matching index or last token if fewer tokens than URLs)
            if tokens:
                token_index = min(i, len(tokens) - 1)
                if provider == "openai":
                    config["api_key"] = tokens[token_index]
                elif provider == "infinity":
                    config["token"] = tokens[token_index]
            
            # Add provider-specific settings
            if provider == "openai":
                org = os.getenv("OPENAI_ORGANIZATION")
                if org:
                    config["organization"] = org
            
            if common_timeout:
                config["timeout"] = common_timeout
            
            provider_configs.append(config)
        
        logger.debug(
            f"Creating FallbackEmbeddingClient with {len(provider_configs)} {provider} "
            f"instances from environment: {[c['base_url'] for c in provider_configs]}"
        )
        
        return cls.from_config(
            provider_configs=provider_configs,
            common_model=common_model,
            common_embedding_dim=common_dimension,
            common_cache_duration=common_cache_duration,
            max_retries_per_provider=1,
            fail_on_all_providers=True,
        )
    
    def _generate_embedding_raw(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using fallback chain with smart health tracking.
        
        Tries each provider in sequence until one succeeds or all fail.
        Uses cached health status to prefer known-healthy providers and avoid
        repeatedly trying failed providers.
        """
        last_error = None
        providers_tried = []
        all_providers_failed = True
        
        # Get preferred provider from cache (last known healthy)
        preferred_idx = self._get_preferred_provider()
        if preferred_idx is not None:
            # Start with preferred provider
            start_idx = preferred_idx
        else:
            # Start with current index
            start_idx = self.current_provider_index
        
        # Track which providers we've tried to prevent infinite loops
        tried_providers = set()
        
        # Try each provider, starting with preferred/current
        for attempt in range(len(self.providers)):
            # Calculate which provider to try
            idx = (start_idx + attempt) % len(self.providers)
            
            # Skip if already tried (prevents infinite loop)
            if idx in tried_providers:
                continue
            tried_providers.add(idx)
            
            provider = self.providers[idx]
            
            # Check cached health status
            cached_health = self._is_provider_healthy_cached(idx)
            cached_overload = self._is_provider_overloaded_cached(idx)
            
            if cached_overload:
                # Provider is overloaded - skip unless it's time to recheck
                if not self._should_check_health(idx):
                    logger.debug(
                        f"Skipping provider {idx} (temporarily overloaded, "
                        f"next check in {OVERLOAD_STATUS_TTL}s)"
                    )
                    continue
                else:
                    logger.debug(f"Rechecking overloaded provider {idx}")
            elif cached_health is False and not self._should_check_health(idx):
                # Skip known-unhealthy provider if not time for health check
                logger.debug(
                    f"Skipping provider {idx} (known unhealthy, "
                    f"next check in {self.health_check_interval}s)"
                )
                continue
            
            # Try this provider with retries
            provider_succeeded = False
            for retry in range(self.max_retries_per_provider):
                try:
                    start_time = time.time()
                    result = provider._generate_embedding_raw(texts)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Success! Update stats and cache
                    all_providers_failed = False
                    provider_succeeded = True
                    self.current_provider_index = idx  # Prefer this provider next time
                    self.embedding_time_ms = elapsed_ms
                    self._last_health_check[idx] = time.time()
                    
                    # Mark provider as healthy in cache
                    self._mark_provider_healthy(idx)
                    
                    if attempt > 0 or retry > 0:
                        logger.info(
                            f"Embedding succeeded on provider {idx} "
                            f"(attempt {retry + 1}/{self.max_retries_per_provider}) "
                            f"after {len(providers_tried)} previous failures"
                        )
                    
                    return result
                    
                except Exception as e:
                    last_error = e
                    self.provider_failures[idx] = self.provider_failures.get(idx, 0) + 1
                    providers_tried.append(f"{idx}:{type(provider).__name__}")
                    
                    # Get provider info for logging
                    provider_info = type(provider).__name__
                    if hasattr(provider, 'base_url'):
                        provider_info = f"{provider_info} ({provider.base_url})"
                    
                    # Determine error type: connection error vs overload vs other failure
                    is_connection_error = self._is_connection_error(e)
                    is_overload = self._is_overload_error(e)
                    
                    if is_connection_error:
                        # Server is unreachable (network issue, host down)
                        logger.warning(
                            f"Provider {idx} ({provider_info}) is unreachable "
                            f"(retry {retry + 1}/{self.max_retries_per_provider}): {e}"
                        )
                    elif is_overload:
                        # Server is reachable but overloaded/slow
                        self.provider_overloads[idx] = self.provider_overloads.get(idx, 0) + 1
                        logger.warning(
                            f"Provider {idx} ({provider_info}) is overloaded "
                            f"(retry {retry + 1}/{self.max_retries_per_provider}): {e}"
                        )
                    else:
                        # Other type of failure
                        logger.warning(
                            f"Provider {idx} ({provider_info}) failed "
                            f"(retry {retry + 1}/{self.max_retries_per_provider}): {e}"
                        )
                    
                    if retry < self.max_retries_per_provider - 1:
                        time.sleep(0.1 * (retry + 1))  # Exponential backoff
            
            # All retries for this provider failed
            if not provider_succeeded:
                # Mark provider based on error type
                if last_error and self._is_overload_error(last_error):
                    # Temporary overload - mark with shorter TTL
                    self._mark_provider_overloaded(idx)
                else:
                    # Permanent failure or unknown error
                    self._mark_provider_unhealthy(idx)
                
                self._last_health_check[idx] = time.time()
        
        # All providers failed - ensure we don't have infinite loop
        if all_providers_failed:
            error_msg = (
                f"All {len(self.providers)} embedding providers failed "
                f"(tried {len(tried_providers)} providers: {', '.join(providers_tried)}). "
                f"Last error: {last_error}"
            )
            
            logger.error(error_msg)
            
            if self.fail_on_all_providers:
                raise EmbeddingGenerationError(error_msg)
            else:
                return None
        
        # Should never reach here, but safety check
        raise EmbeddingGenerationError(
            f"Unexpected fallback state: tried {len(tried_providers)} providers "
            f"but no result obtained"
        )
    
    def health_check(self) -> bool:
        """Check if at least one provider is healthy.
        
        Checks cached health status first, then performs actual health checks
        if needed. Updates cache with results.
        """
        for i, provider in enumerate(self.providers):
            # Check cached status first
            cached_health = self._is_provider_healthy_cached(i)
            if cached_health is True:
                logger.debug(f"Provider {i} is cached as healthy")
                return True
            
            # Perform actual health check if not cached or cache unavailable
            if cached_health is None or self._should_check_health(i):
                try:
                    if provider.health_check():
                        logger.debug(f"Provider {i} ({type(provider).__name__}) is healthy")
                        self._mark_provider_healthy(i)
                        self._last_health_check[i] = time.time()
                        return True
                    else:
                        self._mark_provider_unhealthy(i)
                        self._last_health_check[i] = time.time()
                except Exception as e:
                    logger.debug(f"Provider {i} health check failed: {e}")
                    self._mark_provider_unhealthy(i)
                    self._last_health_check[i] = time.time()
        
        logger.warning("All providers failed health check")
        return False
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about provider usage and failures.
        
        Returns:
            Dictionary with provider stats including failure counts, overload counts,
            current provider, and cached health status
        """
        stats = {
            "total_providers": len(self.providers),
            "current_provider": self.current_provider_index,
            "provider_failures": self.provider_failures,
            "provider_overloads": self.provider_overloads,
            "health_cache_enabled": self.use_health_cache,
            "providers": [],
        }
        
        for i, provider in enumerate(self.providers):
            provider_stat = {
                "index": i,
                "type": type(provider).__name__,
                "model": provider.model,
                "failures": self.provider_failures.get(i, 0),
                "overloads": self.provider_overloads.get(i, 0),
                "cached_healthy": self._is_provider_healthy_cached(i),
                "cached_overloaded": self._is_provider_overloaded_cached(i),
                "last_health_check": self._last_health_check.get(i),
            }
            
            # Add provider-specific info
            if hasattr(provider, 'base_url'):
                provider_stat["base_url"] = provider.base_url
            
            stats["providers"].append(provider_stat)
        
        # Add preferred provider from cache
        preferred = self._get_preferred_provider()
        if preferred is not None:
            stats["preferred_provider"] = preferred
        
        return stats
    
    def reset_failures(self):
        """Reset failure counters and health cache for all providers."""
        self.provider_failures = {i: 0 for i in range(len(self.providers))}
        self.provider_overloads = {i: 0 for i in range(len(self.providers))}
        self._last_health_check = {}
        
        # Clear health status from cache
        cache = self._get_cache()
        if cache:
            try:
                for i in range(len(self.providers)):
                    # Clear health status
                    health_key = self._get_health_cache_key(i)
                    cache.delete(health_key)
                    # Clear overload status
                    overload_key = self._get_overload_cache_key(i)
                    cache.delete(overload_key)
                # Also clear preferred provider
                pref_key = self._get_preferred_provider_key()
                cache.delete(pref_key)
                logger.info("Reset all provider failure/overload counters and health cache")
            except Exception as e:
                logger.debug(f"Could not clear health cache: {e}")
        else:
            logger.info("Reset all provider failure/overload counters")
    
    def force_provider(self, provider_idx: int):
        """Force use of a specific provider and mark it as healthy.
        
        Args:
            provider_idx: Index of provider to use
            
        Raises:
            ValueError: If provider_idx is out of range
        """
        if provider_idx < 0 or provider_idx >= len(self.providers):
            raise ValueError(
                f"Provider index {provider_idx} out of range (0-{len(self.providers)-1})"
            )
        
        self.current_provider_index = provider_idx
        self._mark_provider_healthy(provider_idx)
        logger.info(f"Forced provider {provider_idx} and marked as healthy")
