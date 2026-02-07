"""Fallback reranker client with automatic provider switching on failures.

Provides transparent failover between multiple reranker providers/hosts for
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

from .base import BaseRerankerClient, RerankerError, RerankResult
from .factory import RerankerFactory
from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()

# Default TTLs for health status caching
HEALTH_STATUS_TTL = 300  # 5 minutes - how long to remember a provider is healthy
FAILURE_STATUS_TTL = 60  # 1 minute - how long to remember a provider failed
OVERLOAD_STATUS_TTL = 30  # 30 seconds - shorter TTL for overload (temporary condition)

# HTTP status codes that indicate temporary overload (not permanent failure)
OVERLOAD_STATUS_CODES = {503, 429}  # Service Unavailable, Too Many Requests


class FallbackRerankerClient(BaseRerankerClient):
    """Reranker client with automatic fallback to backup providers.
    
    Transparently switches between multiple reranker providers when one fails.
    Supports mixing different providers or multiple instances of the same provider
    on different hosts.
    
    Example:
        ```python
        from core_lib.reranker import FallbackRerankerClient
        
        # Multiple Infinity hosts for redundancy
        client = FallbackRerankerClient.from_config([
            {"provider": "infinity", "base_url": "http://infinity1:7997"},
            {"provider": "infinity", "base_url": "http://infinity2:7997"},
            {"provider": "infinity", "base_url": "http://infinity3:7997"},
        ])
        
        # Mixed providers with fallback to cloud
        client = FallbackRerankerClient.from_config([
            {"provider": "infinity", "base_url": "http://localhost:7997"},
            {"provider": "cohere", "api_key": "..."},
        ])
        ```
    """
    
    def __init__(
        self,
        providers: List[BaseRerankerClient],
        model: Optional[str] = None,
        cache_duration_seconds: Optional[int] = None,
        max_retries_per_provider: int = 1,
        fail_on_all_providers: bool = True,
        health_check_interval: int = 60,
        use_health_cache: bool = True,
    ):
        """Initialize fallback client with multiple providers.
        
        Args:
            providers: List of reranker client instances to use as fallback chain
            model: Common model name (overrides individual provider models)
            cache_duration_seconds: Cache TTL in seconds
            max_retries_per_provider: How many times to retry each provider before moving to next
            fail_on_all_providers: If True, raise error when all providers fail. If False, return empty.
            health_check_interval: Seconds between health checks for failed providers
            use_health_cache: Whether to use cache for tracking provider health status
        """
        if not providers:
            raise ValueError("At least one provider must be specified")
        
        # Extract model from first provider if not specified
        first_provider = providers[0]
        model = model or first_provider.model
        
        super().__init__(
            model=model,
            cache_duration_seconds=cache_duration_seconds,
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
            f"Initialized FallbackRerankerClient with {len(providers)} providers: "
            f"model={model}, health_cache={'enabled' if use_health_cache else 'disabled'}"
        )
    
    def _generate_client_id(self) -> str:
        """Generate unique ID for this client based on provider configuration."""
        # Create hash from provider URLs/types to identify this specific configuration
        provider_info = []
        for provider in self.providers:
            info = f"{type(provider).__name__}"
            if hasattr(provider, '_api_client') and hasattr(provider._api_client, 'base_urls'):
                info += f":{','.join(provider._api_client.base_urls)}"
            elif hasattr(provider, 'base_url'):
                info += f":{provider.base_url}"
            elif hasattr(provider, 'api_key'):
                # Use hash of API key to avoid logging sensitive data
                info += f":key_{hashlib.md5(provider.api_key.encode()).hexdigest()[:8]}"
            provider_info.append(info)
        
        config_str = "|".join(provider_info)
        return f"reranker_fallback_{hashlib.md5(config_str.encode()).hexdigest()[:12]}"
    
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
        return f"reranker:fallback:{self._client_id}:provider:{provider_idx}:healthy"
    
    def _get_preferred_provider_key(self) -> str:
        """Get cache key for preferred (currently healthy) provider index."""
        return f"reranker:fallback:{self._client_id}:preferred_provider"
    
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
                logger.debug(f"Marked reranker provider {provider_idx} as healthy in cache")
            except Exception as e:
                logger.debug(f"Could not cache health status: {e}")
    
    def _mark_provider_unhealthy(self, provider_idx: int):
        """Mark a provider as unhealthy in cache."""
        cache = self._get_cache()
        if cache:
            try:
                key = self._get_health_cache_key(provider_idx)
                cache.delete(key)  # Remove healthy status
                logger.debug(f"Marked reranker provider {provider_idx} as unhealthy in cache")
            except Exception as e:
                logger.debug(f"Could not update health status: {e}")
    
    def _get_preferred_provider_index(self) -> Optional[int]:
        """Get the index of the last known healthy provider from cache."""
        cache = self._get_cache()
        if cache:
            try:
                pref_key = self._get_preferred_provider_key()
                cached_idx = cache.get(pref_key)
                if cached_idx:
                    return int(cached_idx)
            except Exception as e:
                logger.debug(f"Could not get preferred provider: {e}")
        return None
    
    def _rerank_raw(
        self,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> tuple[List[RerankResult], Optional[Dict[str, int]]]:
        """Rerank with automatic failover between providers."""
        errors = []
        start_idx = self._get_preferred_provider_index() or self.current_provider_index
        
        # Try each provider in turn
        for attempt in range(len(self.providers)):
            provider_idx = (start_idx + attempt) % len(self.providers)
            provider = self.providers[provider_idx]
            
            # Check if we should retry this provider
            if self.provider_failures.get(provider_idx, 0) >= self.max_retries_per_provider:
                # Check if enough time has passed for a health check
                last_check = self._last_health_check.get(provider_idx, 0)
                if time.time() - last_check < self.health_check_interval:
                    logger.debug(
                        f"Skipping provider {provider_idx} (too many failures, "
                        f"next health check in {self.health_check_interval - (time.time() - last_check):.0f}s)"
                    )
                    continue
                # Time for a health check - reset failure count
                self.provider_failures[provider_idx] = 0
                self._last_health_check[provider_idx] = time.time()
            
            try:
                logger.debug(f"Attempting rerank with provider {provider_idx} ({type(provider).__name__})")
                
                results, usage = provider._rerank_raw(query, documents, top_k)
                
                # Success - update tracking
                self.current_provider_index = provider_idx
                self.provider_failures[provider_idx] = 0
                self.provider_overloads[provider_idx] = 0
                self._mark_provider_healthy(provider_idx)
                
                logger.debug(
                    f"Reranking succeeded with provider {provider_idx} "
                    f"(returned {len(results)} results)"
                )
                
                return results, usage
                
            except RerankerError as e:
                error_msg = f"Provider {provider_idx} failed: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                
                # Track failure
                self.provider_failures[provider_idx] = self.provider_failures.get(provider_idx, 0) + 1
                self._mark_provider_unhealthy(provider_idx)
                
                # Check if this looks like temporary overload
                if any(code in str(e) for code in ['503', '429', 'timeout', 'overload']):
                    self.provider_overloads[provider_idx] = self.provider_overloads.get(provider_idx, 0) + 1
                    logger.debug(f"Provider {provider_idx} appears overloaded")
                
            except Exception as e:
                error_msg = f"Provider {provider_idx} unexpected error: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
                self.provider_failures[provider_idx] = self.provider_failures.get(provider_idx, 0) + 1
                self._mark_provider_unhealthy(provider_idx)
        
        # All providers failed
        if self.fail_on_all_providers:
            error_summary = (
                f"All {len(self.providers)} reranker providers failed. "
                f"Errors: {'; '.join(errors[:3])}"  # Limit to first 3 errors
            )
            logger.error(error_summary)
            raise RerankerError(error_summary)
        else:
            logger.warning(f"All reranker providers failed, returning empty results")
            return [], None
    
    def health_check(self) -> bool:
        """Check if at least one provider is healthy."""
        for idx, provider in enumerate(self.providers):
            try:
                if provider.health_check():
                    self._mark_provider_healthy(idx)
                    return True
            except Exception as e:
                logger.debug(f"Health check failed for provider {idx}: {e}")
                self._mark_provider_unhealthy(idx)
        
        logger.warning("All reranker providers failed health check")
        return False
    
    def get_provider_status(self) -> List[Dict[str, Any]]:
        """Get status information for all providers.
        
        Returns:
            List of dicts with provider status info
        """
        status = []
        for idx, provider in enumerate(self.providers):
            provider_info = {
                'index': idx,
                'type': type(provider).__name__,
                'model': provider.model,
                'is_current': idx == self.current_provider_index,
                'failures': self.provider_failures.get(idx, 0),
                'overloads': self.provider_overloads.get(idx, 0),
            }
            
            # Add URL info if available
            if hasattr(provider, '_api_client') and hasattr(provider._api_client, 'base_urls'):
                provider_info['urls'] = provider._api_client.base_urls
            elif hasattr(provider, 'base_url'):
                provider_info['url'] = provider.base_url
            
            status.append(provider_info)
        
        return status
    
    @classmethod
    def from_config(
        cls,
        configs: List[Dict[str, Any]],
        **fallback_kwargs
    ) -> "FallbackRerankerClient":
        """Create fallback client from provider configuration list.
        
        Args:
            configs: List of provider configurations, each a dict with:
                - provider: Provider name ('infinity', 'cohere', 'local')
                - Additional provider-specific params (base_url, api_key, etc.)
            **fallback_kwargs: Arguments for FallbackRerankerClient constructor
            
        Returns:
            Configured fallback reranker client
            
        Example:
            ```python
            client = FallbackRerankerClient.from_config([
                {"provider": "infinity", "base_url": "http://192.168.1.100:7997"},
                {"provider": "infinity", "base_url": "http://192.168.1.101:7997"},
                {"provider": "cohere", "api_key": "..."},
            ], max_retries_per_provider=2)
            ```
        """
        providers = []
        for config in configs:
            provider_type = config.pop('provider', 'infinity')
            provider = RerankerFactory.create(provider_type, **config)
            providers.append(provider)
        
        return cls(providers=providers, **fallback_kwargs)
