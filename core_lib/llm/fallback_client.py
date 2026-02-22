"""Fallback LLM client with automatic provider switching on failures.

Provides transparent failover between multiple LLM providers for production
reliability. Automatically retries failed requests on backup providers and
tracks provider health for intelligent routing.

Features:
- Automatic failover to backup providers on failure
- Health tracking with TTL-based recovery
- Transparent to the application - just call chat() like normal
- Smart error classification for appropriate recovery times
- Support for intelligence-level-based provider selection

Example:
    ```python
    from core_lib.llm import FallbackLLMClient
    
    # Create from environment/configuration
    client = FallbackLLMClient.from_env()
    
    # Or with explicit providers
    client = FallbackLLMClient.from_config([
        {"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash"},
        {"provider": "openai", "api_key": "...", "model": "gpt-4o-mini"},
        {"provider": "ollama", "host": "http://localhost:11434", "model": "llama3.2"},
    ])
    
    # Use just like a normal LLMClient - fallback is transparent
    response = client.chat("What is the capital of France?")
    
    # Check which provider was used
    print(f"Used provider: {client.last_used_provider}")
    print(f"Was fallback: {client.last_was_fallback}")
    ```
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

from core_lib.tracing.logger import get_module_logger
from core_lib.tracing.service_usage import set_intelligence_level

from .llm_client import LLMClient
from .provider_health import classify_error, get_health_tracker, ProviderHealthTracker
from .provider_registry import ProviderConfig, ProviderRegistry

logger = get_module_logger()


@dataclass
class FallbackResult:
    """Result from a fallback-enabled chat call with metadata."""
    
    content: Any
    """The response content (string, dict, or structured output)."""
    
    provider: str
    """Provider name that generated the response."""
    
    model: str
    """Model name that generated the response."""
    
    was_fallback: bool
    """Whether a fallback provider was used."""
    
    attempts: int
    """Number of provider attempts before success."""
    
    usage: Dict[str, Any]
    """Token usage information."""
    
    tool_calls: Optional[List[Dict[str, Any]]] = None
    """Tool calls if any were made."""
    
    structured: bool = False
    """Whether structured output was requested."""
    
    error: Optional[str] = None
    """Error message if the request failed."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard LLMClient response dict format."""
        return {
            "content": self.content,
            "usage": self.usage,
            "tool_calls": self.tool_calls or [],
            "structured": self.structured,
            "error": self.error,
            "_fallback_metadata": {
                "provider": self.provider,
                "model": self.model,
                "was_fallback": self.was_fallback,
                "attempts": self.attempts,
            },
        }


class FallbackLLMClient:
    """LLM client with automatic fallback to backup providers.
    
    Transparently switches between multiple LLM providers when one fails.
    Tracks provider health and prefers healthy providers to minimize latency.
    
    This class provides the same `chat()` interface as `LLMClient` but adds:
    - Automatic retry with fallback providers
    - Health tracking with TTL-based recovery
    - Intelligent error classification
    - Metadata about which provider was actually used
    
    Attributes:
        last_used_provider: Name of the provider used for the last request
        last_used_model: Model name used for the last request
        last_was_fallback: Whether the last request used a fallback provider
        last_attempts: Number of attempts for the last request
    """
    
    def __init__(
        self,
        registry: ProviderRegistry,
        health_tracker: Optional[ProviderHealthTracker] = None,
        max_retries_per_provider: int = 1,
        intelligence_level: Optional[int] = None,
        http_timeout_ms: Optional[int] = None,
    ):
        """Initialize fallback client with a provider registry.
        
        Args:
            registry: ProviderRegistry with configured providers
            health_tracker: Optional custom health tracker (uses global if None)
            max_retries_per_provider: Retries per provider before moving to next
            intelligence_level: Optional default intelligence level for filtering
            http_timeout_ms: Optional HTTP timeout override (ms) for Google GenAI providers.
                Overrides GOOGLE_GENAI_HTTP_TIMEOUT_MS env var when set.
        """
        if not registry or not registry.providers:
            raise ValueError("Registry must have at least one configured provider")
        
        self._registry = registry
        self._health_tracker = health_tracker or get_health_tracker()
        self._max_retries = max_retries_per_provider
        self._default_intelligence_level = intelligence_level
        self._http_timeout_ms = http_timeout_ms
        
        # Last request metadata (for observability)
        self.last_used_provider: Optional[str] = None
        self.last_used_model: Optional[str] = None
        self.last_was_fallback: bool = False
        self.last_attempts: int = 0
        
        # Client cache to avoid recreating clients
        self._client_cache: Dict[str, LLMClient] = {}
        
        # Log providers in priority order with their level ranges (skip disabled ones)
        provider_details = []
        for p in sorted(registry.providers, key=lambda x: x.priority):
            if not getattr(p, 'enabled', True):
                continue
            tier_info = f" [{p.tier}]" if hasattr(p, 'tier') and p.tier else ""
            level_info = f" (IQ{p.min_intelligence_level}-{p.max_intelligence_level})" if p.min_intelligence_level is not None else ""
            provider_details.append(
                f"{p.provider}:{p.model}{tier_info}{level_info} #{p.priority}"
            )
        
        logger.info(
            f"Initialized FallbackLLMClient with {len(provider_details)} providers: {', '.join(provider_details)}"
        )
    
    def _get_client(self, config: ProviderConfig) -> LLMClient:
        """Get or create a cached LLMClient for a provider config."""
        cache_key = self._build_cache_key(config)
        if cache_key not in self._client_cache:
            if self._http_timeout_ms is not None and config.provider in ("gemini", "vertex"):
                llm_config = config.to_llm_config()
                llm_config.http_timeout_ms = self._http_timeout_ms
                self._client_cache[cache_key] = LLMClient(llm_config)
            else:
                self._client_cache[cache_key] = config.to_client()
        return self._client_cache[cache_key]

    def _build_cache_key(self, config: ProviderConfig) -> str:
        """Build a stable cache key for a specific provider configuration.

        Includes enough config identity to avoid collisions when multiple entries
        share the same provider/model but use different credentials or routing.
        """
        api_key_fingerprint = ""
        if config.api_key:
            api_key_fingerprint = hashlib.sha256(config.api_key.encode("utf-8")).hexdigest()[:10]

        return "|".join([
            f"provider={config.provider}",
            f"model={config.model}",
            f"priority={config.priority}",
            f"host={config.host or ''}",
            f"tier={config.tier or ''}",
            f"level={config.min_intelligence_level}-{config.max_intelligence_level}",
            f"keyfp={api_key_fingerprint}",
            f"timeout_ms={self._http_timeout_ms or ''}",
        ])
    
    def _iter_providers(
        self,
        intelligence_level: Optional[int] = None,
    ) -> Iterator[Tuple[ProviderConfig, bool]]:
        """Iterate through providers in health-aware order.
        
        Yields healthy providers first, then unhealthy ones as last resort.
        
        Args:
            intelligence_level: Optional filter by intelligence level
            
        Yields:
            Tuple of (ProviderConfig, is_fallback)
        """
        level = intelligence_level or self._default_intelligence_level
        
        # Get all applicable providers
        if level is not None:
            providers = self._registry.get_providers_for_level(level)
            if not providers:
                providers = self._registry.providers
        else:
            providers = self._registry.providers
        
        # Separate healthy and unhealthy
        healthy = self._health_tracker.filter_healthy(providers)
        unhealthy = [p for p in providers if p not in healthy]
        
        # Yield healthy first
        for i, provider in enumerate(healthy):
            yield (provider, i > 0)
        
        # Then unhealthy as last resort
        for provider in unhealthy:
            yield (provider, True)
    
    def chat(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        system_message: Optional[str] = None,
        use_search_grounding: bool = False,
        thinking_enabled: Optional[bool] = None,
        intelligence_level: Optional[int] = None,
        return_fallback_result: bool = False,
    ) -> Union[Dict[str, Any], FallbackResult]:
        """Send a chat message with automatic fallback on failure.
        
        This method has the same signature as `LLMClient.chat()` plus:
        - `intelligence_level`: Filter providers by intelligence level
        - `return_fallback_result`: If True, return FallbackResult with metadata
        
        Args:
            messages: Message(s) to send
            tools: Optional tools in OpenAI format
            structured_output: Optional Pydantic model for structured output
            system_message: Optional system message
            use_search_grounding: Enable search grounding (provider-specific)
            thinking_enabled: Enable thinking mode (provider-specific)
            intelligence_level: Filter providers by intelligence level
            return_fallback_result: Return FallbackResult instead of dict
            
        Returns:
            Standard LLMClient response dict, or FallbackResult if requested
            
        Raises:
            RuntimeError: If all providers fail
        """
        last_error: Optional[Exception] = None
        attempts = 0
        providers_tried: List[str] = []
        
        level = intelligence_level or self._default_intelligence_level
        
        # Log provider selection context
        if level is not None:
            eligible_providers = self._registry.get_providers_for_level(level)
            if eligible_providers:
                eligible_details = []
                for p in sorted(eligible_providers, key=lambda x: x.priority):
                    tier = f" [{p.tier}]" if hasattr(p, 'tier') and p.tier else ""
                    eligible_details.append(
                        f"{p.provider}:{p.model}{tier} #{p.priority}"
                    )
                logger.debug(
                    f"IQ{level}: Eligible providers: "
                    + ", ".join(eligible_details)
                )
            else:
                logger.warning(
                    f"No providers match IQ{level}. Using all {len(self._registry.providers)} providers."
                )
        else:
            logger.debug(f"No IQ specified. Using all {len(self._registry.providers)} providers.")
        
        for config, is_fallback in self._iter_providers(level):
            provider_id = f"{config.provider}:{config.model}"
            
            # Try this provider with retries
            for retry in range(self._max_retries):
                attempts += 1
                
                # Log provider selection reasoning
                if retry == 0:
                    tier_info = f" [{config.tier}]" if hasattr(config, 'tier') and config.tier else ""
                    level_match = ""
                    if level is not None:
                        if config.min_intelligence_level is not None and config.max_intelligence_level is not None:
                            level_match = f" (matches IQ{level}: range {config.min_intelligence_level}-{config.max_intelligence_level})"
                        else:
                            level_match = f" (no IQ restriction)"
                    
                    status = "fallback" if is_fallback else "primary"
                    logger.debug(
                        f"Trying {status}: {provider_id}{tier_info} #{config.priority}{level_match}"
                    )
                
                try:
                    client = self._get_client(config)
                    start_time = time.time()
                    
                    # Set intelligence level in context for usage logging
                    if level is not None:
                        set_intelligence_level(level)
                    
                    response = client.chat(
                        messages=messages,
                        tools=tools,
                        structured_output=structured_output,
                        system_message=system_message,
                        use_search_grounding=use_search_grounding,
                        thinking_enabled=thinking_enabled,
                    )
                    
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Check for error in response
                    if response.get("error"):
                        raise RuntimeError(response["error"])
                    
                    # Success! Mark healthy and record metadata
                    self._health_tracker.mark_healthy(config.provider, config.model)
                    
                    self.last_used_provider = config.provider
                    self.last_used_model = config.model
                    self.last_was_fallback = is_fallback
                    self.last_attempts = attempts
                    
                    tier_info = f" [{config.tier}]" if hasattr(config, 'tier') and config.tier else ""
                    level_info = f" IQ{level}" if level is not None else ""
                    
                    if is_fallback:
                        logger.info(
                            f"✓ Fallback{level_info}: {provider_id}{tier_info} #{config.priority} "
                            f"succeeded in {elapsed_ms:.0f}ms (attempt {attempts}, "
                            f"tried: {', '.join(providers_tried)})"
                        )
                    else:
                        logger.info(
                            f"✓ Selected{level_info}: {provider_id}{tier_info} #{config.priority} "
                            f"({elapsed_ms:.0f}ms)"
                        )
                    
                    if return_fallback_result:
                        return FallbackResult(
                            content=response.get("content"),
                            provider=config.provider,
                            model=config.model,
                            was_fallback=is_fallback,
                            attempts=attempts,
                            usage=response.get("usage", {}),
                            tool_calls=response.get("tool_calls"),
                            structured=response.get("structured", False),
                        )
                    
                    # Add fallback metadata to response
                    response["_fallback_metadata"] = {
                        "provider": config.provider,
                        "model": config.model,
                        "was_fallback": is_fallback,
                        "attempts": attempts,
                    }
                    return response
                    
                except Exception as e:
                    last_error = e
                    error_reason = classify_error(e)

                    wake_on_lan_cfg = getattr(config, "wake_on_lan", None)
                    wol_enabled = isinstance(wake_on_lan_cfg, dict) and bool(
                        wake_on_lan_cfg.get("enabled", True)
                    )
                    is_expected_wol_connection_error = (
                        error_reason == "connection_error"
                        and config.provider == "ollama"
                        and wol_enabled
                    )

                    log_message = (
                        f"Provider {provider_id} failed (attempt {retry + 1}/{self._max_retries}): "
                        f"{type(e).__name__}: {e} (classified as: {error_reason})"
                    )

                    if is_expected_wol_connection_error:
                        logger.info(log_message)
                    else:
                        logger.warning(log_message)
                    
                    # Mark unhealthy on last retry
                    if retry == self._max_retries - 1:
                        self._health_tracker.mark_unhealthy(
                            config.provider, config.model, reason=error_reason
                        )
                        providers_tried.append(provider_id)
        
        # All providers failed
        self.last_used_provider = None
        self.last_used_model = None
        self.last_was_fallback = True
        self.last_attempts = attempts
        
        error_msg = (
            f"All {len(providers_tried)} LLM providers failed. "
            f"Tried: {', '.join(providers_tried)}. Last error: {last_error}"
        )
        logger.error(error_msg)
        
        if return_fallback_result:
            return FallbackResult(
                content=None,
                provider="",
                model="",
                was_fallback=True,
                attempts=attempts,
                usage={},
                error=error_msg,
            )
        
        raise RuntimeError(error_msg)
    
    def get_provider_status(self) -> List[Dict[str, Any]]:
        """Get health status for all configured providers.
        
        Returns:
            List of provider status dictionaries
        """
        status_list = []
        for config in self._registry.providers:
            health = self._health_tracker.get_status(config.provider, config.model)
            status_list.append({
                "provider": config.provider,
                "model": config.model,
                "priority": config.priority,
                "tier": config.tier,
                "intelligence_range": (config.min_intelligence_level, config.max_intelligence_level),
                "is_healthy": health.is_healthy,
                "failure_reason": health.failure_reason,
                "recovery_at": health.recovery_at.isoformat() if health.recovery_at else None,
            })
        return status_list
    
    def mark_provider_healthy(self, provider: str, model: str) -> None:
        """Manually mark a provider as healthy."""
        self._health_tracker.mark_healthy(provider, model)
    
    def mark_provider_unhealthy(
        self,
        provider: str,
        model: str,
        reason: str = "manual",
    ) -> None:
        """Manually mark a provider as unhealthy."""
        self._health_tracker.mark_unhealthy(provider, model, reason=reason)
    
    def close(self) -> None:
        """Close all cached clients."""
        for client in self._client_cache.values():
            try:
                client.close()
            except Exception:
                pass
        self._client_cache.clear()
    
    def __enter__(self) -> "FallbackLLMClient":
        return self
    
    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False
    
    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def from_registry(
        cls,
        registry: ProviderRegistry,
        intelligence_level: Optional[int] = None,
        max_retries: int = 1,
        http_timeout_ms: Optional[int] = None,
    ) -> "FallbackLLMClient":
        """Create from an existing ProviderRegistry.
        
        Args:
            registry: Configured ProviderRegistry
            intelligence_level: Default intelligence level filter
            max_retries: Retries per provider
            http_timeout_ms: Optional HTTP timeout override (ms) for Google GenAI providers.
            
        Returns:
            FallbackLLMClient instance
        """
        return cls(
            registry=registry,
            intelligence_level=intelligence_level,
            max_retries_per_provider=max_retries,
            http_timeout_ms=http_timeout_ms,
        )
    
    @classmethod
    def from_env(
        cls,
        env_var: str = "LLM_PROVIDERS",
        file_env_var: str = "LLM_PROVIDERS_FILE",
        intelligence_level: Optional[int] = None,
        max_retries: int = 1,
        http_timeout_ms: Optional[int] = None,
    ) -> "FallbackLLMClient":
        """Create from environment configuration.
        
        Loads provider configuration from:
        1. LLM_PROVIDERS_FILE (YAML/JSON file path)
        2. LLM_PROVIDERS (JSON array in env var)
        3. Individual env vars (GEMINI_API_KEY, etc.)
        
        Args:
            env_var: Environment variable with JSON config
            file_env_var: Environment variable with config file path
            intelligence_level: Default intelligence level filter
            max_retries: Retries per provider
            http_timeout_ms: Optional HTTP timeout override (ms) for Google GenAI providers.
            
        Returns:
            FallbackLLMClient instance
        """
        registry = ProviderRegistry.from_env(env_var=env_var, file_env_var=file_env_var)
        return cls.from_registry(
            registry=registry,
            intelligence_level=intelligence_level,
            max_retries=max_retries,
            http_timeout_ms=http_timeout_ms,
        )
    
    @classmethod
    def from_config(
        cls,
        providers: List[Dict[str, Any]],
        intelligence_level: Optional[int] = None,
        max_retries: int = 1,
        http_timeout_ms: Optional[int] = None,
    ) -> "FallbackLLMClient":
        """Create from a list of provider configuration dictionaries.
        
        Args:
            providers: List of provider config dicts
            intelligence_level: Default intelligence level filter
            max_retries: Retries per provider
            http_timeout_ms: Optional HTTP timeout override (ms) for Google GenAI providers.
            
        Returns:
            FallbackLLMClient instance
            
        Example:
            ```python
            client = FallbackLLMClient.from_config([
                {"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash", "priority": 1},
                {"provider": "openai", "api_key": "...", "model": "gpt-4o-mini", "priority": 2},
            ])
            ```
        """
        registry = ProviderRegistry.from_list(providers)
        return cls.from_registry(
            registry=registry,
            intelligence_level=intelligence_level,
            max_retries=max_retries,
            http_timeout_ms=http_timeout_ms,
        )


# Convenience factory function
def create_fallback_llm_client(
    providers: Optional[List[Dict[str, Any]]] = None,
    intelligence_level: Optional[int] = None,
    max_retries: int = 1,
    http_timeout_ms: Optional[int] = None,
) -> FallbackLLMClient:
    """Create a FallbackLLMClient from config or environment.
    
    If `providers` is given, uses that configuration.
    Otherwise, loads from environment (LLM_PROVIDERS_FILE or LLM_PROVIDERS).
    
    Args:
        providers: Optional list of provider config dicts
        intelligence_level: Default intelligence level filter
        max_retries: Retries per provider before fallback
        
    Returns:
        Configured FallbackLLMClient
        
    Example:
        ```python
        # From environment
        client = create_fallback_llm_client()
        
        # With explicit config
        client = create_fallback_llm_client([
            {"provider": "gemini", "api_key": "..."},
            {"provider": "ollama", "host": "http://localhost:11434"},
        ])
        ```
    """
    if providers:
        return FallbackLLMClient.from_config(
            providers=providers,
            intelligence_level=intelligence_level,
            max_retries=max_retries,
            http_timeout_ms=http_timeout_ms,
        )
    return FallbackLLMClient.from_env(
        intelligence_level=intelligence_level,
        max_retries=max_retries,
        http_timeout_ms=http_timeout_ms,
    )
