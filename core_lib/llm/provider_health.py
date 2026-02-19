"""Provider health tracking for intelligent LLM fallback.

This module provides Redis-based health tracking for LLM providers to enable
smart fallback behavior. When a provider fails (rate limit, timeout, errors),
it is marked as unhealthy for a configurable duration, and healthy alternatives
are preferred until the unhealthy period expires.

Key Features:
    - Redis-backed health status with TTL-based auto-recovery
    - Configurable unhealthy duration (default: 5 minutes)
    - Thread-safe health checks and updates
    - Graceful degradation when Redis is unavailable
    - Support for different failure types with different recovery times

Example Usage:
    from core_lib.llm.provider_health import ProviderHealthTracker
    
    tracker = ProviderHealthTracker(unhealthy_ttl=300)  # 5 minutes
    
    # Check if provider is healthy
    if tracker.is_healthy("gemini", "gemini-2.0-flash"):
        try:
            result = client.chat(messages)
            tracker.mark_healthy("gemini", "gemini-2.0-flash")
        except RateLimitError:
            tracker.mark_unhealthy("gemini", "gemini-2.0-flash", reason="rate_limit")
    
    # Get the best healthy provider from a list
    healthy_providers = tracker.filter_healthy(providers)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core_lib import get_module_logger

logger = get_module_logger()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

if TYPE_CHECKING:
    from .provider_registry import ProviderConfig

# Default TTL for different failure types (in seconds)
DEFAULT_UNHEALTHY_TTL = 300  # 5 minutes
FAILURE_TTL_MAP = {
    "rate_limit": 60,       # 1 minute for rate limits (was 300)
    "quota_exceeded": 3600, # 1 hour for quota exceeded
    "timeout": 60,          # 1 minute for timeouts (was 300)
    "server_error": 60,     # 1 minute for server errors (was 300)
    "connection_error": 60, # 1 minute for connection errors
    "auth_error": 3600,     # 1 hour for auth errors (unlikely to self-resolve)
    "unknown": 60,          # 1 minute for unknown errors (was 180)
}


@dataclass
class HealthStatus:
    """Health status for a provider."""
    is_healthy: bool
    last_check: datetime
    failure_reason: Optional[str] = None
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    recovery_at: Optional[datetime] = None


class ProviderHealthTracker:
    """Track LLM provider health status in Redis for intelligent fallback.
    
    This class manages health status for LLM providers, marking them as
    unhealthy when they fail and automatically recovering after a TTL period.
    
    Attributes:
        unhealthy_ttl: Default seconds to mark a provider as unhealthy
        cache_prefix: Redis key prefix for health status keys
        max_failure_count: Number of failures before considering unhealthy
    """
    
    def __init__(
        self,
        unhealthy_ttl: int = DEFAULT_UNHEALTHY_TTL,
        cache_prefix: str = "llm:health:",
        max_failure_count: int = 1,
        cache_client: Optional[Any] = None,
    ):
        """Initialize the health tracker.
        
        Args:
            unhealthy_ttl: Default TTL in seconds for unhealthy status
            cache_prefix: Redis key prefix for health keys
            max_failure_count: Failures needed before marking unhealthy
            cache_client: Optional cache client (uses global cache if None)
        """
        self.unhealthy_ttl = unhealthy_ttl
        self.cache_prefix = cache_prefix
        self.max_failure_count = max_failure_count
        self._cache_client = cache_client
        self._local_health: Dict[str, HealthStatus] = {}  # Fallback when no Redis
    
    def _get_cache(self) -> Optional[Any]:
        """Get the cache wrapper, initializing if needed.
        
        Returns the BaseCache wrapper (not raw client) for proper set/get with TTL support.
        """
        if self._cache_client is not None:
            return self._cache_client
        
        try:
            from core_lib.cache import get_cache
            cache = get_cache()
            # get_cache() returns False if caching is disabled/failed
            if cache and cache is not False and cache.connected:
                # Return the wrapper for proper set/get with TTL support
                return cache
        except Exception as e:
            logger.debug(f"Cache not available for health tracking: {e}")
        
        return None
    
    def _get_key(self, provider: str, model: str) -> str:
        """Generate the cache key for a provider/model combination."""
        # Sanitize provider and model names for safe key generation
        safe_provider = provider.replace(":", "_").replace(" ", "_")
        safe_model = model.replace(":", "_").replace(" ", "_").replace("/", "_")
        return f"{self.cache_prefix}{safe_provider}:{safe_model}"
    
    def _get_ttl_for_reason(self, reason: str) -> int:
        """Get the appropriate TTL for a failure reason."""
        return FAILURE_TTL_MAP.get(reason, self.unhealthy_ttl)
    
    def is_healthy(self, provider: str, model: str) -> bool:
        """Check if a provider/model is currently healthy.
        
        Args:
            provider: Provider name (e.g., "gemini", "openai")
            model: Model name (e.g., "gemini-2.0-flash")
            
        Returns:
            True if healthy or status unknown, False if marked unhealthy
        """
        cache = self._get_cache()
        key = self._get_key(provider, model)
        
        if cache:
            try:
                status = cache.get(key)
                if status:
                    # If key exists, provider is unhealthy
                    return False
                return True
            except Exception as e:
                logger.warning(f"Error checking health status: {e}")
                # Fall through to local check
        
        # Fallback to local tracking
        if key in self._local_health:
            status = self._local_health[key]
            if status.recovery_at and _utcnow() < status.recovery_at:
                return False
            else:
                # Expired, remove from local cache
                del self._local_health[key]
        
        return True
    
    def mark_unhealthy(
        self,
        provider: str,
        model: str,
        reason: str = "unknown",
        ttl: Optional[int] = None,
    ) -> None:
        """Mark a provider/model as unhealthy.
        
        Args:
            provider: Provider name
            model: Model name
            reason: Failure reason (affects recovery TTL)
            ttl: Optional override for unhealthy TTL
        """
        cache = self._get_cache()
        key = self._get_key(provider, model)
        effective_ttl = ttl or self._get_ttl_for_reason(reason)
        
        status_data = {
            "reason": reason,
            "marked_at": _utcnow().isoformat(),
            "recovery_at": (_utcnow().timestamp() + effective_ttl),
        }
        
        if cache:
            try:
                cache.set(key, status_data, ttl=effective_ttl)
                logger.info(
                    f"Marked provider {provider}:{model} unhealthy for {effective_ttl}s "
                    f"(reason: {reason})"
                )
                return
            except Exception as e:
                logger.warning(f"Error setting unhealthy status in cache: {e}")
        
        # Fallback to local tracking
        from datetime import timedelta
        self._local_health[key] = HealthStatus(
            is_healthy=False,
            last_check=_utcnow(),
            failure_reason=reason,
            failure_count=1,
            last_failure=_utcnow(),
            recovery_at=_utcnow() + timedelta(seconds=effective_ttl),
        )
        logger.info(
            f"Marked provider {provider}:{model} unhealthy locally for {effective_ttl}s "
            f"(reason: {reason})"
        )
    
    def mark_healthy(self, provider: str, model: str) -> None:
        """Mark a provider/model as healthy (clear unhealthy status).
        
        Args:
            provider: Provider name
            model: Model name
        """
        cache = self._get_cache()
        key = self._get_key(provider, model)
        
        if cache:
            try:
                cache.delete(key)
                logger.debug(f"Marked provider {provider}:{model} as healthy")
            except Exception as e:
                logger.warning(f"Error clearing health status: {e}")
        
        # Also clear local tracking
        if key in self._local_health:
            del self._local_health[key]
    
    def get_status(self, provider: str, model: str) -> HealthStatus:
        """Get detailed health status for a provider/model.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            HealthStatus with current state
        """
        cache = self._get_cache()
        key = self._get_key(provider, model)
        
        if cache:
            try:
                status = cache.get(key)
                if status and isinstance(status, dict):
                    recovery_at = None
                    if "recovery_at" in status:
                        recovery_at = datetime.fromtimestamp(status["recovery_at"])
                    
                    return HealthStatus(
                        is_healthy=False,
                        last_check=_utcnow(),
                        failure_reason=status.get("reason"),
                        failure_count=status.get("failure_count", 1),
                        last_failure=datetime.fromisoformat(status["marked_at"]) if "marked_at" in status else None,
                        recovery_at=recovery_at,
                    )
            except Exception as e:
                logger.warning(f"Error getting health status: {e}")
        
        # Check local cache
        if key in self._local_health:
            return self._local_health[key]
        
        # Default: healthy
        return HealthStatus(
            is_healthy=True,
            last_check=_utcnow(),
        )
    
    def filter_healthy(
        self,
        providers: List["ProviderConfig"],
    ) -> List["ProviderConfig"]:
        """Filter a list of providers to only include healthy ones.
        
        Args:
            providers: List of ProviderConfig instances
            
        Returns:
            List of healthy ProviderConfig instances (preserving order)
        """
        healthy = []
        for p in providers:
            if self.is_healthy(p.provider, p.model):
                healthy.append(p)
            else:
                # Include reason and recovery time in the single log line
                reason = "unknown"
                recovery_in = ""
                try:
                    cache = self._get_cache()
                    key = self._get_key(p.provider, p.model)
                    status = cache.get(key) if cache else None
                    if status and isinstance(status, dict):
                        reason = status.get("reason", reason)
                        recovery_at = status.get("recovery_at")
                        if recovery_at:
                            import time as _time
                            secs_left = int(recovery_at - _time.time())
                            if secs_left > 0:
                                recovery_in = f", recovers in {secs_left}s"
                    elif key in self._local_health:
                        s = self._local_health[key]
                        reason = s.failure_reason or reason
                except Exception:
                    pass
                logger.debug(f"Skipping unhealthy provider {p.provider}:{p.model} (reason={reason}{recovery_in})")

        return healthy
    
    def get_first_healthy(
        self,
        providers: List["ProviderConfig"],
    ) -> Optional["ProviderConfig"]:
        """Get the first healthy provider from a list.
        
        Args:
            providers: List of ProviderConfig instances (should be priority-sorted)
            
        Returns:
            First healthy ProviderConfig or None if all unhealthy
        """
        healthy = self.filter_healthy(providers)
        return healthy[0] if healthy else None
    
    def clear_all(self) -> None:
        """Clear all health status (useful for testing)."""
        self._local_health.clear()
        
        cache = self._get_cache()
        if cache:
            try:
                # This would require pattern-based deletion which may not be available
                logger.debug("Health status clear requested (Redis pattern delete not implemented)")
            except Exception:
                pass


# Global health tracker instance
_health_tracker: Optional[ProviderHealthTracker] = None


def get_health_tracker(
    unhealthy_ttl: int = DEFAULT_UNHEALTHY_TTL,
) -> ProviderHealthTracker:
    """Get or create the global health tracker instance.
    
    Args:
        unhealthy_ttl: Default TTL for unhealthy status
        
    Returns:
        ProviderHealthTracker instance
    """
    global _health_tracker
    
    if _health_tracker is None:
        # Check for environment configuration
        env_ttl = os.getenv("LLM_UNHEALTHY_TTL")
        if env_ttl:
            try:
                unhealthy_ttl = int(env_ttl)
            except ValueError:
                pass
        
        _health_tracker = ProviderHealthTracker(unhealthy_ttl=unhealthy_ttl)
    
    return _health_tracker


def reset_health_tracker() -> None:
    """Reset the global health tracker (for testing)."""
    global _health_tracker
    if _health_tracker:
        _health_tracker.clear_all()
    _health_tracker = None


def classify_error(error: Exception) -> str:
    """Classify an exception into a failure reason category.
    
    Args:
        error: The exception that occurred
        
    Returns:
        Failure reason string for TTL lookup
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Rate limit detection
    if any(x in error_str for x in ["rate limit", "rate_limit", "ratelimit", "429", "quota"]):
        if "quota" in error_str:
            return "quota_exceeded"
        return "rate_limit"
    
    # Timeout detection
    if any(x in error_str for x in ["timeout", "timed out", "deadline"]):
        return "timeout"
    
    if any(x in error_type for x in ["timeout", "timedout"]):
        return "timeout"

    # Server-side cancellation (Google/Vertex 499 CANCELLED — the server aborted the call,
    # not the client; treat as a brief transient server error with a short cooldown).
    if "499" in error_str or "cancelled" in error_str:
        return "server_error"

    # Connection errors
    if any(x in error_str for x in ["connection", "connect", "unreachable", "network"]):
        return "connection_error"
    
    if any(x in error_type for x in ["connection", "network", "socket"]):
        return "connection_error"
    
    # Auth errors
    if any(x in error_str for x in ["auth", "unauthorized", "forbidden", "401", "403", "api key"]):
        return "auth_error"
    
    # Server errors
    if any(x in error_str for x in ["500", "502", "503", "504", "server error", "internal"]):
        return "server_error"

    # Truncated / incomplete response (hit output token limit)
    if any(x in error_str for x in ["truncated json", "truncated response", "eof while parsing", "unterminated string"]):
        return "truncated_response"

    return "unknown"
