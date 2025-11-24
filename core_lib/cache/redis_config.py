import os
from dataclasses import dataclass
from typing import Optional
from .base_cache import CacheConfig

@dataclass
class RedisConfig(CacheConfig):
    """Redis-specific configuration"""
    pass

    @classmethod
    def from_env(cls) -> "RedisConfig":
        def _get_int(key: str, default: str) -> int:
            """Get integer env var, stripping inline comments."""
            value = os.getenv(key, default)
            # Strip inline comments (e.g., "3600 # comment")
            clean_value = value.split('#')[0].strip()
            return int(clean_value)
        
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=_get_int("REDIS_PORT", "6379"),
            db=_get_int("REDIS_DB", "0"),
            prefix=os.getenv("REDIS_PREFIX", "cache:"),
            ttl=_get_int("REDIS_CACHE_TTL", "3600"),  # Default TTL of 1 hour
            password=os.getenv("REDIS_PASSWORD", None),
            time_out=_get_int("REDIS_TIMEOUT", "4"),  # Default timeout of 4 seconds
            max_connections=_get_int("REDIS_MAX_CONNECTIONS", "50"),
            retry_on_timeout=os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
        )
