import os
from dataclasses import dataclass
from typing import Optional

from .redis_config import RedisConfig


@dataclass
class ValkeyConfig(RedisConfig):
    """Valkey config re-uses RedisConfig fields for compatibility.

    Valkey is a drop-in replacement so it shares host/port/db/prefix/ttl fields.
    We inherit from `RedisConfig` and provide an env-based constructor that
    reads VALKEY-specific environment variables but populates the same fields.
    """

    @classmethod
    def from_env(cls) -> "ValkeyConfig":
        def _get_int(key: str, fallback_key: str, default: str) -> int:
            """Get integer env var with fallback, stripping inline comments."""
            value = os.getenv(key, os.getenv(fallback_key, default))
            # Strip inline comments (e.g., "3600 # comment")
            clean_value = value.split('#')[0].strip()
            return int(clean_value)
        
        return cls(
            host=os.getenv("VALKEY_HOST", os.getenv("REDIS_HOST", "localhost")),
            port=_get_int("VALKEY_PORT", "REDIS_PORT", "6379"),
            db=_get_int("VALKEY_DB", "REDIS_DB", "0"),
            prefix=os.getenv("VALKEY_PREFIX", os.getenv("REDIS_PREFIX", "cache:")),
            ttl=_get_int("VALKEY_CACHE_TTL", "REDIS_CACHE_TTL", "3600"),
            password=os.getenv("VALKEY_PASSWORD", os.getenv("REDIS_PASSWORD", None)),
            time_out=_get_int("VALKEY_TIMEOUT", "REDIS_TIMEOUT", "4"),
            max_connections=_get_int("VALKEY_MAX_CONNECTIONS", "REDIS_MAX_CONNECTIONS", "50"),
            retry_on_timeout=os.getenv("VALKEY_RETRY_ON_TIMEOUT", os.getenv("REDIS_RETRY_ON_TIMEOUT", "true")).lower() == "true"
        )
