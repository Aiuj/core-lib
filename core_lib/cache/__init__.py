"""Cache module for core-lib."""

from .cache_manager import (
    create_cache,
    set_cache,
    get_cache,
    cache_get,
    cache_get_many,
    cache_set,
    cache_set_many,
    cache_delete,
    cache_exists,
    get_cache_client,
    cache_clear_company,
    cache_clear_global,
    cache_clear_all,
)
from .base_cache import BaseCache, CacheConfig
from .redis_cache import RedisCache
from .valkey_cache import ValkeyCache
from .redis_config import RedisConfig
from .valkey_config import ValkeyConfig

__all__ = [
    'BaseCache', 'CacheConfig',
    'RedisCache', 'ValkeyCache',
    'RedisConfig', 'ValkeyConfig',
    'create_cache', 'set_cache', 'get_cache', 'cache_get', 'cache_get_many',
    'cache_set', 'cache_set_many',
    'cache_delete', 'cache_exists', 'get_cache_client',
    'cache_clear_company', 'cache_clear_global', 'cache_clear_all'
]
