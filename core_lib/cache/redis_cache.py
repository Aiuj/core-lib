import redis
from typing import Any, Optional, Iterable, Union
import logging
from .base_cache import BaseCache
from .redis_config import RedisConfig


class RedisCache(BaseCache):
    """Redis-specific cache implementation with connection pooling"""
    
    def __init__(self, name: str = '', config: Optional[RedisConfig] = None, ttl: Optional[int] = None, time_out: Optional[int] = None):
        config = config or RedisConfig.from_env()
        super().__init__(name, config, ttl, time_out)
        self.config: RedisConfig = config
        self.client: Union[redis.Redis, bool, None] = None
        self._pool_kwargs = {
            'host': self.config.host,
            'port': self.config.port,
            'db': self.config.db,
            'decode_responses': True,
            'socket_connect_timeout': self.time_out,
            'socket_timeout': self.time_out,
            'max_connections': self.config.max_connections,
            'retry_on_timeout': self.config.retry_on_timeout
        }
        if self.config.password:
            self._pool_kwargs['password'] = self.config.password
        self._connection_pool = None

    def _create_connection_pool(self) -> redis.ConnectionPool:
        """Create Redis connection pool"""
        return redis.ConnectionPool(**self._pool_kwargs)

    def _create_client(self) -> redis.Redis:
        """Create Redis client using connection pool"""
        if self._connection_pool is None:
            self._connection_pool = self._create_connection_pool()
        return redis.Redis(connection_pool=self._connection_pool)

    def _get_redis_client(self) -> Optional[redis.Redis]:
        """Get the Redis client if connected, otherwise return None."""
        # Avoid strict isinstance to support mocked clients in tests.
        if self.client is False or self.client is None:
            return None
        return self.client  # type: ignore[return-value]

    def connect(self):
        """Establish connection to Redis server"""
        try:
            self.client = self._create_client()
            redis_client = self._get_redis_client()
            if redis_client:
                try:
                    result = redis_client.ping()
                    if result:
                        self.connected = True
                    else:
                        self.connected = False
                        self.client = False
                        logging.error("[RedisCache] Redis ping failed.")
                except Exception as e:
                    logging.error(f"[RedisCache] Could not connect to Redis server: {e}")
                    self.client = False
                    self.connected = False
            else:
                self.client = False
                self.connected = False
        except Exception as e:
            logging.error(f"[RedisCache] Could not connect to Redis server: {e}")
            self.client = False
            self.connected = False

    def _registry_set_key(self, company_id: Optional[str]) -> str:
        """Return the registry set key used to track stored keys for clearing.

        Pattern:
          <prefix>registry:global    (no company_id)
          <prefix>registry:tenant:<company_id>
        """
        base_prefix = self.config.prefix if self.config else "cache:"
        if company_id:
            return f"{base_prefix}registry:tenant:{company_id}"
        return f"{base_prefix}registry:global"

    def _track_key(self, cache_key: str, company_id: Optional[str]):
        try:
            redis_client = self._get_redis_client()
            if redis_client:
                registry_key = self._registry_set_key(company_id)
                # Use sadd so duplicates are ignored
                redis_client.sadd(registry_key, cache_key)
        except Exception:
            # Never fail core caching due to registry tracking
            pass

    def _delete_keys(self, keys: Iterable[str]):
        try:
            if not keys:
                return
            redis_client = self._get_redis_client()
            if redis_client:
                redis_client.delete(*list(keys))
        except Exception:
            pass

    def get(self, input_data: Any, company_id: Optional[str] = None) -> Optional[Any]:
        """Retrieve cached data for the given input"""
        redis_client = self._get_redis_client()
        if not redis_client:
            return None
        key = self._make_key(input_data, company_id=company_id)
        value = redis_client.get(key)
        if value is not None:
            # Ensure value is a string before deserializing
            if isinstance(value, str):
                return self._deserialize_data(value)
        return None

    def get_many(self, input_items: list[Any], company_id: Optional[str] = None) -> list[Optional[Any]]:
        """Retrieve values with one Redis round trip."""
        redis_client = self._get_redis_client()
        if not redis_client or not input_items:
            return [None] * len(input_items)
        if not callable(getattr(redis_client, "mget", None)):
            return super().get_many(input_items, company_id=company_id)
        keys = [self._make_key(item, company_id=company_id) for item in input_items]
        values = redis_client.mget(keys)
        return [
            self._deserialize_data(value) if isinstance(value, str) else None
            for value in values
        ]

    def set(self, input_data: Any, output_data: Any, ttl: Optional[int] = None, company_id: Optional[str] = None):
        """Store data in cache with optional TTL and tenant isolation"""
        redis_client = self._get_redis_client()
        if not redis_client:
            return
        key = self._make_key(input_data, company_id=company_id)
        value = self._serialize_data(output_data)
        ttl_to_use = ttl if ttl is not None else self.ttl
        redis_client.setex(key, ttl_to_use, value)
        # Debug: log TTL set
        logging.debug(f"[RedisCache] Set key {key} with TTL {ttl_to_use}")
        # For some Redis test/mocks, force expire as fallback
        try:
            redis_client.expire(key, ttl_to_use)
        except Exception:
            pass
        # Track key for potential tenant/global clearing
        self._track_key(key, company_id)

    def set_many(
        self,
        items: list[tuple[Any, Any]],
        ttl: Optional[int] = None,
        company_id: Optional[str] = None,
    ) -> None:
        """Store values and their registry membership in one pipeline."""
        redis_client = self._get_redis_client()
        if not redis_client or not items:
            return
        if not callable(getattr(redis_client, "pipeline", None)):
            return super().set_many(items, ttl=ttl, company_id=company_id)
        ttl_to_use = ttl if ttl is not None else self.ttl
        keys = [self._make_key(item, company_id=company_id) for item, _ in items]
        pipeline = redis_client.pipeline(transaction=False)
        for key, (_, output_data) in zip(keys, items):
            pipeline.setex(key, ttl_to_use, self._serialize_data(output_data))
        pipeline.sadd(self._registry_set_key(company_id), *keys)
        pipeline.execute()

    def delete(self, input_data: Any, company_id: Optional[str] = None) -> bool:
        """Delete cached data for the given input and optional company_id.
        
        Returns:
            True if key was deleted, False if key didn't exist
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return False
        key = self._make_key(input_data, company_id=company_id)
        try:
            result = redis_client.delete(key)
            return bool(result)
        except Exception as e:
            logging.error(f"[RedisCache] DELETE failed for key {key}: {e}")
            return False

    def exists(self, input_data: Any, company_id: Optional[str] = None) -> bool:
        """Check if cached data exists for the given input and optional company_id.
        
        Returns:
            True if key exists, False otherwise
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return False
        key = self._make_key(input_data, company_id=company_id)
        try:
            return bool(redis_client.exists(key))
        except Exception as e:
            logging.error(f"[RedisCache] EXISTS failed for key {key}: {e}")
            return False

    # --- Clearing APIs ---
    def clear_company(self, company_id: str):
        redis_client = self._get_redis_client()
        if not redis_client:
            return
        registry_key = self._registry_set_key(company_id)
        try:
            keys = redis_client.smembers(registry_key)
            if keys:
                # Type cast for decode_responses=True behavior
                key_list = list(keys)  # type: ignore
                self._delete_keys(key_list)
            redis_client.delete(registry_key)
        except Exception:
            pass

    def clear_global(self):
        redis_client = self._get_redis_client()
        if not redis_client:
            return
        registry_key = self._registry_set_key(None)
        try:
            keys = redis_client.smembers(registry_key)
            if keys:
                # Type cast for decode_responses=True behavior
                key_list = list(keys)  # type: ignore
                self._delete_keys(key_list)
            redis_client.delete(registry_key)
        except Exception:
            pass

    def clear_all(self):
        redis_client = self._get_redis_client()
        if not redis_client:
            return
        # Naive approach: scan for registry keys and clear referenced keys.
        try:
            # Pattern for registry keys
            base_prefix = self.config.prefix if self.config else "cache:"
            pattern = f"{base_prefix}registry:*"
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=100)  # type: ignore
                for reg_key in keys:
                    try:
                        member_keys = redis_client.smembers(reg_key)
                        if member_keys:
                            # Type cast for decode_responses=True behavior
                            key_list = list(member_keys)  # type: ignore
                            self._delete_keys(key_list)
                        redis_client.delete(reg_key)
                    except Exception:
                        continue
                if cursor == 0:
                    break
        except Exception:
            pass

    def health_check(self) -> bool:
        """Check if Redis server is healthy and responding"""
        redis_client = self._get_redis_client()
        if not redis_client or not self.connected:
            return False
        try:
            result = redis_client.ping()
            return bool(result)
        except Exception as e:
            logging.error(f"[RedisCache] Health check failed: {e}")
            return False

    def close(self):
        """Close connection pool and cleanup resources"""
        if self._connection_pool:
            try:
                self._connection_pool.disconnect()
                logging.info("[RedisCache] Connection pool closed")
            except Exception as e:
                logging.warning(f"[RedisCache] Error closing connection pool: {e}")
            finally:
                self._connection_pool = None
                self.connected = False
                self.client = False
