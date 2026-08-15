try:
    import valkey
except ImportError:
    valkey = None
from typing import Any, Optional, Iterable
import logging
from .base_cache import BaseCache
from .valkey_config import ValkeyConfig


class ValkeyCache(BaseCache):
    """Valkey-specific cache implementation with connection pooling"""
    
    def __init__(self, name: str = '', config: Optional[ValkeyConfig] = None, ttl: Optional[int] = None, time_out: Optional[int] = None):
        config = config or ValkeyConfig.from_env()
        super().__init__(name, config, ttl, time_out)
        self.config: ValkeyConfig = config
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

    def _create_connection_pool(self) -> Any:
        """Create Valkey connection pool"""
        if valkey is None:
            raise ImportError("valkey package is not installed")
        return valkey.ConnectionPool(**self._pool_kwargs)

    def _create_client(self) -> Any:
        """Create Valkey client using connection pool"""
        if valkey is None:
            raise ImportError("valkey package is not installed")
        if self._connection_pool is None:
            self._connection_pool = self._create_connection_pool()
        return valkey.Valkey(connection_pool=self._connection_pool)

    def connect(self):
        """Establish connection to Valkey server"""
        try:
            self.client = self._create_client()
            try:
                result = self.client.ping()
                if result:
                    self.connected = True
                else:
                    self.connected = False
                    self.client = False
                    logging.error("[ValkeyCache] Valkey ping failed.")
            except Exception as e:
                logging.error(f"[ValkeyCache] Could not connect to Valkey server: {e}")
                self.client = False
                self.connected = False
        except Exception as e:
            logging.error(f"[ValkeyCache] Could not connect to Valkey server: {e}")
            self.client = False
            self.connected = False

    def _registry_set_key(self, company_id: Optional[str]) -> str:
        base_prefix = self.config.prefix if self.config else "cache:"
        if company_id:
            return f"{base_prefix}registry:tenant:{company_id}"
        return f"{base_prefix}registry:global"

    def _track_key(self, cache_key: str, company_id: Optional[str]):
        try:
            registry_key = self._registry_set_key(company_id)
            self.client.sadd(registry_key, cache_key)
        except Exception:
            pass

    def _delete_keys(self, keys: Iterable[str]):
        try:
            if not keys:
                return
            self.client.delete(*list(keys))
        except Exception:
            pass

    def get(self, input_data: Any, company_id: Optional[str] = None) -> Optional[Any]:
        """Retrieve cached data for the given input"""
        if self.client is False:
            return None
        key = self._make_key(input_data, company_id=company_id)
        value = self.client.get(key)
        if value is not None:
            return self._deserialize_data(value)
        return None

    def get_many(self, input_items: list[Any], company_id: Optional[str] = None) -> list[Optional[Any]]:
        """Retrieve values with one Valkey round trip."""
        if self.client is False or self.client is None or not input_items:
            return [None] * len(input_items)
        if not callable(getattr(self.client, "mget", None)):
            return super().get_many(input_items, company_id=company_id)
        keys = [self._make_key(item, company_id=company_id) for item in input_items]
        values = self.client.mget(keys)
        return [
            self._deserialize_data(value) if isinstance(value, str) else None
            for value in values
        ]

    def set(self, input_data: Any, output_data: Any, ttl: Optional[int] = None, company_id: Optional[str] = None):
        """Store data in cache with optional TTL"""
        if self.client is False:
            return
        key = self._make_key(input_data, company_id=company_id)
        value = self._serialize_data(output_data)
        ttl_to_use = ttl if ttl is not None else self.ttl
        self.client.setex(key, ttl_to_use, value)
        # Debug: log TTL set
        logging.debug(f"[ValkeyCache] Set key {key} with TTL {ttl_to_use}")
        # For some Valkey test/mocks, force expire as fallback
        try:
            self.client.expire(key, ttl_to_use)
        except Exception:
            pass
        self._track_key(key, company_id)

    def set_many(
        self,
        items: list[tuple[Any, Any]],
        ttl: Optional[int] = None,
        company_id: Optional[str] = None,
    ) -> None:
        """Store values and their registry membership in one pipeline."""
        if self.client is False or self.client is None or not items:
            return
        if not callable(getattr(self.client, "pipeline", None)):
            return super().set_many(items, ttl=ttl, company_id=company_id)
        ttl_to_use = ttl if ttl is not None else self.ttl
        keys = [self._make_key(item, company_id=company_id) for item, _ in items]
        pipeline = self.client.pipeline(transaction=False)
        for key, (_, output_data) in zip(keys, items):
            pipeline.setex(key, ttl_to_use, self._serialize_data(output_data))
        pipeline.sadd(self._registry_set_key(company_id), *keys)
        pipeline.execute()

    def delete(self, input_data: Any, company_id: Optional[str] = None) -> bool:
        """Delete cached data for the given input and optional company_id.
        
        Returns:
            True if key was deleted, False if key didn't exist
        """
        if self.client is False:
            return False
        key = self._make_key(input_data, company_id=company_id)
        try:
            result = self.client.delete(key)
            return bool(result)
        except Exception as e:
            logging.error(f"[ValkeyCache] DELETE failed for key {key}: {e}")
            return False

    def exists(self, input_data: Any, company_id: Optional[str] = None) -> bool:
        """Check if cached data exists for the given input and optional company_id.
        
        Returns:
            True if key exists, False otherwise
        """
        if self.client is False:
            return False
        key = self._make_key(input_data, company_id=company_id)
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logging.error(f"[ValkeyCache] EXISTS failed for key {key}: {e}")
            return False

    def clear_company(self, company_id: str):
        if self.client is False:
            return
        registry_key = self._registry_set_key(company_id)
        try:
            keys = self.client.smembers(registry_key)
            self._delete_keys(keys)
            self.client.delete(registry_key)
        except Exception:
            pass

    def clear_global(self):
        if self.client is False:
            return
        registry_key = self._registry_set_key(None)
        try:
            keys = self.client.smembers(registry_key)
            self._delete_keys(keys)
            self.client.delete(registry_key)
        except Exception:
            pass

    def clear_all(self):
        if self.client is False:
            return
        try:
            base_prefix = self.config.prefix if self.config else "cache:"
            pattern = f"{base_prefix}registry:*"
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=100)
                for reg_key in keys:
                    try:
                        member_keys = self.client.smembers(reg_key)
                        self._delete_keys(member_keys)
                        self.client.delete(reg_key)
                    except Exception:
                        continue
                if cursor == 0:
                    break
        except Exception:
            pass

    def health_check(self) -> bool:
        """Check if Valkey server is healthy and responding"""
        if self.client is False or not self.connected:
            return False
        try:
            result = self.client.ping()
            return bool(result)
        except Exception as e:
            logging.error(f"[ValkeyCache] Health check failed: {e}")
            return False

    def close(self):
        """Close connection pool and cleanup resources"""
        if self._connection_pool:
            try:
                self._connection_pool.disconnect()
                logging.info("[ValkeyCache] Connection pool closed")
            except Exception as e:
                logging.warning(f"[ValkeyCache] Error closing connection pool: {e}")
            finally:
                self._connection_pool = None
                self.connected = False
                self.client = False
