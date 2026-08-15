from abc import ABC, abstractmethod
import json
import hashlib
from typing import Any, Optional
import logging
from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Base configuration class for cache providers"""
    host: str
    port: int
    db: int
    prefix: str
    ttl: int = 3600
    password: Optional[str] = None
    time_out: int = 4
    # Connection pool settings
    max_connections: int = 50
    retry_on_timeout: bool = True


class BaseCache(ABC):
    """Abstract base class for cache providers"""
    
    def __init__(self, name: str = '', config: Optional[CacheConfig] = None, ttl: Optional[int] = None, time_out: Optional[int] = None):
        self.name = name or ''
        self.config = config
        self.ttl = ttl or (config.ttl if config else 3600)
        self.client = None
        self.connected = False
        self.time_out = time_out or (config.time_out if config else 4)

    @abstractmethod
    def connect(self):
        """Establish connection to the cache server"""
        pass

    @abstractmethod
    def _create_client(self) -> Any:
        """Create and return the client instance for the specific provider"""
        pass

    def _make_key(self, input_data: Any, company_id: Optional[str] = None) -> str:
        """Generate a cache key from input data with optional tenant isolation.

        Key patterns:
          Global (no tenant): <prefix><name>:<hash>
          Tenant:            <prefix>tenant:<company_id>:<name>:<hash>

        A hard wall between tenants is achieved by including the company_id
        early in the key path. The prefix always ends with a colon (by config convention).
        """
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        hash_key = hashlib.sha256(input_str.encode('utf-8')).hexdigest()
        prefix = self.config.prefix if self.config else "cache:"
        if company_id:
            return f"{prefix}tenant:{company_id}:{self.name}:{hash_key}"
        return f"{prefix}{self.name}:{hash_key}"

    @abstractmethod
    def get(self, input_data: Any, company_id: Optional[str] = None) -> Optional[Any]:
        """Retrieve cached data for the given input and optional company_id"""
        pass

    @abstractmethod
    def set(self, input_data: Any, output_data: Any, ttl: Optional[int] = None, company_id: Optional[str] = None):
        """Store data in cache with optional TTL and optional company_id"""
        pass

    def get_many(self, input_items: list[Any], company_id: Optional[str] = None) -> list[Optional[Any]]:
        """Retrieve several values, preserving input order."""
        return [self.get(item, company_id=company_id) for item in input_items]

    def set_many(
        self,
        items: list[tuple[Any, Any]],
        ttl: Optional[int] = None,
        company_id: Optional[str] = None,
    ) -> None:
        """Store several values using the backend's best available strategy."""
        for input_data, output_data in items:
            self.set(input_data, output_data, ttl=ttl, company_id=company_id)

    @abstractmethod
    def delete(self, input_data: Any, company_id: Optional[str] = None) -> bool:
        """Delete cached data for the given input and optional company_id.
        
        Returns:
            True if key was deleted, False if key didn't exist
        """
        pass

    @abstractmethod
    def exists(self, input_data: Any, company_id: Optional[str] = None) -> bool:
        """Check if cached data exists for the given input and optional company_id.
        
        Returns:
            True if key exists, False otherwise
        """
        pass

    def _serialize_data(self, data: Any) -> str:
        """Serialize data for storage"""
        return json.dumps(data, default=str)

    def _deserialize_data(self, data: str) -> Any:
        """Deserialize data from storage"""
        return json.loads(data)
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the cache server is healthy and responding"""
        pass

    def get_client(self) -> Any:
        """Get direct access to the underlying cache client.
        
        Returns:
            The underlying cache client instance (e.g., Redis client)
        """
        return self.client

    def close(self):
        """Close connections and cleanup resources - override in subclasses if needed"""
        self.connected = False
        self.client = None
