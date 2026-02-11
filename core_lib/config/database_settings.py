"""Database Configuration Settings.

This module contains configuration classes for database providers
including PostgreSQL with connection string generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .base_settings import BaseSettings, SettingsError, EnvParser


@dataclass(frozen=True)
class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration settings."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "doc-qa"
    username: str = "rfp_user"
    password: str = "rfp_password"
    sslmode: str = "disable"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    connect_timeout: int = 5
    
    @classmethod
    def from_env(
        cls,
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides
    ) -> "DatabaseSettings":
        """Create database settings from environment variables."""
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        settings_dict = {
            "host": EnvParser.get_env("POSTGRES_HOST", "DATABASE_HOST", default="localhost"),
            "port": EnvParser.get_env("POSTGRES_PORT", "DATABASE_PORT", default=5432, env_type=int),
            "database": EnvParser.get_env("POSTGRES_DB", "DATABASE_NAME", default="app-database"),
            "username": EnvParser.get_env("POSTGRES_USER", "DATABASE_USER", default="app_user"),
            "password": EnvParser.get_env("POSTGRES_PASSWORD", "DATABASE_PASSWORD", default="app_password"),
            "sslmode": EnvParser.get_env("POSTGRES_SSLMODE", "DATABASE_SSLMODE", default="disable"),
            "pool_size": EnvParser.get_env("POSTGRES_POOL_SIZE", "DATABASE_POOL_SIZE", default=10, env_type=int),
            "max_overflow": EnvParser.get_env("POSTGRES_MAX_OVERFLOW", "DATABASE_MAX_OVERFLOW", default=20, env_type=int),
            "pool_timeout": EnvParser.get_env("POSTGRES_POOL_TIMEOUT", "DATABASE_POOL_TIMEOUT", default=30, env_type=int),
            "connect_timeout": EnvParser.get_env("POSTGRES_CONNECT_TIMEOUT", "DATABASE_CONNECT_TIMEOUT", default=5, env_type=int),
        }
        
        settings_dict.update(overrides)
        return cls(**settings_dict)
    
    def validate(self) -> None:
        """Validate database configuration."""
        if self.port <= 0 or self.port > 65535:
            raise SettingsError("Database port must be between 1 and 65535")
        if not self.database:
            raise SettingsError("Database name is required")
        if not self.username:
            raise SettingsError("Database username is required")
        if self.pool_size <= 0:
            raise SettingsError("Pool size must be positive")
        if self.max_overflow < 0:
            raise SettingsError("Max overflow must be non-negative")
        if self.pool_timeout <= 0:
            raise SettingsError("Pool timeout must be positive")
        if self.connect_timeout <= 0:
            raise SettingsError("Connect timeout must be positive")
        if self.sslmode not in ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]:
            raise SettingsError("Invalid SSL mode. Must be one of: disable, allow, prefer, require, verify-ca, verify-full")
    
    def get_connection_string(self, driver: str = "postgresql") -> str:
        """Generate database connection string.
        
        Args:
            driver: Database driver (postgresql, postgresql+psycopg2, postgresql+asyncpg, etc.)
            
        Returns:
            Database connection string
        """
        return (
            f"{driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.sslmode}&connect_timeout={self.connect_timeout}"
        )
    
    def get_async_connection_string(self) -> str:
        """Generate async database connection string for asyncpg."""
        return self.get_connection_string("postgresql+asyncpg")
    
    def get_sync_connection_string(self) -> str:
        """Generate sync database connection string for psycopg2."""
        return self.get_connection_string("postgresql+psycopg2")
    
    def as_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": self.password,
            "sslmode": self.sslmode,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "connect_timeout": self.connect_timeout,
        }