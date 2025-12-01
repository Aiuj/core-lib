"""
API Settings Implementation

API-focused settings implementation that contains the minimum configuration
needed to run an API server: cache, tracing, MCP server, and FastAPI server.

This module provides a mid-level settings class between BaseSettings and 
StandardSettings, making it easier to configure API-focused applications
without unnecessary dependencies on LLM, embeddings, or database components.

Features:
- Inherits from BaseSettings for core functionality
- Optional cache, tracing, MCP server, and FastAPI server configuration
- Auto-detection of required services
- Can be used standalone or as a base for StandardSettings
- Comprehensive validation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from .base_settings import BaseSettings, SettingsError, EnvParser, NullConfig
from .cache_settings import CacheSettings
from .tracing_settings import TracingSettings
from .logger_settings import LoggerSettings
from .mcp_settings import MCPServerSettings
from .fastapi_settings import FastAPIServerSettings
from .app_settings import AppSettings


@dataclass(frozen=True)
class ApiSettings(BaseSettings):
    """API-focused settings for running API servers.
    
    This class provides configuration for the minimum services needed to run
    an API server: cache, tracing, MCP server, and FastAPI server. It serves
    as an intermediate layer between BaseSettings and StandardSettings.
    
    Use this class when you need to configure an API server without the full
    set of services (LLM, embeddings, database) provided by StandardSettings.
    """
    
    # Core app settings component
    app: AppSettings = field(default_factory=lambda: AppSettings())
    
    # Default server host and port for API servers (configurable via SERVER_HOST/SERVER_PORT env vars)
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    
    # Optional service configurations for API servers
    cache: Optional[CacheSettings] = None
    tracing: Optional[TracingSettings] = None
    logger: Optional[LoggerSettings] = None
    mcp_server: Optional[MCPServerSettings] = None
    fastapi_server: Optional[FastAPIServerSettings] = None
    
    # Service enablement flags
    enable_cache: bool = field(default=False)
    enable_tracing: bool = field(default=False)
    enable_logger: bool = field(default=False)
    enable_mcp_server: bool = field(default=False)
    enable_fastapi_server: bool = field(default=False)
    
    @classmethod
    def from_env(
        cls,
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides
    ) -> "ApiSettings":
        """Create API settings from environment variables and auto-detect services."""
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        # Extract app-level overrides for AppSettings
        app_overrides = {}
        for key in ["app_name", "version", "environment", "log_level", "project_root"]:
            if key in overrides:
                app_overrides[key] = overrides.pop(key)
        
        # Parse core app settings using AppSettings with overrides
        app_settings = AppSettings.from_env(
            load_dotenv=False, 
            dotenv_paths=dotenv_paths, 
            **app_overrides
        )
        
        # Auto-detect services from environment
        enable_cache = cls._should_enable_cache(overrides)
        enable_tracing = cls._should_enable_tracing(overrides)
        enable_logger = cls._should_enable_logger(overrides)
        enable_mcp_server = cls._should_enable_mcp_server(overrides)
        enable_fastapi_server = cls._should_enable_fastapi_server(overrides)
        
        # Create service configurations if enabled
        cache_config = None
        if enable_cache:
            try:
                cache_config = CacheSettings.from_env(load_dotenv=False)
            except Exception:
                enable_cache = False
        
        tracing_config = None
        if enable_tracing:
            try:
                tracing_config = TracingSettings.from_env(load_dotenv=False)
            except Exception:
                enable_tracing = False
        
        logger_config = None
        if enable_logger:
            try:
                logger_config = LoggerSettings.from_env(load_dotenv=False)
            except Exception:
                enable_logger = False
        
        mcp_server_config = None
        if enable_mcp_server:
            try:
                mcp_server_config = MCPServerSettings.from_env(load_dotenv=False)
            except Exception:
                enable_mcp_server = False
        
        fastapi_server_config = None
        if enable_fastapi_server:
            try:
                fastapi_server_config = FastAPIServerSettings.from_env(load_dotenv=False)
            except Exception:
                enable_fastapi_server = False
        
        # Parse server_host and server_port from environment
        server_host = overrides.get("server_host")
        if server_host is None:
            server_host = EnvParser.get_env("SERVER_HOST", default="0.0.0.0")
        
        server_port = overrides.get("server_port")
        if server_port is None:
            server_port = EnvParser.get_env("SERVER_PORT", env_type=int, default=8080)
        
        # Build the settings dict
        settings_dict = {
            # Core app settings
            "app": app_settings,
            # Server host and port
            "server_host": server_host,
            "server_port": server_port,
            # Service configurations
            "cache": cache_config,
            "tracing": tracing_config,
            "logger": logger_config,
            "mcp_server": mcp_server_config,
            "fastapi_server": fastapi_server_config,
            "enable_cache": enable_cache,
            "enable_tracing": enable_tracing,
            "enable_logger": enable_logger,
            "enable_mcp_server": enable_mcp_server,
            "enable_fastapi_server": enable_fastapi_server,
        }
        
        # Apply overrides
        settings_dict.update(overrides)
        
        return cls(**settings_dict)
    
    @staticmethod
    def _should_enable_cache(overrides: dict) -> bool:
        """Check if cache should be enabled based on environment variables."""
        if "enable_cache" in overrides:
            return overrides["enable_cache"]
        
        if EnvParser.get_env("ENABLE_CACHE", env_type=bool) is not None:
            return EnvParser.get_env("ENABLE_CACHE", env_type=bool)
        
        # Auto-detect based on Redis/Valkey settings
        return (
            EnvParser.get_env("REDIS_HOST") or
            EnvParser.get_env("VALKEY_HOST") or
            EnvParser.get_env("CACHE_PROVIDER")
        ) is not None
    
    @staticmethod
    def _should_enable_tracing(overrides: dict) -> bool:
        """Check if tracing should be enabled based on environment variables."""
        if "enable_tracing" in overrides:
            return overrides["enable_tracing"]
        
        if EnvParser.get_env("ENABLE_TRACING", env_type=bool) is not None:
            return EnvParser.get_env("ENABLE_TRACING", env_type=bool)
        
        # Auto-detect based on Langfuse settings
        return (
            EnvParser.get_env("LANGFUSE_PUBLIC_KEY") and
            EnvParser.get_env("LANGFUSE_SECRET_KEY")
        ) is not None
    
    @staticmethod
    def _should_enable_logger(overrides: dict) -> bool:
        """Check if advanced logger should be enabled based on environment variables."""
        if "enable_logger" in overrides:
            return overrides["enable_logger"]
        
        if EnvParser.get_env("ENABLE_LOGGER", env_type=bool) is not None:
            return EnvParser.get_env("ENABLE_LOGGER", env_type=bool)
        
        # Auto-detect based on OVH LDP or file logging settings
        return (
            EnvParser.get_env("OVH_LDP_ENABLED", env_type=bool) or
            EnvParser.get_env("LOG_FILE_ENABLED", env_type=bool)
        ) or False
    
    @staticmethod
    def _should_enable_mcp_server(overrides: dict) -> bool:
        """Check if MCP server should be enabled based on environment variables."""
        if "enable_mcp_server" in overrides:
            return overrides["enable_mcp_server"]
        
        if EnvParser.get_env("ENABLE_MCP_SERVER", env_type=bool) is not None:
            return EnvParser.get_env("ENABLE_MCP_SERVER", env_type=bool)
        
        # Auto-detect based on MCP server settings (including SERVER_HOST/SERVER_PORT)
        return (
            EnvParser.get_env("MCP_SERVER_HOST") or
            EnvParser.get_env("MCP_SERVER_PORT") or
            EnvParser.get_env("MCP_PORT") or
            EnvParser.get_env("SERVER_HOST") or
            EnvParser.get_env("SERVER_PORT") or
            EnvParser.get_env("MCP_SERVER_NAME") or
            EnvParser.get_env("MCP_TRANSPORT")
        ) is not None

    @staticmethod
    def _should_enable_fastapi_server(overrides: dict) -> bool:
        """Check if FastAPI server should be enabled based on environment variables."""
        if "enable_fastapi_server" in overrides:
            return overrides["enable_fastapi_server"]
        
        if EnvParser.get_env("ENABLE_FASTAPI_SERVER", env_type=bool) is not None:
            return EnvParser.get_env("ENABLE_FASTAPI_SERVER", env_type=bool)
        
        # Auto-detect based on FastAPI server settings (including SERVER_HOST/SERVER_PORT)
        return (
            EnvParser.get_env("FASTAPI_HOST") or
            EnvParser.get_env("FASTAPI_PORT") or
            EnvParser.get_env("SERVER_HOST") or
            EnvParser.get_env("SERVER_PORT") or
            EnvParser.get_env("FASTAPI_RELOAD") or
            EnvParser.get_env("API_AUTH_ENABLED") or
            EnvParser.get_env("API_KEYS")
        ) is not None
    
    def validate(self) -> None:
        """Validate the API settings configuration."""
        # Validate individual service configurations
        if self.cache:
            self.cache.validate()
        if self.tracing:
            self.tracing.validate()
        if self.logger:
            self.logger.validate()
        if self.mcp_server:
            if not self.mcp_server.is_valid:
                raise SettingsError(f"MCP server configuration invalid: {', '.join(self.mcp_server.validate())}")
        if self.fastapi_server:
            self.fastapi_server.validate()
    
    def _get_attr(self, obj, attr, default=None):
        """Get attribute from object or dictionary."""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)
    
    def get_redis_config(self):
        """Get Redis configuration compatible with existing RedisConfig."""
        if not self.cache:
            raise SettingsError("Cache not configured")
        
        from ..cache.redis_config import RedisConfig
        return RedisConfig(
            host=self._get_attr(self.cache, "host"),
            port=self._get_attr(self.cache, "port"),
            db=self._get_attr(self.cache, "db"),
            prefix=self._get_attr(self.cache, "prefix"),
            ttl=self._get_attr(self.cache, "ttl"),
            password=self._get_attr(self.cache, "password"),
            time_out=self._get_attr(self.cache, "timeout"),
        )
    
    def get_mcp_server_config(self) -> MCPServerSettings:
        """Get MCP server configuration.""" 
        if not self.mcp_server:
            raise SettingsError("MCP server not configured")
        return self.mcp_server

    def get_fastapi_server_config(self) -> FastAPIServerSettings:
        """Get FastAPI server configuration."""
        if not self.fastapi_server:
            raise SettingsError("FastAPI server not configured")
        return self.fastapi_server

    # Backward-compatible properties for core app fields
    @property
    def app_name(self) -> str:
        """Get app name from app component."""
        return self.app.app_name
    
    @property
    def version(self) -> str:
        """Get version from app component."""
        return self.app.version
    
    @property
    def environment(self) -> str:
        """Get environment from app component."""
        return self.app.environment
    
    @property
    def log_level(self) -> str:
        """Get log level from app component."""
        return self.app.log_level
    
    @property
    def project_root(self) -> Optional[Path]:
        """Get project root from app component."""
        return self.app.project_root

    # Null-safe convenience properties for sub-configs
    @property
    def mcp_server_safe(self) -> MCPServerSettings | NullConfig:
        return self.mcp_server if self.mcp_server is not None else NullConfig()

    @property
    def fastapi_server_safe(self) -> FastAPIServerSettings | NullConfig:
        return self.fastapi_server if self.fastapi_server is not None else NullConfig()

    @property
    def cache_safe(self) -> CacheSettings | NullConfig:
        return self.cache if self.cache is not None else NullConfig()

    @property
    def tracing_safe(self) -> TracingSettings | NullConfig:
        return self.tracing if self.tracing is not None else NullConfig()
    
    @property
    def logger_safe(self) -> LoggerSettings | NullConfig:
        return self.logger if self.logger is not None else NullConfig()
    
    def as_dict(self) -> dict:
        """Convert to dictionary representation.
        
        Flattens app fields to root level for backward compatibility.
        """
        result = {
            # Flatten app fields to root level
            "app_name": self.app.app_name,
            "version": self.app.version,
            "environment": self.app.environment,
            "log_level": self.app.log_level,
            "project_root": str(self.app.project_root) if self.app.project_root else None,
            # Server host and port
            "server_host": self.server_host,
            "server_port": self.server_port,
            # Service configurations
            "cache": self.cache.as_dict() if self.cache else None,
            "tracing": self.tracing.as_dict() if self.tracing else None,
            "logger": self.logger.as_dict() if self.logger else None,
            "mcp_server": self.mcp_server.as_dict() if self.mcp_server else None,
            "fastapi_server": self.fastapi_server.as_dict() if self.fastapi_server else None,
            "enable_cache": self.enable_cache,
            "enable_tracing": self.enable_tracing,
            "enable_logger": self.enable_logger,
            "enable_mcp_server": self.enable_mcp_server,
            "enable_fastapi_server": self.enable_fastapi_server,
        }
        return result
