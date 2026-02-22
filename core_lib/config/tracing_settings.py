"""Tracing Configuration Settings.

This module contains configuration classes for tracing providers
including Langfuse and OpenTelemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .base_settings import BaseSettings, SettingsError, EnvParser


@dataclass(frozen=True)
class TracingSettings(BaseSettings):
    """Tracing configuration settings."""
    
    enabled: bool = True
    service_name: Optional[str] = None
    service_version: str = "0.1.0"
    
    # Langfuse settings
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "http://localhost:3000"
    
    @classmethod
    def from_env(
        cls,
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides
    ) -> "TracingSettings":
        """Create tracing settings from environment variables."""
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        # Read LANGFUSE_TRACING_ENABLED (preferred) with fallback to legacy TRACING_ENABLED
        langfuse_enabled = EnvParser.get_env("LANGFUSE_TRACING_ENABLED", env_type=bool)
        if langfuse_enabled is None:
            langfuse_enabled = EnvParser.get_env("TRACING_ENABLED", default=True, env_type=bool)

        settings_dict = {
            "enabled": langfuse_enabled,
            "service_name": EnvParser.get_env("APP_NAME", "SERVICE_NAME"),
            "service_version": EnvParser.get_env("APP_VERSION", "SERVICE_VERSION", default="0.1.0"),
            "langfuse_public_key": EnvParser.get_env("LANGFUSE_PUBLIC_KEY"),
            "langfuse_secret_key": EnvParser.get_env("LANGFUSE_SECRET_KEY"),
            "langfuse_host": EnvParser.get_env("LANGFUSE_HOST", default="http://localhost:3000"),
        }
        
        settings_dict.update(overrides)
        return cls(**settings_dict)
    
    def validate(self) -> None:
        """Validate tracing configuration."""
        if self.enabled and not self.langfuse_public_key:
            raise SettingsError("Tracing enabled but langfuse_public_key not provided")
        if self.enabled and not self.langfuse_secret_key:
            raise SettingsError("Tracing enabled but langfuse_secret_key not provided")
    
    def as_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "enabled": self.enabled,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "langfuse_public_key": self.langfuse_public_key,
            "langfuse_secret_key": self.langfuse_secret_key,
            "langfuse_host": self.langfuse_host,
        }