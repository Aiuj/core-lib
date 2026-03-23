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
    otlp_instance_id: Optional[str] = None
    otlp_log_channel: Optional[str] = None
    
    # Langfuse settings
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "http://localhost:3000"

    @staticmethod
    def _resolve_langfuse_credentials() -> tuple[Optional[str], Optional[str]]:
        """Resolve Langfuse credentials based on ENVIRONMENT.

        Production uses the unprefixed ``LANGFUSE_*`` variables.
        Non-production environments prefer ``LANGFUSE_TEST_*`` and fall back to
        ``LANGFUSE_*`` for backward compatibility.
        """
        environment = (EnvParser.get_env("ENVIRONMENT", default="dev") or "dev").lower()
        is_production = environment in {"prod", "production"}

        if is_production:
            return (
                EnvParser.get_env("LANGFUSE_PUBLIC_KEY"),
                EnvParser.get_env("LANGFUSE_SECRET_KEY"),
            )

        return (
            EnvParser.get_env("LANGFUSE_TEST_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY"),
            EnvParser.get_env("LANGFUSE_TEST_SECRET_KEY", "LANGFUSE_SECRET_KEY"),
        )
    
    @classmethod
    def from_env(
        cls,
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides
    ) -> "TracingSettings":
        """Create tracing settings from environment variables."""
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        langfuse_public_key, langfuse_secret_key = cls._resolve_langfuse_credentials()
        
        # Read LANGFUSE_TRACING_ENABLED (preferred) with fallback to legacy TRACING_ENABLED
        langfuse_enabled = EnvParser.get_env("LANGFUSE_TRACING_ENABLED", env_type=bool)
        if langfuse_enabled is None:
            langfuse_enabled = EnvParser.get_env("TRACING_ENABLED", default=True, env_type=bool)

        settings_dict = {
            "enabled": langfuse_enabled,
            "service_name": EnvParser.get_env("APP_NAME", "SERVICE_NAME"),
            "service_version": EnvParser.get_env("APP_VERSION", "SERVICE_VERSION", default="0.1.0"),
            "otlp_instance_id": EnvParser.get_env("OTLP_INSTANCE_ID"),
            "otlp_log_channel": EnvParser.get_env("OTLP_LOG_CHANNEL"),
            "langfuse_public_key": langfuse_public_key,
            "langfuse_secret_key": langfuse_secret_key,
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
        if self.otlp_instance_id is not None and not self.otlp_instance_id.strip():
            raise SettingsError("OTLP instance id cannot be empty when provided")
        if self.otlp_log_channel is not None and not self.otlp_log_channel.strip():
            raise SettingsError("OTLP log channel cannot be empty when provided")
    
    def as_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "enabled": self.enabled,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "otlp_instance_id": self.otlp_instance_id,
            "otlp_log_channel": self.otlp_log_channel,
            "langfuse_public_key": self.langfuse_public_key,
            "langfuse_secret_key": self.langfuse_secret_key,
            "langfuse_host": self.langfuse_host,
        }