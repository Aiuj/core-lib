"""LLM Provider Configuration Settings.

This module contains configuration classes for Language Model providers
including OpenAI, Azure OpenAI, Google Gemini, and Ollama.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from .base_settings import BaseSettings, SettingsError, EnvParser


@dataclass(frozen=True)
class LLMSettings(BaseSettings):
    """LLM provider configuration settings."""
    
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    thinking_enabled: bool = False
    
    # OpenAI/Azure specific
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2024-08-01-preview"
    
    # Ollama specific
    ollama_host: Optional[str] = None
    ollama_timeout: Optional[int] = None
    
    # Gemini specific
    google_api_key: Optional[str] = None
    safety_settings: Optional[Dict[str, Any]] = None
    google_project: Optional[str] = None
    google_location: Optional[str] = None
    google_service_account_file: Optional[str] = None
    
    @classmethod
    def from_env(
        cls, 
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides
    ) -> "LLMSettings":
        """Create LLM settings from environment variables."""
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        # Auto-detect provider if not specified
        provider = overrides.get("provider") or cls._detect_provider()
        
        # Parse common settings
        model = EnvParser.get_env(
            "LLM_MODEL", f"{provider.upper()}_MODEL", 
            default="gpt-4o-mini"
        )
        temperature = EnvParser.get_env(
            "LLM_TEMPERATURE", f"{provider.upper()}_TEMPERATURE",
            default=0.7, env_type=float
        )
        max_tokens = EnvParser.get_env(
            "LLM_MAX_TOKENS", f"{provider.upper()}_MAX_TOKENS",
            env_type=int
        )
        thinking_enabled = EnvParser.get_env(
            "LLM_THINKING_ENABLED", f"{provider.upper()}_THINKING_ENABLED",
            default=False, env_type=bool
        )
        
        # Provider-specific settings
        settings_dict = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking_enabled": thinking_enabled,
        }
        
        if provider.lower() in ("openai", "azure"):
            settings_dict.update({
                "api_key": EnvParser.get_env("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
                "base_url": EnvParser.get_env("OPENAI_BASE_URL"),
                "organization": EnvParser.get_env("OPENAI_ORG", "OPENAI_ORGANIZATION"),
                "project": EnvParser.get_env("OPENAI_PROJECT"),
                "azure_endpoint": EnvParser.get_env("AZURE_OPENAI_ENDPOINT"),
                "azure_api_version": EnvParser.get_env("AZURE_OPENAI_API_VERSION", default="2024-08-01-preview"),
            })
        elif provider.lower() == "ollama":
            settings_dict.update({
                "ollama_host": EnvParser.get_env("OLLAMA_HOST", "OLLAMA_BASE_URL"),
                "ollama_timeout": EnvParser.get_env("OLLAMA_TIMEOUT", env_type=int),
            })
        elif provider.lower() in ("gemini", "google"):
            settings_dict.update({
                "google_api_key": EnvParser.get_env("GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"),
                "google_project": EnvParser.get_env("GOOGLE_CLOUD_PROJECT", "GOOGLE_PROJECT_ID"),
                "google_location": EnvParser.get_env("GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION"),
                "google_service_account_file": EnvParser.get_env(
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "GEMINI_SERVICE_ACCOUNT_FILE",
                ),
            })
        
        # Apply overrides
        settings_dict.update(overrides)
        
        return cls(**settings_dict)
    
    @staticmethod
    def _detect_provider() -> str:
        """Auto-detect LLM provider from environment variables."""
        if EnvParser.get_env("LLM_PROVIDER"):
            return EnvParser.get_env("LLM_PROVIDER")
        
        # Check for provider-specific API keys in order of preference
        # Azure first since it's more specific than generic OpenAI
        if EnvParser.get_env("AZURE_OPENAI_API_KEY") or EnvParser.get_env("AZURE_OPENAI_ENDPOINT"):
            return "azure"
        elif EnvParser.get_env(
            "GEMINI_API_KEY",
            "GOOGLE_GENAI_API_KEY",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_PROJECT_ID",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_CLOUD_REGION",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ):
            return "gemini"
        elif EnvParser.get_env("OLLAMA_HOST", "OLLAMA_BASE_URL"):
            return "ollama"
        elif EnvParser.get_env("OPENAI_API_KEY"):
            return "openai"
        
        return "openai"  # Default
    
    def validate(self) -> None:
        """Validate LLM configuration."""
        if self.provider.lower() in ("openai", "azure") and not self.api_key:
            raise SettingsError("OpenAI/Azure provider requires api_key")
        elif self.provider.lower() in ("gemini", "google"):
            has_vertex_config = bool(self.google_project and self.google_location)
            if not self.google_api_key and not has_vertex_config:
                raise SettingsError(
                    "Gemini provider requires google_api_key or Vertex settings (google_project + google_location)"
                )
        
        if self.temperature < 0 or self.temperature > 2:
            raise SettingsError("Temperature must be between 0 and 2")
        
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise SettingsError("max_tokens must be positive")
    
    def as_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking_enabled": self.thinking_enabled,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "organization": self.organization,
            "project": self.project,
            "azure_endpoint": self.azure_endpoint,
            "azure_api_version": self.azure_api_version,
            "ollama_host": self.ollama_host,
            "ollama_timeout": self.ollama_timeout,
            "google_api_key": self.google_api_key,
            "safety_settings": self.safety_settings,
            "google_project": self.google_project,
            "google_location": self.google_location,
            "google_service_account_file": self.google_service_account_file,
        }

    def to_llm_config(self):
        """Convert settings to a provider-specific LLMConfig instance."""
        provider = self.provider.lower()

        if provider in ("openai", "azure"):
            from core_lib.llm import OpenAIConfig
            return OpenAIConfig(
                api_key=self.api_key or "",
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_enabled=self.thinking_enabled,
                base_url=self.base_url,
                organization=self.organization,
                project=self.project,
                azure_endpoint=self.azure_endpoint,
                azure_api_version=self.azure_api_version,
            )

        if provider in ("gemini", "google"):
            from core_lib.llm import GeminiConfig
            return GeminiConfig(
                api_key=self.google_api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_enabled=self.thinking_enabled,
                base_url=self.base_url or "https://generativelanguage.googleapis.com",
                safety_settings=self.safety_settings,
                project=self.google_project,
                location=self.google_location,
                service_account_file=self.google_service_account_file,
            )

        if provider == "ollama":
            from core_lib.llm import OllamaConfig
            return OllamaConfig(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_enabled=self.thinking_enabled,
                base_url=self.ollama_host or "http://localhost:11434",
            )

        raise SettingsError(f"Unsupported LLM provider: {self.provider}")