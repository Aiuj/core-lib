"""Embeddings Provider Configuration Settings.

This module contains configuration classes for embeddings providers
including OpenAI, Google, Hugging Face, Ollama, and local models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_settings import BaseSettings, EnvParser
from .provider_config_loader import get_service_provider_configs


@dataclass(frozen=True) 
class EmbeddingsSettings(BaseSettings):
    """Embeddings provider configuration settings."""
    
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    embedding_dimension: Optional[int] = None
    task_type: Optional[str] = None
    title: Optional[str] = None
    
    # Provider-specific settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    google_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    # Timeout settings (EMBEDDING_TIMEOUT is common default, provider-specific overrides)
    timeout: int = 10  # Default 10 seconds; from EMBEDDING_TIMEOUT
    
    # Ollama settings
    ollama_host: Optional[str] = None
    ollama_url: Optional[str] = None
    ollama_timeout: Optional[int] = None
    
    # Infinity settings
    infinity_url: Optional[str] = None
    infinity_timeout: Optional[int] = None
    infinity_token: Optional[str] = None  # Authentication token(s), comma-separated
    
    # Local model settings
    device: str = "auto"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    use_sentence_transformers: bool = True
    
    # Cache settings
    cache_duration_seconds: int = 7200
    
    # Prefix settings for asymmetric retrieval models (E5, BGE, etc.)
    # If None, prefixes are auto-detected based on model name
    # Set to empty string "" to explicitly disable prefixes
    query_prefix: Optional[str] = None
    passage_prefix: Optional[str] = None
    auto_detect_prefixes: bool = True  # Auto-detect prefixes from model database

    # YAML-driven provider chain (priority-ordered) for failover/routing
    provider_configs: Tuple[Dict[str, Any], ...] = ()
    config_file: Optional[str] = None
    intelligence_level: Optional[int] = None
    usage: Optional[str] = None

    @staticmethod
    def _normalize_provider_name(provider: Optional[str]) -> str:
        if not provider:
            return "openai"
        normalized = str(provider).strip().lower()
        if normalized in {"google", "google_genai", "gemini"}:
            return "google_genai"
        if normalized == "huggingface":
            return "local"
        return normalized

    @classmethod
    def _build_embedding_provider_configs(
        cls,
        config_file: Optional[str],
        intelligence_level: Optional[int],
        usage: Optional[str],
        defaults: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        configs = get_service_provider_configs(
            service="embedding",
            config_file_path=config_file,
            intelligence_level=intelligence_level,
            usage=usage,
        )
        normalized: List[Dict[str, Any]] = []

        for config in configs:
            provider = cls._normalize_provider_name(config.get("provider") or config.get("type"))
            if not provider:
                continue

            entry: Dict[str, Any] = {
                "provider": provider,
                "model": config.get("model") or defaults.get("model"),
                "priority": int(config.get("priority", 100)),
            }

            base_url = (
                config.get("base_url")
                or config.get("host")
                or config.get("endpoint")
                or config.get("url")
            )
            if base_url:
                entry["base_url"] = base_url

            timeout = config.get("timeout")
            if timeout is not None:
                entry["timeout"] = int(timeout)

            dim = config.get("embedding_dimension") or config.get("dimension")
            if dim is not None:
                entry["embedding_dim"] = int(dim)

            if provider == "openai":
                api_key = config.get("api_key") or config.get("key") or defaults.get("api_key")
                if api_key:
                    entry["api_key"] = api_key
                if config.get("organization"):
                    entry["organization"] = config.get("organization")
                if config.get("project"):
                    entry["project"] = config.get("project")
            elif provider == "google_genai":
                api_key = config.get("api_key") or config.get("google_api_key") or defaults.get("google_api_key")
                if api_key:
                    entry["api_key"] = api_key
                if config.get("task_type"):
                    entry["task_type"] = config.get("task_type")
                if config.get("title"):
                    entry["title"] = config.get("title")
            elif provider == "infinity":
                token = (
                    config.get("token")
                    or config.get("infinity_token")
                    or config.get("embedding_token")
                    or defaults.get("infinity_token")
                )
                if token:
                    entry["token"] = token
            elif provider == "local":
                if config.get("device"):
                    entry["device"] = config.get("device")
                if config.get("cache_dir"):
                    entry["cache_dir"] = config.get("cache_dir")
                if "trust_remote_code" in config:
                    entry["trust_remote_code"] = bool(config.get("trust_remote_code"))

            normalized.append(entry)

        return normalized
    
    @classmethod
    def from_env(
        cls,
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides
    ) -> "EmbeddingsSettings":
        """Create embeddings settings from environment variables."""
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        provider = EnvParser.get_env("EMBEDDING_PROVIDER", default="openai")
        model = EnvParser.get_env("EMBEDDING_MODEL", default="text-embedding-3-small")
        
        # Unified base URL and timeout (common defaults for all providers)
        embedding_base_url = EnvParser.get_env("EMBEDDING_BASE_URL")
        # Common timeout with 10-second default
        embedding_timeout = EnvParser.get_env("EMBEDDING_TIMEOUT", env_type=int, default=10)
        
        # Provider-specific URLs with fallback to common EMBEDDING_BASE_URL
        ollama_url = EnvParser.get_env("OLLAMA_URL") or embedding_base_url
        infinity_url = (
            EnvParser.get_env("INFINITY_BASE_URL") or 
            embedding_base_url
        )
        infinity_token = EnvParser.get_env("INFINITY_TOKEN") or EnvParser.get_env("EMBEDDING_TOKEN")
        openai_base_url = (
            EnvParser.get_env("OPENAI_BASE_URL") or 
            EnvParser.get_env("BASE_URL") or 
            embedding_base_url
        )
        
        config_file = (
            EnvParser.get_env("EMBEDDING_PROVIDERS_FILE")
            or EnvParser.get_env("LLM_PROVIDERS_FILE")
        )
        intelligence_level = EnvParser.get_env("EMBEDDING_INTELLIGENCE_LEVEL", env_type=int)
        usage = EnvParser.get_env("EMBEDDING_USAGE")

        settings_dict = {
            "provider": str(provider).strip().lower(),
            "model": model,
            "embedding_dimension": EnvParser.get_env("EMBEDDING_DIMENSION", env_type=int),
            "task_type": EnvParser.get_env("EMBEDDING_TASK_TYPE"),
            "title": EnvParser.get_env("EMBEDDING_TITLE"),
            "timeout": embedding_timeout,
            "api_key": EnvParser.get_env("OPENAI_API_KEY", "API_KEY"),
            "base_url": openai_base_url,
            "organization": EnvParser.get_env("OPENAI_ORGANIZATION"),
            "project": EnvParser.get_env("OPENAI_PROJECT"),
            "google_api_key": EnvParser.get_env("GOOGLE_GENAI_API_KEY", "GEMINI_API_KEY"),
            "huggingface_api_key": EnvParser.get_env("HUGGINGFACE_API_KEY"),
            "ollama_host": EnvParser.get_env("OLLAMA_HOST"),
            "ollama_url": ollama_url,
            "ollama_timeout": EnvParser.get_env("OLLAMA_TIMEOUT", env_type=int) if EnvParser.get_env("OLLAMA_TIMEOUT") else embedding_timeout,
            "infinity_url": infinity_url,
            "infinity_timeout": EnvParser.get_env("INFINITY_TIMEOUT", env_type=int) if EnvParser.get_env("INFINITY_TIMEOUT") else embedding_timeout,
            "infinity_token": infinity_token,
            "device": EnvParser.get_env("EMBEDDING_DEVICE", default="auto"),
            "cache_dir": EnvParser.get_env("EMBEDDING_CACHE_DIR"),
            "trust_remote_code": EnvParser.get_env("EMBEDDING_TRUST_REMOTE_CODE", default=False, env_type=bool),
            "use_sentence_transformers": EnvParser.get_env("EMBEDDING_USE_SENTENCE_TRANSFORMERS", default=True, env_type=bool),
            "cache_duration_seconds": EnvParser.get_env("EMBEDDING_CACHE_DURATION_SECONDS", default=7200, env_type=int),
            # Prefix settings - None means auto-detect, "" means disable
            "query_prefix": EnvParser.get_env("EMBEDDING_QUERY_PREFIX"),
            "passage_prefix": EnvParser.get_env("EMBEDDING_PASSAGE_PREFIX"),
            "auto_detect_prefixes": EnvParser.get_env("EMBEDDING_AUTO_DETECT_PREFIXES", default=True, env_type=bool),
            "config_file": config_file,
            "intelligence_level": intelligence_level,
            "usage": usage,
        }

        yaml_provider_configs = cls._build_embedding_provider_configs(
            config_file=config_file,
            intelligence_level=intelligence_level,
            usage=usage,
            defaults=settings_dict,
        )
        if yaml_provider_configs:
            selected = yaml_provider_configs[0]
            settings_dict["provider"] = str(selected.get("provider", settings_dict["provider"])).strip().lower()
            settings_dict["model"] = selected.get("model") or settings_dict["model"]

            # Apply selected provider overrides while preserving env defaults
            if selected.get("embedding_dim") is not None:
                settings_dict["embedding_dimension"] = selected.get("embedding_dim")
            if selected.get("base_url"):
                if settings_dict["provider"] == "infinity":
                    settings_dict["infinity_url"] = selected.get("base_url")
                elif settings_dict["provider"] == "openai":
                    settings_dict["base_url"] = selected.get("base_url")
                elif settings_dict["provider"] == "ollama":
                    settings_dict["ollama_url"] = selected.get("base_url")

            if selected.get("timeout") is not None:
                timeout_val = int(selected.get("timeout"))
                settings_dict["timeout"] = timeout_val
                if settings_dict["provider"] == "infinity":
                    settings_dict["infinity_timeout"] = timeout_val
                if settings_dict["provider"] == "ollama":
                    settings_dict["ollama_timeout"] = timeout_val

            if selected.get("api_key"):
                if settings_dict["provider"] in {"google", "google_genai", "gemini"}:
                    settings_dict["google_api_key"] = selected.get("api_key")
                else:
                    settings_dict["api_key"] = selected.get("api_key")
            if selected.get("token"):
                settings_dict["infinity_token"] = selected.get("token")
            if selected.get("organization"):
                settings_dict["organization"] = selected.get("organization")
            if selected.get("project"):
                settings_dict["project"] = selected.get("project")
            if selected.get("task_type"):
                settings_dict["task_type"] = selected.get("task_type")
            if selected.get("title"):
                settings_dict["title"] = selected.get("title")
            if selected.get("device"):
                settings_dict["device"] = selected.get("device")
            if selected.get("cache_dir"):
                settings_dict["cache_dir"] = selected.get("cache_dir")
            if "trust_remote_code" in selected:
                settings_dict["trust_remote_code"] = bool(selected.get("trust_remote_code"))

        settings_dict["provider_configs"] = tuple(yaml_provider_configs)
        
        settings_dict.update(overrides)
        return cls(**settings_dict)
    
    def as_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "provider": self.provider,
            "model": self.model,
            "embedding_dimension": self.embedding_dimension,
            "task_type": self.task_type,
            "title": self.title,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "organization": self.organization,
            "project": self.project,
            "google_api_key": self.google_api_key,
            "huggingface_api_key": self.huggingface_api_key,
            "timeout": self.timeout,
            "ollama_host": self.ollama_host,
            "ollama_url": self.ollama_url,
            "ollama_timeout": self.ollama_timeout,
            "infinity_url": self.infinity_url,
            "infinity_timeout": self.infinity_timeout,
            "infinity_token": self.infinity_token,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "use_sentence_transformers": self.use_sentence_transformers,
            "cache_duration_seconds": self.cache_duration_seconds,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "auto_detect_prefixes": self.auto_detect_prefixes,
            "provider_configs": list(self.provider_configs),
            "config_file": self.config_file,
            "intelligence_level": self.intelligence_level,
            "usage": self.usage,
        }