"""
Reranker configuration module.

Provides reranker configuration loaded from environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..config.base_settings import BaseSettings, EnvParser
from ..config.provider_config_loader import get_service_provider_configs


@dataclass(frozen=True)
class RerankerSettings(BaseSettings):
    """Reranker provider configuration settings."""
    
    # Provider selection (infinity, cohere, local)
    provider: str = "infinity"
    
    # Model name - defaults to a good multilingual reranker
    model: str = "BAAI/bge-reranker-v2-m3"
    
    # Provider-specific settings
    api_key: Optional[str] = None  # For Cohere
    
    # Infinity settings
    infinity_url: Optional[str] = None
    infinity_timeout: Optional[int] = None
    infinity_token: Optional[str] = None
    infinity_wake_on_lan: Optional[Dict[str, Any]] = None
    
    # Local model settings
    device: str = "auto"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    
    # General settings
    timeout: int = 30  # Default timeout in seconds
    cache_duration_seconds: int = 3600  # Cache results for 1 hour by default
    
    # Reranking behavior
    default_top_k: int = 10  # Default number of results to return
    score_threshold: Optional[float] = None  # Optional minimum score threshold

    # YAML-driven provider chain (priority-ordered) for failover/routing
    provider_configs: Tuple[Dict[str, Any], ...] = ()
    config_file: Optional[str] = None
    intelligence_level: Optional[int] = None
    usage: Optional[str] = None

    @staticmethod
    def _normalize_provider_name(provider: Optional[str]) -> str:
        if not provider:
            return "infinity"
        normalized = str(provider).strip().lower()
        if normalized in {"crossencoder", "cross-encoder"}:
            return "local"
        return normalized

    @classmethod
    def _build_reranker_provider_configs(
        cls,
        config_file: Optional[str],
        intelligence_level: Optional[int],
        usage: Optional[str],
        defaults: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        configs = get_service_provider_configs(
            service="reranker",
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

            if provider == "cohere":
                api_key = config.get("api_key") or config.get("key") or defaults.get("api_key")
                if api_key:
                    entry["api_key"] = api_key
            elif provider == "infinity":
                token = (
                    config.get("token")
                    or config.get("infinity_token")
                    or config.get("reranker_token")
                    or defaults.get("infinity_token")
                )
                if token:
                    entry["token"] = token
                wake_on_lan = config.get("wake_on_lan")
                if isinstance(wake_on_lan, dict):
                    entry["wake_on_lan"] = wake_on_lan
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
    ) -> "RerankerSettings":
        """Create reranker settings from environment variables.
        
        Environment variables:
            RERANKER_PROVIDER: Provider name (infinity, cohere, local)
            RERANKER_MODEL: Model name
            RERANKER_TIMEOUT: Request timeout in seconds
            RERANKER_CACHE_DURATION_SECONDS: Cache duration
            RERANKER_DEFAULT_TOP_K: Default number of results
            RERANKER_SCORE_THRESHOLD: Minimum score threshold
            
            # Infinity-specific
            INFINITY_BASE_URL: Infinity server URL (also used for embeddings)
            INFINITY_RERANK_URL: Dedicated reranker URL (overrides INFINITY_BASE_URL)
            INFINITY_TIMEOUT: Request timeout
            INFINITY_TOKEN: Authentication token
            
            # Cohere-specific
            COHERE_API_KEY: Cohere API key
            
            # Local model-specific
            RERANKER_DEVICE: Device to run on (cpu, cuda, auto)
            RERANKER_CACHE_DIR: Model cache directory
            RERANKER_TRUST_REMOTE_CODE: Whether to trust remote code
        """
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        provider = cls._normalize_provider_name(EnvParser.get_env("RERANKER_PROVIDER", default="infinity"))
        model = EnvParser.get_env(
            "RERANKER_MODEL", 
            default="BAAI/bge-reranker-v2-m3"  # Good multilingual reranker
        )

        config_file = (
            EnvParser.get_env("RERANKER_PROVIDERS_FILE")
            or EnvParser.get_env("LLM_PROVIDERS_FILE")
        )
        intelligence_level = EnvParser.get_env("RERANKER_INTELLIGENCE_LEVEL", env_type=int)
        usage = EnvParser.get_env("RERANKER_USAGE")
        
        # Get Infinity URL - check reranker-specific first, then general
        infinity_url = (
            EnvParser.get_env("INFINITY_RERANK_URL") or
            EnvParser.get_env("INFINITY_BASE_URL") or
            EnvParser.get_env("RERANKER_BASE_URL")
        )
        
        # Get timeout
        timeout = EnvParser.get_env("RERANKER_TIMEOUT", env_type=int, default=30)
        infinity_timeout = EnvParser.get_env("INFINITY_TIMEOUT", env_type=int) or timeout
        
        settings_dict = {
            "provider": provider,
            "model": model,
            "api_key": EnvParser.get_env("COHERE_API_KEY"),
            "infinity_url": infinity_url,
            "infinity_timeout": infinity_timeout,
            "infinity_token": EnvParser.get_env("INFINITY_TOKEN") or EnvParser.get_env("RERANKER_TOKEN"),
            "infinity_wake_on_lan": None,
            "device": EnvParser.get_env("RERANKER_DEVICE", default="auto"),
            "cache_dir": EnvParser.get_env("RERANKER_CACHE_DIR"),
            "trust_remote_code": EnvParser.get_env("RERANKER_TRUST_REMOTE_CODE", default=False, env_type=bool),
            "timeout": timeout,
            "cache_duration_seconds": EnvParser.get_env("RERANKER_CACHE_DURATION_SECONDS", default=3600, env_type=int),
            "default_top_k": EnvParser.get_env("RERANKER_DEFAULT_TOP_K", default=10, env_type=int),
            "score_threshold": EnvParser.get_env("RERANKER_SCORE_THRESHOLD", env_type=float),
            "config_file": config_file,
            "intelligence_level": intelligence_level,
            "usage": usage,
        }

        yaml_provider_configs = cls._build_reranker_provider_configs(
            config_file=config_file,
            intelligence_level=intelligence_level,
            usage=usage,
            defaults=settings_dict,
        )
        if yaml_provider_configs:
            selected = yaml_provider_configs[0]
            settings_dict["provider"] = cls._normalize_provider_name(selected.get("provider", settings_dict["provider"]))
            settings_dict["model"] = selected.get("model") or settings_dict["model"]

            if selected.get("base_url"):
                settings_dict["infinity_url"] = selected.get("base_url")
            if selected.get("timeout") is not None:
                timeout_val = int(selected.get("timeout"))
                settings_dict["timeout"] = timeout_val
                settings_dict["infinity_timeout"] = timeout_val
            if selected.get("api_key"):
                settings_dict["api_key"] = selected.get("api_key")
            if selected.get("token"):
                settings_dict["infinity_token"] = selected.get("token")
            if isinstance(selected.get("wake_on_lan"), dict):
                settings_dict["infinity_wake_on_lan"] = selected.get("wake_on_lan")
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
            "api_key": self.api_key,
            "infinity_url": self.infinity_url,
            "infinity_timeout": self.infinity_timeout,
            "infinity_token": self.infinity_token,
            "infinity_wake_on_lan": self.infinity_wake_on_lan,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "timeout": self.timeout,
            "cache_duration_seconds": self.cache_duration_seconds,
            "default_top_k": self.default_top_k,
            "score_threshold": self.score_threshold,
            "provider_configs": list(self.provider_configs),
            "config_file": self.config_file,
            "intelligence_level": self.intelligence_level,
            "usage": self.usage,
        }


# Singleton used by reranker modules
reranker_settings: RerankerSettings = RerankerSettings.from_env(load_dotenv=False)
