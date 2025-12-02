"""
Reranker configuration module.

Provides reranker configuration loaded from environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from ..config.base_settings import BaseSettings, EnvParser


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
        
        provider = EnvParser.get_env("RERANKER_PROVIDER", default="infinity")
        model = EnvParser.get_env(
            "RERANKER_MODEL", 
            default="BAAI/bge-reranker-v2-m3"  # Good multilingual reranker
        )
        
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
            "device": EnvParser.get_env("RERANKER_DEVICE", default="auto"),
            "cache_dir": EnvParser.get_env("RERANKER_CACHE_DIR"),
            "trust_remote_code": EnvParser.get_env("RERANKER_TRUST_REMOTE_CODE", default=False, env_type=bool),
            "timeout": timeout,
            "cache_duration_seconds": EnvParser.get_env("RERANKER_CACHE_DURATION_SECONDS", default=3600, env_type=int),
            "default_top_k": EnvParser.get_env("RERANKER_DEFAULT_TOP_K", default=10, env_type=int),
            "score_threshold": EnvParser.get_env("RERANKER_SCORE_THRESHOLD", env_type=float),
        }
        
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
            "device": self.device,
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "timeout": self.timeout,
            "cache_duration_seconds": self.cache_duration_seconds,
            "default_top_k": self.default_top_k,
            "score_threshold": self.score_threshold,
        }


# Singleton used by reranker modules
reranker_settings: RerankerSettings = RerankerSettings.from_env(load_dotenv=False)
