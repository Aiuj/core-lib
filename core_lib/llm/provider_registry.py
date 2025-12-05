"""Provider registry for managing multiple LLM provider configurations.

This module provides a flexible way to configure multiple LLM providers with
their credentials, models, and settings. It supports:
- Loading from environment variables (JSON or individual vars)
- Loading from configuration files (JSON/YAML)
- Programmatic configuration via dictionaries
- Automatic fallback selection when primary provider fails

Example environment variable (LLM_PROVIDERS):
    [
        {"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash"},
        {"provider": "openai", "api_key": "sk-...", "model": "gpt-4o-mini"},
        {"provider": "ollama", "host": "http://localhost:11434", "model": "llama3.2"}
    ]

Example usage:
    from core_lib.llm.provider_registry import ProviderRegistry
    
    # Load from environment
    registry = ProviderRegistry.from_env()
    
    # Get primary provider client
    client = registry.get_client()
    
    # Get fallback client (next available)
    fallback = registry.get_fallback_client()
    
    # Iterate through all configured clients
    for client, is_fallback in registry.iter_clients():
        try:
            response = client.chat(messages)
            break
        except Exception:
            continue
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

from core_lib import get_module_logger

logger = get_module_logger()


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider.
    
    This is a provider-agnostic configuration that can be converted to
    provider-specific configs (GeminiConfig, OpenAIConfig, OllamaConfig).
    
    Attributes:
        provider: Provider name (gemini, openai, azure-openai, ollama)
        model: Model name/identifier
        api_key: API key (for cloud providers)
        host: Base URL/host (for Ollama or custom endpoints)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        thinking_enabled: Enable step-by-step thinking
        priority: Lower number = higher priority (for fallback ordering)
        enabled: Whether this provider is enabled
        min_intelligence_level: Minimum intelligence level to use this provider (0-10)
        max_intelligence_level: Maximum intelligence level to use this provider (0-10)
        tier: Model tier label ("low", "standard", "high") for categorization
        extra: Additional provider-specific settings
    """
    provider: str
    model: str = ""
    api_key: Optional[str] = None
    host: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    thinking_enabled: bool = False
    priority: int = 100
    enabled: bool = True
    
    # Intelligence level configuration
    min_intelligence_level: int = 0   # Minimum level this provider handles
    max_intelligence_level: int = 10  # Maximum level this provider handles
    tier: str = "standard"            # "low", "standard", "high"
    
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # Provider-specific aliases
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    
    def __post_init__(self):
        """Normalize provider name and set defaults."""
        self.provider = self.provider.lower().strip()
        
        # Normalize provider aliases
        if self.provider in ("google", "google-genai", "google_genai"):
            self.provider = "gemini"
        elif self.provider in ("azure", "azure_openai"):
            self.provider = "azure-openai"
        
        # Set default models if not specified
        if not self.model:
            defaults = {
                "gemini": "gemini-2.0-flash",
                "openai": "gpt-4o-mini",
                "azure-openai": "gpt-4o-mini",
                "ollama": "llama3.2",
            }
            self.model = defaults.get(self.provider, "")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        """Create a ProviderConfig from a dictionary.
        
        Supports flexible key names for ease of configuration.
        """
        # Normalize keys
        normalized = {}
        
        # Provider (required)
        normalized["provider"] = data.get("provider", data.get("type", ""))
        
        # Model
        normalized["model"] = data.get("model", data.get("model_name", ""))
        
        # API Key (multiple possible key names)
        normalized["api_key"] = data.get("api_key") or data.get("apiKey") or data.get("key")
        
        # Host/Base URL
        normalized["host"] = (
            data.get("host") or 
            data.get("base_url") or 
            data.get("baseUrl") or
            data.get("endpoint")
        )
        
        # Temperature
        if "temperature" in data:
            normalized["temperature"] = float(data["temperature"])
        
        # Max tokens
        max_tokens = data.get("max_tokens") or data.get("maxTokens")
        if max_tokens is not None:
            normalized["max_tokens"] = int(max_tokens)
        
        # Thinking mode
        thinking = data.get("thinking_enabled") or data.get("thinkingEnabled") or data.get("thinking")
        if thinking is not None:
            normalized["thinking_enabled"] = bool(thinking) if not isinstance(thinking, str) else thinking.lower() == "true"
        
        # Priority
        if "priority" in data:
            normalized["priority"] = int(data["priority"])
        
        # Enabled
        enabled = data.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.lower() in ("true", "1", "yes")
        normalized["enabled"] = enabled
        
        # Intelligence level configuration
        min_level = data.get("min_intelligence_level") or data.get("minIntelligenceLevel") or data.get("min_level")
        if min_level is not None:
            normalized["min_intelligence_level"] = int(min_level)
        
        max_level = data.get("max_intelligence_level") or data.get("maxIntelligenceLevel") or data.get("max_level")
        if max_level is not None:
            normalized["max_intelligence_level"] = int(max_level)
        
        tier = data.get("tier") or data.get("model_tier") or data.get("modelTier")
        if tier:
            normalized["tier"] = str(tier).lower()
        
        # Azure-specific
        normalized["azure_endpoint"] = data.get("azure_endpoint") or data.get("azureEndpoint")
        normalized["azure_api_version"] = data.get("azure_api_version") or data.get("azureApiVersion")
        
        # OpenAI-specific
        normalized["organization"] = data.get("organization") or data.get("org")
        normalized["project"] = data.get("project")
        
        # Collect remaining keys as extra
        known_keys = {
            "provider", "type", "model", "model_name", "api_key", "apiKey", "key",
            "host", "base_url", "baseUrl", "endpoint", "temperature", "max_tokens",
            "maxTokens", "thinking_enabled", "thinkingEnabled", "thinking", "priority",
            "enabled", "azure_endpoint", "azureEndpoint", "azure_api_version",
            "azureApiVersion", "organization", "org", "project"
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        normalized["extra"] = extra
        
        return cls(**{k: v for k, v in normalized.items() if v is not None or k in ("api_key", "host")})
    
    def to_llm_config(self):
        """Convert to a provider-specific LLMConfig object.
        
        Returns:
            GeminiConfig, OpenAIConfig, or OllamaConfig instance
        """
        if self.provider == "gemini":
            from .providers.google_genai_provider import GeminiConfig
            return GeminiConfig(
                api_key=self.api_key or "",
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_enabled=self.thinking_enabled,
                base_url=self.host or "https://generativelanguage.googleapis.com",
            )
        
        elif self.provider in ("openai", "azure-openai"):
            from .providers.openai_provider import OpenAIConfig
            return OpenAIConfig(
                api_key=self.api_key or "",
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_enabled=self.thinking_enabled,
                base_url=self.host,
                organization=self.organization,
                project=self.project,
                azure_endpoint=self.azure_endpoint,
                azure_api_version=self.azure_api_version or "2024-08-01-preview",
            )
        
        elif self.provider == "ollama":
            from .providers.ollama_provider import OllamaConfig
            return OllamaConfig(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_enabled=self.thinking_enabled,
                base_url=self.host or "http://localhost:11434",
                **{k: v for k, v in self.extra.items() if k in (
                    "timeout", "num_ctx", "num_predict", "repeat_penalty", "top_k", "top_p"
                )}
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def to_client(self):
        """Create an LLMClient from this configuration.
        
        Returns:
            LLMClient instance
        """
        from .llm_client import LLMClient
        return LLMClient(self.to_llm_config())
    
    def is_configured(self) -> bool:
        """Check if the provider has minimum required configuration.
        
        Returns:
            True if provider appears to be properly configured
        """
        if not self.enabled:
            return False
        
        if self.provider in ("gemini", "openai"):
            return bool(self.api_key)
        elif self.provider == "azure-openai":
            return bool(self.api_key) and bool(self.azure_endpoint)
        elif self.provider == "ollama":
            # Ollama doesn't require API key
            return True
        
        return False
    
    def supports_intelligence_level(self, level: int) -> bool:
        """Check if this provider supports the given intelligence level.
        
        Args:
            level: Intelligence level (0-10)
            
        Returns:
            True if the provider's min/max range includes this level
        """
        return self.min_intelligence_level <= level <= self.max_intelligence_level
    
    def is_high_tier(self) -> bool:
        """Check if this is a high-tier (more capable) model."""
        return self.tier == "high"
    
    def is_low_tier(self) -> bool:
        """Check if this is a low-tier (cheaper/faster) model."""
        return self.tier == "low"


class ProviderRegistry:
    """Registry for managing multiple LLM provider configurations.
    
    Supports loading configurations from:
    - Environment variable (JSON list)
    - Environment variable prefix (individual vars per provider)
    - Configuration file (JSON/YAML)
    - Programmatic configuration
    """
    
    def __init__(self, providers: Optional[List[ProviderConfig]] = None):
        """Initialize the registry with optional provider configurations.
        
        Args:
            providers: List of ProviderConfig instances
        """
        self._providers: List[ProviderConfig] = providers or []
        self._clients_cache: Dict[int, Any] = {}
    
    def add(self, config: ProviderConfig) -> "ProviderRegistry":
        """Add a provider configuration.
        
        Args:
            config: ProviderConfig to add
            
        Returns:
            Self for chaining
        """
        self._providers.append(config)
        return self
    
    def add_from_dict(self, data: Dict[str, Any]) -> "ProviderRegistry":
        """Add a provider from a dictionary configuration.
        
        Args:
            data: Dictionary with provider configuration
            
        Returns:
            Self for chaining
        """
        return self.add(ProviderConfig.from_dict(data))
    
    @property
    def providers(self) -> List[ProviderConfig]:
        """Get all provider configurations sorted by priority."""
        return sorted(
            [p for p in self._providers if p.enabled and p.is_configured()],
            key=lambda p: p.priority
        )
    
    @property
    def all_providers(self) -> List[ProviderConfig]:
        """Get all provider configurations (including disabled/unconfigured)."""
        return list(self._providers)
    
    def get_primary(self) -> Optional[ProviderConfig]:
        """Get the primary (highest priority) provider configuration."""
        providers = self.providers
        return providers[0] if providers else None
    
    def get_fallbacks(self) -> List[ProviderConfig]:
        """Get all fallback provider configurations (excluding primary)."""
        providers = self.providers
        return providers[1:] if len(providers) > 1 else []
    
    def get_client(self, index: int = 0):
        """Get an LLMClient for the specified provider index.
        
        Args:
            index: Provider index (0 = primary, 1+ = fallbacks)
            
        Returns:
            LLMClient instance or None if index out of range
        """
        providers = self.providers
        if index >= len(providers):
            return None
        
        if index not in self._clients_cache:
            self._clients_cache[index] = providers[index].to_client()
        
        return self._clients_cache[index]
    
    def get_fallback_client(self):
        """Get the first fallback LLMClient.
        
        Returns:
            LLMClient for first fallback or None
        """
        return self.get_client(1)
    
    def iter_clients(self) -> Iterator[Tuple[Any, bool]]:
        """Iterate through all configured clients.
        
        Yields:
            Tuple of (LLMClient, is_fallback)
        """
        for i, _ in enumerate(self.providers):
            client = self.get_client(i)
            if client:
                yield (client, i > 0)
    
    def get_providers_for_level(self, intelligence_level: int) -> List[ProviderConfig]:
        """Get providers that support a specific intelligence level.
        
        Filters providers by their min/max intelligence level range and
        returns them sorted by priority.
        
        Args:
            intelligence_level: Intelligence level (0-10)
            
        Returns:
            List of ProviderConfig instances that support the level
        """
        matching = [
            p for p in self.providers
            if p.supports_intelligence_level(intelligence_level)
        ]
        return sorted(matching, key=lambda p: p.priority)
    
    def get_best_provider_for_level(self, intelligence_level: int) -> Optional[ProviderConfig]:
        """Get the best (highest priority) provider for an intelligence level.
        
        Args:
            intelligence_level: Intelligence level (0-10)
            
        Returns:
            Best matching ProviderConfig or None if no match
        """
        providers = self.get_providers_for_level(intelligence_level)
        return providers[0] if providers else None
    
    def get_client_for_level(self, intelligence_level: int):
        """Get an LLMClient for the specified intelligence level.
        
        Selects the best provider that supports the given level.
        
        Args:
            intelligence_level: Intelligence level (0-10)
            
        Returns:
            LLMClient instance or None if no provider supports the level
        """
        provider = self.get_best_provider_for_level(intelligence_level)
        if provider is None:
            # Fall back to primary provider if no level-specific match
            provider = self.get_primary()
        
        if provider is None:
            return None
        
        # Use a level-specific cache key
        cache_key = f"level_{intelligence_level}_{provider.provider}_{provider.model}"
        if cache_key not in self._clients_cache:
            self._clients_cache[cache_key] = provider.to_client()
        
        return self._clients_cache[cache_key]
    
    def iter_clients_for_level(self, intelligence_level: int) -> Iterator[Tuple[Any, bool, Dict[str, Any]]]:
        """Iterate through clients that support a specific intelligence level.
        
        Yields:
            Tuple of (LLMClient, is_fallback, provider_info)
        """
        providers = self.get_providers_for_level(intelligence_level)
        
        # If no level-specific providers, fall back to all providers
        if not providers:
            providers = self.providers
        
        for i, provider in enumerate(providers):
            cache_key = f"level_{intelligence_level}_{provider.provider}_{provider.model}"
            if cache_key not in self._clients_cache:
                self._clients_cache[cache_key] = provider.to_client()
            
            client = self._clients_cache[cache_key]
            if client:
                yield (
                    client,
                    i > 0,
                    {
                        "provider": provider.provider,
                        "model": provider.model,
                        "tier": provider.tier,
                        "intelligence_range": (provider.min_intelligence_level, provider.max_intelligence_level),
                    }
                )
    
    def get_low_tier_providers(self) -> List[ProviderConfig]:
        """Get all low-tier (cheaper/faster) providers."""
        return [p for p in self.providers if p.is_low_tier()]
    
    def get_high_tier_providers(self) -> List[ProviderConfig]:
        """Get all high-tier (more capable) providers."""
        return [p for p in self.providers if p.is_high_tier()]
    
    def clear_cache(self) -> None:
        """Clear cached client instances."""
        for client in self._clients_cache.values():
            try:
                close_method = getattr(client, "close", None)
                if callable(close_method):
                    close_method()
            except Exception:
                pass
        self._clients_cache.clear()
    
    @classmethod
    def from_env(cls, env_var: str = "LLM_PROVIDERS") -> "ProviderRegistry":
        """Load provider configurations from environment.
        
        Checks for:
        1. JSON list in the specified env var
        2. Legacy individual environment variables
        
        Args:
            env_var: Environment variable name containing JSON config
            
        Returns:
            ProviderRegistry instance
        """
        registry = cls()
        
        # Try to load from JSON env var
        json_config = os.getenv(env_var)
        if json_config:
            try:
                providers_data = json.loads(json_config)
                if isinstance(providers_data, list):
                    for p in providers_data:
                        registry.add_from_dict(p)
                    logger.debug(f"Loaded {len(providers_data)} providers from {env_var}")
                    return registry
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse {env_var} as JSON: {e}")
        
        # Fall back to individual environment variables
        registry._load_legacy_env_vars()
        
        return registry
    
    def _load_legacy_env_vars(self) -> None:
        """Load providers from legacy individual environment variables."""
        
        # Check for Gemini
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
        if gemini_key:
            self.add(ProviderConfig(
                provider="gemini",
                api_key=gemini_key,
                model=os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_GENAI_MODEL") or "gemini-2.0-flash",
                temperature=float(os.getenv("GEMINI_TEMPERATURE") or "0.7"),
                priority=10,
            ))
        
        # Check for OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if azure_key:
            self.add(ProviderConfig(
                provider="azure-openai",
                api_key=azure_key,
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_MODEL") or "gpt-4o-mini",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE") or "0.7"),
                priority=20,
            ))
        elif openai_key:
            self.add(ProviderConfig(
                provider="openai",
                api_key=openai_key,
                model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
                host=os.getenv("OPENAI_BASE_URL"),
                organization=os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION"),
                project=os.getenv("OPENAI_PROJECT"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE") or "0.7"),
                priority=20,
            ))
        
        # Check for Ollama
        ollama_host = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
        if ollama_host or os.getenv("OLLAMA_MODEL"):
            self.add(ProviderConfig(
                provider="ollama",
                model=os.getenv("OLLAMA_MODEL") or "llama3.2",
                host=ollama_host or "http://localhost:11434",
                temperature=float(os.getenv("OLLAMA_TEMPERATURE") or "0.7"),
                priority=30,
            ))
    
    @classmethod
    def from_file(cls, path: str) -> "ProviderRegistry":
        """Load provider configurations from a file.
        
        Supports JSON and YAML formats.
        
        Args:
            path: Path to configuration file
            
        Returns:
            ProviderRegistry instance
        """
        import os.path
        
        registry = cls()
        
        if not os.path.exists(path):
            logger.warning(f"Configuration file not found: {path}")
            return registry
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Determine format
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                logger.error("PyYAML not installed, cannot load YAML config")
                return registry
        else:
            data = json.loads(content)
        
        # Handle different config structures
        providers_list = data if isinstance(data, list) else data.get("providers", data.get("llm_providers", []))
        
        for p in providers_list:
            registry.add_from_dict(p)
        
        logger.debug(f"Loaded {len(providers_list)} providers from {path}")
        return registry
    
    @classmethod
    def from_list(cls, providers: List[Dict[str, Any]]) -> "ProviderRegistry":
        """Create registry from a list of provider dictionaries.
        
        Args:
            providers: List of provider configuration dictionaries
            
        Returns:
            ProviderRegistry instance
        """
        registry = cls()
        for p in providers:
            registry.add_from_dict(p)
        return registry
    
    def __len__(self) -> int:
        return len(self.providers)
    
    def __bool__(self) -> bool:
        return len(self.providers) > 0
    
    def __enter__(self) -> "ProviderRegistry":
        return self
    
    def __exit__(self, exc_type, exc, tb) -> bool:
        self.clear_cache()
        return False
