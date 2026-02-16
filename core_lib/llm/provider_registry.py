"""Provider registry for managing multiple LLM provider configurations.

This module provides a flexible way to configure multiple LLM providers with
their credentials, models, and settings. It supports:
- Loading from environment variables (JSON or individual vars)
- Loading from configuration files (JSON/YAML) with environment variable substitution
- Programmatic configuration via dictionaries
- Automatic fallback selection when primary provider fails
- Health-aware provider selection with automatic recovery

Configuration Options (in priority order):
    1. LLM_PROVIDERS_FILE: Path to YAML/JSON configuration file
    2. LLM_PROVIDERS: JSON array in environment variable
    3. Individual env vars: GEMINI_API_KEY, OPENAI_API_KEY, etc.

Example YAML configuration file (llm_providers.yaml):
    providers:
      - provider: gemini
        api_key: ${GEMINI_API_KEY}  # Environment variable substitution
        model: gemini-2.0-flash
        priority: 1
        
      - provider: ollama
        host: ${OLLAMA_HOST:-http://localhost:11434}  # With default value
        model: llama3.2
        priority: 2

Example environment variable (LLM_PROVIDERS):
    [
        {"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash"},
        {"provider": "openai", "api_key": "sk-...", "model": "gpt-4o-mini"},
        {"provider": "ollama", "host": "http://localhost:11434", "model": "llama3.2"}
    ]

Example usage:
    from core_lib.llm.provider_registry import ProviderRegistry
    
    # Load from environment (auto-detects file or env var)
    registry = ProviderRegistry.from_env()
    
    # Get primary provider client
    client = registry.get_client()
    
    # Get fallback client (next available)
    fallback = registry.get_fallback_client()
    
    # Health-aware iteration with automatic fallback
    for client, is_fallback in registry.iter_clients_with_fallback():
        try:
            response = client.chat(messages)
            registry.mark_healthy(client)  # Success - mark as healthy
            break
        except Exception as e:
            registry.mark_unhealthy(client, error=e)  # Failure - mark for recovery
            continue
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

from core_lib import get_module_logger

logger = get_module_logger()


# Environment variable substitution pattern: ${VAR} or ${VAR:-default}
_ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::-([^}]*))?\}')


def substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in a value.
    
    Supports patterns:
        ${VAR_NAME}           - Required variable
        ${VAR_NAME:-default}  - Variable with default value
    
    Args:
        value: Value to process (str, dict, list, or other)
        
    Returns:
        Value with environment variables substituted
    
    Example:
        >>> os.environ["API_KEY"] = "secret123"
        >>> substitute_env_vars("${API_KEY}")
        'secret123'
        >>> substitute_env_vars("${MISSING:-default_value}")
        'default_value'
        >>> substitute_env_vars({"key": "${API_KEY}", "host": "${HOST:-localhost}"})
        {'key': 'secret123', 'host': 'localhost'}
    """
    if isinstance(value, str):
        def replace_match(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2)  # May be None
            
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                # Return empty string for missing required vars (log warning)
                logger.warning(f"Environment variable {var_name} not set and no default provided")
                return ""
        
        return _ENV_VAR_PATTERN.sub(replace_match, value)
    
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    
    else:
        return value


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider.
    
    This is a provider-agnostic configuration that can be converted to
    provider-specific configs (GeminiConfig, OpenAIConfig, OllamaConfig).
    
    Attributes:
        provider: Provider name (gemini, vertex, openai, azure-openai, ollama)
        model: Model name/identifier
        api_key: API key (for cloud providers)
        host: Base URL/host (for Ollama or custom endpoints)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        thinking_enabled: Enable step-by-step thinking
        thinking_config: Optional provider-specific thinking configuration
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
    thinking_config: Optional[Dict[str, Any]] = None
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
    location: Optional[str] = None
    service_account_file: Optional[str] = None
    _missing_service_account_logged: ClassVar[Set[str]] = set()
    
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
                "vertex": "gemini-2.0-flash",
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
        thinking = data.get("thinking")
        if thinking is None and ("thinking_enabled" in data or "thinkingEnabled" in data):
            thinking = data.get("thinking_enabled", data.get("thinkingEnabled"))

        thinking_level = data.get("thinking_level") if "thinking_level" in data else data.get("thinkingLevel")
        thinking_budget = data.get("thinking_budget") if "thinking_budget" in data else data.get("thinkingBudget")
        include_thoughts = data.get("include_thoughts") if "include_thoughts" in data else data.get("includeThoughts")

        thinking_cfg: Dict[str, Any] = {}

        if isinstance(thinking, dict):
            thinking_cfg.update(thinking)
            if "enabled" in thinking:
                normalized["thinking_enabled"] = bool(thinking.get("enabled"))
        elif isinstance(thinking, (int, float)):
            budget = int(thinking)
            thinking_cfg["budget"] = budget
            normalized["thinking_enabled"] = budget > 0
        elif isinstance(thinking, str):
            thinking_lc = thinking.lower().strip()
            if thinking_lc in {"true", "false"}:
                normalized["thinking_enabled"] = thinking_lc == "true"
            else:
                thinking_cfg["level"] = thinking_lc
                normalized["thinking_enabled"] = thinking_lc not in {"off", "none", "disabled", "disable", "0"}
        elif thinking is not None:
            normalized["thinking_enabled"] = bool(thinking)

        if thinking_level is not None:
            thinking_cfg["level"] = str(thinking_level).lower()
        if thinking_budget is not None:
            thinking_cfg["budget"] = int(thinking_budget)
        if include_thoughts is not None:
            thinking_cfg["include_thoughts"] = bool(include_thoughts)

        if thinking_cfg:
            normalized["thinking_config"] = thinking_cfg
        
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
        normalized["location"] = data.get("location") or data.get("region")
        normalized["service_account_file"] = (
            data.get("service_account_file") or 
            data.get("serviceAccountFile") or 
            data.get("credentials_file") or
            data.get("google_application_credentials")
        )
        
        # Collect remaining keys as extra
        known_keys = {
            "provider", "type", "model", "model_name", "api_key", "apiKey", "key",
            "host", "base_url", "baseUrl", "endpoint", "temperature", "max_tokens",
            "maxTokens", "thinking_enabled", "thinkingEnabled", "thinking", "thinking_config",
            "thinkingConfig", "thinking_level", "thinkingLevel", "thinking_budget", "thinkingBudget",
            "include_thoughts", "includeThoughts", "priority",
            "enabled", "azure_endpoint", "azureEndpoint", "azure_api_version",
            "azureApiVersion", "organization", "org", "project", "location", "region",
            "service_account_file", "serviceAccountFile", "credentials_file", "google_application_credentials"
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
            api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
            return GeminiConfig(
                api_key=api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_enabled=self.thinking_enabled,
                thinking_config=self.thinking_config,
                base_url=self.host or "https://generativelanguage.googleapis.com",
                project=None,
                location=None,
                service_account_file=None,
            )

        elif self.provider == "vertex":
            from .providers.google_genai_provider import GeminiConfig
            return GeminiConfig(
                api_key=None,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_enabled=self.thinking_enabled,
                thinking_config=self.thinking_config,
                base_url=self.host or "https://generativelanguage.googleapis.com",
                project=self.project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT_ID"),
                location=self.location or os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_CLOUD_REGION"),
                service_account_file=self.service_account_file or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
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
                thinking_config=self.thinking_config,
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
        
        if self.provider == "gemini":
            # AI Studio (API Key) only
            return bool(self.api_key) or bool(os.getenv("GEMINI_API_KEY")) or bool(os.getenv("GOOGLE_GENAI_API_KEY"))

        if self.provider == "vertex":
            has_project = bool(self.project) or bool(os.getenv("GOOGLE_CLOUD_PROJECT")) or bool(os.getenv("GOOGLE_PROJECT_ID"))
            has_location = bool(self.location) or bool(os.getenv("GOOGLE_CLOUD_LOCATION")) or bool(os.getenv("GOOGLE_CLOUD_REGION"))
            service_account = self.service_account_file or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

            if not service_account:
                return has_project and has_location and False

            normalized_path = str(service_account).strip().strip('"').strip("'")
            expanded_path = os.path.expanduser(normalized_path)
            has_service_account = os.path.isfile(expanded_path)

            if not has_service_account:
                cache_key = f"vertex:{expanded_path}"
                if cache_key not in self._missing_service_account_logged:
                    logger.error(
                        "Vertex provider disabled: GOOGLE_APPLICATION_CREDENTIALS file not found: %s",
                        expanded_path,
                    )
                    self._missing_service_account_logged.add(cache_key)

            return has_project and has_location and has_service_account
            
        elif self.provider == "openai":
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
    def from_env(
        cls,
        env_var: str = "LLM_PROVIDERS",
        file_env_var: str = "LLM_PROVIDERS_FILE",
    ) -> "ProviderRegistry":
        """Load provider configurations from environment.
        
        Checks for (in priority order):
        1. Configuration file path in LLM_PROVIDERS_FILE
        2. JSON list in the LLM_PROVIDERS env var
        3. Legacy individual environment variables (GEMINI_API_KEY, etc.)
        
        Args:
            env_var: Environment variable name containing JSON config
            file_env_var: Environment variable with path to config file
            
        Returns:
            ProviderRegistry instance
        """
        registry = cls()
        
        # Priority 1: Check for config file path
        config_file = os.getenv(file_env_var)
        if config_file:
            # Resolve relative paths
            if not os.path.isabs(config_file):
                # Try current directory first, then common config locations
                for base in [os.getcwd(), os.path.dirname(__file__), "/etc/llm", "~/.config/llm"]:
                    candidate = os.path.expanduser(os.path.join(base, config_file))
                    if os.path.exists(candidate):
                        config_file = candidate
                        break
            
            if os.path.exists(config_file):
                registry = cls.from_file(config_file, substitute_env=True)
                if registry:
                    logger.debug(f"Loaded {len(registry)} providers from {config_file}")
                    return registry
                else:
                    logger.warning(f"Config file {config_file} loaded but no providers found")
            else:
                logger.warning(f"Config file not found: {config_file}")
        
        # Priority 2: Try to load from JSON env var
        json_config = os.getenv(env_var)
        if json_config:
            try:
                providers_data = json.loads(json_config)
                if isinstance(providers_data, list):
                    # Apply environment variable substitution
                    providers_data = substitute_env_vars(providers_data)
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
    def from_file(cls, path: str, substitute_env: bool = True) -> "ProviderRegistry":
        """Load provider configurations from a file.
        
        Supports JSON and YAML formats with optional environment variable
        substitution using ${VAR_NAME} or ${VAR_NAME:-default} syntax.
        
        Args:
            path: Path to configuration file
            substitute_env: Whether to substitute environment variables in values
            
        Returns:
            ProviderRegistry instance
        """
        import os.path as ospath
        
        registry = cls()
        
        if not ospath.exists(path):
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
        
        # Apply environment variable substitution if enabled
        if substitute_env:
            data = substitute_env_vars(data)
        
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
    
    # -------------------------------------------------------------------------
    # Health-Aware Provider Selection
    # -------------------------------------------------------------------------
    
    def _get_health_tracker(self):
        """Get the health tracker instance (lazy import to avoid circular deps)."""
        from .provider_health import get_health_tracker
        return get_health_tracker()
    
    def get_healthy_providers(
        self,
        intelligence_level: Optional[int] = None,
    ) -> List[ProviderConfig]:
        """Get all healthy providers, optionally filtered by intelligence level.
        
        Args:
            intelligence_level: Optional intelligence level filter (0-10)
            
        Returns:
            List of healthy ProviderConfig instances sorted by priority
        """
        if intelligence_level is not None:
            providers = self.get_providers_for_level(intelligence_level)
        else:
            providers = self.providers
        
        tracker = self._get_health_tracker()
        return tracker.filter_healthy(providers)
    
    def get_healthy_client(
        self,
        intelligence_level: Optional[int] = None,
    ):
        """Get the first healthy LLM client.
        
        Selects the highest priority healthy provider. If no providers are
        healthy, falls back to the primary provider anyway.
        
        Args:
            intelligence_level: Optional intelligence level filter
            
        Returns:
            LLMClient instance
        """
        healthy_providers = self.get_healthy_providers(intelligence_level)
        
        if healthy_providers:
            provider = healthy_providers[0]
            cache_key = f"healthy_{provider.provider}_{provider.model}"
            if cache_key not in self._clients_cache:
                self._clients_cache[cache_key] = provider.to_client()
            return self._clients_cache[cache_key]
        
        # All unhealthy - fall back to primary
        logger.warning("All providers unhealthy, falling back to primary")
        return self.get_client(0)
    
    def iter_clients_with_fallback(
        self,
        intelligence_level: Optional[int] = None,
    ) -> Iterator[Tuple[Any, bool, ProviderConfig]]:
        """Iterate through clients in health-aware order.
        
        Yields healthy providers first, then unhealthy ones as last resort.
        
        Args:
            intelligence_level: Optional intelligence level filter
            
        Yields:
            Tuple of (LLMClient, is_fallback, ProviderConfig)
        """
        if intelligence_level is not None:
            all_providers = self.get_providers_for_level(intelligence_level)
            if not all_providers:
                all_providers = self.providers
        else:
            all_providers = self.providers
        
        tracker = self._get_health_tracker()
        healthy = tracker.filter_healthy(all_providers)
        unhealthy = [p for p in all_providers if p not in healthy]
        
        # Yield healthy providers first
        for i, provider in enumerate(healthy):
            cache_key = f"fallback_{provider.provider}_{provider.model}"
            if cache_key not in self._clients_cache:
                self._clients_cache[cache_key] = provider.to_client()
            yield (self._clients_cache[cache_key], i > 0, provider)
        
        # Then yield unhealthy providers as last resort
        for provider in unhealthy:
            cache_key = f"fallback_{provider.provider}_{provider.model}"
            if cache_key not in self._clients_cache:
                self._clients_cache[cache_key] = provider.to_client()
            yield (self._clients_cache[cache_key], True, provider)
    
    def mark_healthy(
        self,
        client_or_provider: Union[Any, ProviderConfig, str],
        model: Optional[str] = None,
    ) -> None:
        """Mark a provider as healthy after successful request.
        
        Args:
            client_or_provider: LLMClient, ProviderConfig, or provider name string
            model: Model name (required if passing provider name string)
        """
        provider_name, model_name = self._resolve_provider_model(client_or_provider, model)
        if provider_name and model_name:
            tracker = self._get_health_tracker()
            tracker.mark_healthy(provider_name, model_name)
    
    def mark_unhealthy(
        self,
        client_or_provider: Union[Any, ProviderConfig, str],
        model: Optional[str] = None,
        error: Optional[Exception] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Mark a provider as unhealthy after failure.
        
        Args:
            client_or_provider: LLMClient, ProviderConfig, or provider name string
            model: Model name (required if passing provider name string)
            error: Optional exception that caused the failure
            reason: Optional explicit reason string
        """
        provider_name, model_name = self._resolve_provider_model(client_or_provider, model)
        if provider_name and model_name:
            tracker = self._get_health_tracker()
            
            # Classify error if not explicitly provided
            if reason is None and error is not None:
                from .provider_health import classify_error
                reason = classify_error(error)
            elif reason is None:
                reason = "unknown"
            
            tracker.mark_unhealthy(provider_name, model_name, reason=reason)
    
    def _resolve_provider_model(
        self,
        client_or_provider: Union[Any, ProviderConfig, str],
        model: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve provider and model names from various input types."""
        if isinstance(client_or_provider, ProviderConfig):
            return client_or_provider.provider, client_or_provider.model
        
        if isinstance(client_or_provider, str):
            return client_or_provider, model
        
        # Try to extract from LLMClient
        try:
            config = getattr(client_or_provider, "config", None)
            if config:
                return getattr(config, "provider", None), getattr(config, "model", None)
        except Exception:
            pass
        
        return None, None
    
    def __len__(self) -> int:
        return len(self.providers)
    
    def __bool__(self) -> bool:
        return len(self.providers) > 0
    
    def __enter__(self) -> "ProviderRegistry":
        return self
    
    def __exit__(self, exc_type, exc, tb) -> bool:
        self.clear_cache()
        return False
