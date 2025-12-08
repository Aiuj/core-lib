"""Configuration helpers for core_lib.

Expose configuration classes and category constants for easy import.
"""
from .doc_categories import DOC_CATEGORIES, DOC_CATEGORIES_BY_KEY, DOC_CATEGORY_CHOICES
from .industry_categories import INDUSTRY_CATEGORIES, INDUSTRY_CATEGORIES_BY_KEY, INDUSTRY_CATEGORY_CHOICES
from .confidentiality_levels import (
    CONFIDENTIALITY_LEVELS,
    CONFIDENTIALITY_LEVEL_NAMES,
    DEFAULT_CONFIDENTIALITY_LEVEL,
    CONFIDENTIALITY_LEVEL_DESCRIPTION,
    validate_confidentiality_level,
    get_confidentiality_level_name,
    get_confidentiality_level_value,
)
from .base_settings import (
    BaseSettings, 
    SettingsError, 
    EnvironmentVariableError, 
    SettingsManager, 
    settings_manager,
    EnvParser,
    DotEnvLoader
)
from .app_settings import AppSettings
from .api_settings import ApiSettings
from .llm_settings import LLMSettings
from .embeddings_settings import EmbeddingsSettings
from .cache_settings import CacheSettings
from .tracing_settings import TracingSettings
from .logger_settings import LoggerSettings
from .database_settings import DatabaseSettings
from .opensearch_settings import OpenSearchSettings
from .mcp_settings import MCPServerSettings
from .fastapi_settings import FastAPIServerSettings
from .standard_settings import StandardSettings
from .settings_singleton import (
    SettingsSingletonManager,
    initialize_settings,
    get_settings,
    set_settings,
    reset_settings,
    has_settings,
    get_settings_safe,
)

# Import AuthSettings from api_utils
from ..api_utils.auth_settings import AuthSettings

__all__ = [
    # Legacy doc categories
    "DOC_CATEGORIES", 
    "DOC_CATEGORIES_BY_KEY", 
    "DOC_CATEGORY_CHOICES",
    
    # Confidentiality levels
    "CONFIDENTIALITY_LEVELS",
    "CONFIDENTIALITY_LEVEL_NAMES",
    "DEFAULT_CONFIDENTIALITY_LEVEL",
    "CONFIDENTIALITY_LEVEL_DESCRIPTION",
    "validate_confidentiality_level",
    "get_confidentiality_level_name",
    "get_confidentiality_level_value",
    
    # New settings system
    "BaseSettings",
    "SettingsError",
    "EnvironmentVariableError", 
    "SettingsManager",
    "settings_manager",
    "EnvParser",
    "DotEnvLoader",
    "AppSettings",
    "ApiSettings",
    "EmbeddingsSettings", 
    "CacheSettings",
    "TracingSettings",
    "LoggerSettings",
    "DatabaseSettings",
    "DatabaseSettings",
    "OpenSearchSettings",
    "MCPServerSettings",
    "FastAPIServerSettings",
    "StandardSettings",
    "AuthSettings",
    
    # Settings singleton
    "SettingsSingletonManager",
    "initialize_settings",
    "get_settings",
    "set_settings",
    "reset_settings",
    "has_settings",
    "get_settings_safe",
]
