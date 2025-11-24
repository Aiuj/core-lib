"""core-lib: Shared library for MCP agent tools."""

__version__ = "0.2.9"

from .cache import (
    create_cache, 
    set_cache, get_cache, cache_get, cache_set
)
from .jobs import (
    create_job_queue, set_job_queue, get_job_queue,
    submit_job, get_job_status, get_job_result,
    update_job_status, update_job_progress,
    complete_job, fail_job, cancel_job,
    list_jobs, cleanup_old_jobs,
    BaseJobQueue, JobConfig, JobStatus, Job,
    JobWorker, JobHandler
)
from .mcp_utils import get_transport_from_args
from .tracing import setup_tracing, setup_logging, get_logger, get_module_logger, get_last_logging_config, FROM_FIELD_DESCRIPTION, INTELLIGENCE_LEVEL_DESCRIPTION, LoggingContext, parse_from
from .llm import (
    LLMClient, LLMConfig, GeminiConfig, OllamaConfig, OpenAIConfig,
    create_llm_client, create_gemini_client, create_ollama_client, 
    create_openai_client, create_client_from_env, clean_and_parse_json_response
)
from .utils import LanguageUtils, HealthChecker, HealthCheckResult, HealthStatus, create_lazy_check
from .tools import ExcelManager

# New settings management system
from .config import (
    StandardSettings, BaseSettings, LLMSettings, EmbeddingsSettings, AppSettings,
    CacheSettings, TracingSettings, DatabaseSettings, SettingsManager, settings_manager,
    SettingsError, EnvironmentVariableError, AuthSettings,
    DOC_CATEGORIES, DOC_CATEGORIES_BY_KEY, DOC_CATEGORY_CHOICES,
    CONFIDENTIALITY_LEVELS, CONFIDENTIALITY_LEVEL_NAMES, DEFAULT_CONFIDENTIALITY_LEVEL,
    CONFIDENTIALITY_LEVEL_DESCRIPTION, validate_confidentiality_level,
    get_confidentiality_level_name, get_confidentiality_level_value
)

# API utilities for time-based authentication
from .api_utils import (
    generate_time_key, verify_time_key, TimeBasedAuthError,
    verify_mcp_auth, get_auth_headers, get_auth_env_vars, MCPAuthError,
    APIClient, run_uvicorn_server, run_uvicorn_from_settings
)

__all__ = [
    # Cache
    "create_cache",
    "set_cache", 
    "get_cache", 
    "cache_get",
    "cache_set",
    
    # Jobs
    "create_job_queue",
    "set_job_queue",
    "get_job_queue",
    "submit_job",
    "get_job_status",
    "get_job_result",
    "update_job_status",
    "update_job_progress",
    "complete_job",
    "fail_job",
    "cancel_job",
    "list_jobs",
    "cleanup_old_jobs",
    "BaseJobQueue",
    "JobConfig",
    "JobStatus",
    "Job",
    "JobWorker",
    "JobHandler",
    
    # MCP Utils
    "parse_from",
    "get_transport_from_args",
    
    # Tracing & logging
    "setup_tracing",
    "setup_logging",
    "get_logger",
    "get_module_logger",
    "get_last_logging_config",
    "FROM_FIELD_DESCRIPTION",
    "INTELLIGENCE_LEVEL_DESCRIPTION",
    "LoggingContext",
    
    # LLM
    "LLMClient",
    "LLMConfig",
    "GeminiConfig",
    "OllamaConfig",
    "OpenAIConfig",
    "create_llm_client",
    "create_gemini_client",
    "create_ollama_client",
    "create_openai_client",
    "create_client_from_env",
    "clean_and_parse_json_response",
    
    # Utils
    "LanguageUtils",
    "AppSettings",
    "HealthChecker",
    "HealthCheckResult",
    "HealthStatus",
    "create_lazy_check",
    
    # Tools
    "ExcelManager",
    
    # Settings management system
    "StandardSettings",
    "BaseSettings", 
    "LLMSettings",
    "EmbeddingsSettings",
    "CacheSettings",
    "TracingSettings",
    "DatabaseSettings",
    "AuthSettings",
    "SettingsManager",
    "settings_manager",
    "SettingsError",
    "EnvironmentVariableError",
    "DOC_CATEGORIES",
    "DOC_CATEGORIES_BY_KEY",
    "DOC_CATEGORY_CHOICES",
    "CONFIDENTIALITY_LEVELS",
    "CONFIDENTIALITY_LEVEL_NAMES",
    "DEFAULT_CONFIDENTIALITY_LEVEL",
    "CONFIDENTIALITY_LEVEL_DESCRIPTION",
    "validate_confidentiality_level",
    "get_confidentiality_level_name",
    "get_confidentiality_level_value",
    
    # API utilities for authentication
    "generate_time_key",
    "verify_time_key",
    "TimeBasedAuthError",
    "verify_mcp_auth",
    "get_auth_headers",
    "get_auth_env_vars",
    "MCPAuthError",
    "APIClient",
    "run_uvicorn_server",
    "run_uvicorn_from_settings",
    
    "__version__",
]

