"""
Test suite for the settings management system.

Tests cover:
- BaseSettings abstract functionality
- Individual settings classes (LLM, embeddings, cache, tracing)
- StandardSettings comprehensive configuration
- Environment variable parsing and type conversion
- .env file loading and discovery
- Settings validation and error handling
- Settings manager functionality
- Integration with existing configuration classes
"""

import os
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union
from unittest.mock import patch, Mock

import pytest

from core_lib.config.base_settings import (
    BaseSettings, SettingsError, EnvironmentVariableError,
    EnvParser, DotEnvLoader, SettingsManager, settings_manager
)
from core_lib.config.standard_settings import (
    StandardSettings, LLMSettings, EmbeddingsSettings, 
    CacheSettings, TracingSettings, DatabaseSettings
)


class TestEnvParser(unittest.TestCase):
    """Test environment variable parsing utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        # Clear test env vars
        test_vars = ["TEST_STR", "TEST_INT", "TEST_FLOAT", "TEST_BOOL", "TEST_LIST", "TEST_MISSING"]
        for var in test_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_get_env_string_default(self):
        """Test string parsing with default."""
        result = EnvParser.get_env("TEST_MISSING", default="default_value")
        self.assertEqual(result, "default_value")
        
        os.environ["TEST_STR"] = "hello world"
        result = EnvParser.get_env("TEST_STR", default="default")
        self.assertEqual(result, "hello world")
    
    def test_get_env_multiple_names(self):
        """Test fallback behavior with multiple names."""
        os.environ["TEST_FALLBACK"] = "fallback_value"
        result = EnvParser.get_env("TEST_MISSING", "TEST_FALLBACK", default="default")
        self.assertEqual(result, "fallback_value")
    
    def test_get_env_integer_conversion(self):
        """Test integer type conversion."""
        os.environ["TEST_INT"] = "42"
        result = EnvParser.get_env("TEST_INT", env_type=int)
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)
        
        # Test invalid integer
        os.environ["TEST_INT"] = "not_a_number"
        with self.assertRaises(EnvironmentVariableError):
            EnvParser.get_env("TEST_INT", env_type=int)
    
    def test_get_env_float_conversion(self):
        """Test float type conversion."""
        os.environ["TEST_FLOAT"] = "3.14"
        result = EnvParser.get_env("TEST_FLOAT", env_type=float)
        self.assertEqual(result, 3.14)
        self.assertIsInstance(result, float)
    
    def test_get_env_boolean_conversion(self):
        """Test boolean type conversion."""
        true_values = ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]
        false_values = ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF", ""]
        
        for value in true_values:
            os.environ["TEST_BOOL"] = value
            result = EnvParser.get_env("TEST_BOOL", env_type=bool)
            self.assertTrue(result, f"'{value}' should be True")
        
        for value in false_values:
            os.environ["TEST_BOOL"] = value
            result = EnvParser.get_env("TEST_BOOL", env_type=bool)
            self.assertFalse(result, f"'{value}' should be False")
    
    def test_get_env_list_conversion(self):
        """Test list type conversion."""
        os.environ["TEST_LIST"] = "item1,item2,item3"
        result = EnvParser.get_env("TEST_LIST", env_type=list)
        self.assertEqual(result, ["item1", "item2", "item3"])
        
        # Test with spaces
        os.environ["TEST_LIST"] = "item1, item2 , item3"
        result = EnvParser.get_env("TEST_LIST", env_type=list)
        self.assertEqual(result, ["item1", "item2", "item3"])
        
        # Test empty list
        os.environ["TEST_LIST"] = ""
        result = EnvParser.get_env("TEST_LIST", env_type=list)
        self.assertEqual(result, [])
    
    def test_get_env_required(self):
        """Test required parameter behavior."""
        # Should raise when required and missing
        with self.assertRaises(EnvironmentVariableError) as cm:
            EnvParser.get_env("TEST_MISSING", required=True)
        self.assertIn("Required environment variable not found", str(cm.exception))
        
        # Should work when required and present
        os.environ["TEST_STR"] = "value"
        result = EnvParser.get_env("TEST_STR", required=True)
        self.assertEqual(result, "value")


class TestDotEnvLoader(unittest.TestCase):
    """Test .env file loading functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Save original env
        self.original_env = dict(os.environ)
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original env
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('core_lib.config.base_settings.HAS_DOTENV', True)
    @patch('core_lib.config.base_settings.load_dotenv')
    def test_load_dotenv_files_success(self, mock_load_dotenv):
        """Test successful .env file loading."""
        # Create .env file
        env_file = self.temp_path / ".env"
        env_file.write_text("TEST_VAR=test_value\n")
        
        # Test loading
        result = DotEnvLoader.load_dotenv_files([self.temp_path])
        
        self.assertTrue(result)
        mock_load_dotenv.assert_called_once_with(env_file, override=False)
    
    @patch('core_lib.config.base_settings.HAS_DOTENV', False)
    def test_load_dotenv_files_no_dotenv(self):
        """Test behavior when python-dotenv is not available."""
        result = DotEnvLoader.load_dotenv_files([self.temp_path])
        self.assertFalse(result)
    
    @patch('core_lib.config.base_settings.HAS_DOTENV', True)
    def test_load_dotenv_files_missing_file(self):
        """Test behavior when .env file doesn't exist."""
        result = DotEnvLoader.load_dotenv_files([self.temp_path / "nonexistent"])
        self.assertFalse(result)
    
    def test_get_default_search_paths(self):
        """Test default search path generation."""
        paths = DotEnvLoader._get_default_search_paths()
        
        self.assertIsInstance(paths, list)
        self.assertTrue(len(paths) >= 2)  # At least CWD and home
        self.assertIn(Path.cwd(), paths)
        self.assertIn(Path.home(), paths)


@dataclass(frozen=True)
class CustomTestSettings(BaseSettings):
    """Test settings class for testing BaseSettings functionality."""
    
    name: str = "test"
    count: int = 10
    enabled: bool = True
    items: Optional[List[str]] = None
    
    @classmethod
    def from_env(cls, load_dotenv=True, dotenv_paths=None, **overrides):
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        settings_dict = {
            "name": EnvParser.get_env("TEST_NAME", default="test"),
            "count": EnvParser.get_env("TEST_COUNT", default=10, env_type=int),
            "enabled": EnvParser.get_env("TEST_ENABLED", default=True, env_type=bool),
            "items": EnvParser.get_env("TEST_ITEMS", env_type=list),
        }
        
        settings_dict.update(overrides)
        return cls(**settings_dict)
    
    def validate(self):
        """Test validation."""
        if self.count < 0:
            raise SettingsError("Count must be non-negative")


class TestBaseSettings(unittest.TestCase):
    """Test BaseSettings abstract base class functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        test_vars = ["TEST_NAME", "TEST_COUNT", "TEST_ENABLED", "TEST_ITEMS"]
        for var in test_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_from_env_basic(self):
        """Test basic from_env functionality."""
        settings = CustomTestSettings.from_env()
        
        self.assertEqual(settings.name, "test")
        self.assertEqual(settings.count, 10)
        self.assertTrue(settings.enabled)
        self.assertIsNone(settings.items)
    
    def test_from_env_with_environment_variables(self):
        """Test from_env with environment variables."""
        os.environ["TEST_NAME"] = "custom"
        os.environ["TEST_COUNT"] = "42"
        os.environ["TEST_ENABLED"] = "false"
        os.environ["TEST_ITEMS"] = "a,b,c"
        
        settings = CustomTestSettings.from_env()
        
        self.assertEqual(settings.name, "custom")
        self.assertEqual(settings.count, 42)
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.items, ["a", "b", "c"])
    
    def test_from_env_with_overrides(self):
        """Test from_env with direct overrides."""
        os.environ["TEST_NAME"] = "env_name"
        
        settings = CustomTestSettings.from_env(name="override_name", count=99)
        
        self.assertEqual(settings.name, "override_name")  # Override wins
        self.assertEqual(settings.count, 99)  # Override wins
    
    def test_validation_success(self):
        """Test successful validation."""
        settings = CustomTestSettings(name="test", count=5)
        self.assertTrue(settings.is_valid)
        self.assertEqual(len(settings.validation_errors), 0)
    
    def test_validation_failure(self):
        """Test validation failure."""
        settings = CustomTestSettings(name="test", count=-1)
        self.assertFalse(settings.is_valid)
        self.assertEqual(len(settings.validation_errors), 1)
        self.assertIn("Count must be non-negative", settings.validation_errors[0])
    
    def test_as_dict(self):
        """Test as_dict functionality."""
        settings = CustomTestSettings(name="test", count=42, items=["a", "b"])
        result = settings.as_dict()
        
        expected = {
            "name": "test",
            "count": 42,
            "enabled": True,
            "items": ["a", "b"]
        }
        self.assertEqual(result, expected)
    
    def test_merge(self):
        """Test settings merging."""
        original = CustomTestSettings(name="original", count=10)
        merged = original.merge(name="merged", count=20)
        
        self.assertEqual(original.name, "original")  # Original unchanged
        self.assertEqual(original.count, 10)
        
        self.assertEqual(merged.name, "merged")  # New instance with changes
        self.assertEqual(merged.count, 20)


class TestLLMSettings(unittest.TestCase):
    """Test LLM-specific settings."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        # Clear LLM-related env vars
        llm_vars = [
            "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
            "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
            "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "OLLAMA_HOST", "OLLAMA_BASE_URL"
        ]
        for var in llm_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_llm_settings_defaults(self):
        """Test LLM settings with defaults."""
        settings = LLMSettings.from_env(load_dotenv=False)
        
        self.assertEqual(settings.provider, "openai")
        self.assertEqual(settings.model, "gpt-4o-mini")
        self.assertEqual(settings.temperature, 0.7)
        self.assertIsNone(settings.max_tokens)
        self.assertFalse(settings.thinking_enabled)
    
    def test_llm_provider_detection_openai(self):
        """Test OpenAI provider detection."""
        os.environ["OPENAI_API_KEY"] = "sk-test"
        
        settings = LLMSettings.from_env(load_dotenv=False)
        self.assertEqual(settings.provider, "openai")
        self.assertEqual(settings.api_key, "sk-test")
    
    def test_llm_provider_detection_azure(self):
        """Test Azure provider detection."""
        os.environ["AZURE_OPENAI_API_KEY"] = "azure-key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
        
        settings = LLMSettings.from_env(load_dotenv=False)
        self.assertEqual(settings.provider, "azure")
        self.assertEqual(settings.api_key, "azure-key")
        self.assertEqual(settings.azure_endpoint, "https://test.openai.azure.com")
    
    def test_llm_provider_detection_gemini(self):
        """Test Gemini provider detection."""
        os.environ["GEMINI_API_KEY"] = "gemini-key"
        
        settings = LLMSettings.from_env(load_dotenv=False)
        self.assertEqual(settings.provider, "gemini")
        self.assertEqual(settings.google_api_key, "gemini-key")
    
    def test_llm_provider_detection_ollama(self):
        """Test Ollama provider detection."""
        os.environ["OLLAMA_HOST"] = "http://localhost:11434"
        
        settings = LLMSettings.from_env(load_dotenv=False)
        self.assertEqual(settings.provider, "ollama")
        self.assertEqual(settings.ollama_host, "http://localhost:11434")
    
    def test_llm_explicit_provider(self):
        """Test explicit provider override."""
        os.environ["OPENAI_API_KEY"] = "sk-test"  # Would auto-detect OpenAI
        
        settings = LLMSettings.from_env(load_dotenv=False, provider="gemini")
        self.assertEqual(settings.provider, "gemini")
    
    def test_llm_validation_openai_missing_key(self):
        """Test validation failure for OpenAI without API key."""
        settings = LLMSettings(provider="openai", model="gpt-4")
        self.assertFalse(settings.is_valid)
        self.assertIn("OpenAI/Azure provider requires api_key", settings.validation_errors[0])
    
    def test_llm_validation_temperature_range(self):
        """Test temperature validation."""
        settings = LLMSettings(provider="openai", api_key="sk-test", temperature=3.0)
        self.assertFalse(settings.is_valid)
        self.assertIn("Temperature must be between 0 and 2", settings.validation_errors[0])


class TestEmbeddingsSettings(unittest.TestCase):
    """Test embeddings-specific settings."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        embeddings_vars = [
            "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION",
            "OPENAI_API_KEY", "GOOGLE_GENAI_API_KEY", "EMBEDDING_DEVICE"
        ]
        for var in embeddings_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_embeddings_settings_defaults(self):
        """Test embeddings settings with defaults."""
        settings = EmbeddingsSettings.from_env(load_dotenv=False)
        
        self.assertEqual(settings.provider, "openai")
        self.assertEqual(settings.model, "text-embedding-3-small")
        self.assertEqual(settings.device, "auto")
        self.assertTrue(settings.use_sentence_transformers)
        self.assertEqual(settings.cache_duration_seconds, 7200)
    
    def test_embeddings_custom_settings(self):
        """Test embeddings with custom environment variables."""
        os.environ["EMBEDDING_PROVIDER"] = "google"
        os.environ["EMBEDDING_MODEL"] = "text-embedding-004"
        os.environ["EMBEDDING_DIMENSION"] = "768"
        os.environ["GOOGLE_GENAI_API_KEY"] = "google-key"
        
        settings = EmbeddingsSettings.from_env(load_dotenv=False)
        
        self.assertEqual(settings.provider, "google")
        self.assertEqual(settings.model, "text-embedding-004")
        self.assertEqual(settings.embedding_dimension, 768)
        self.assertEqual(settings.google_api_key, "google-key")


class TestCacheSettings(unittest.TestCase):
    """Test cache-specific settings."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        cache_vars = [
            "CACHE_PROVIDER", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD",
            "VALKEY_HOST", "VALKEY_PORT", "VALKEY_PASSWORD"
        ]
        for var in cache_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_cache_settings_redis_defaults(self):
        """Test Redis cache settings with defaults."""
        settings = CacheSettings.from_env(load_dotenv=False)
        
        self.assertEqual(settings.provider, "redis")
        self.assertEqual(settings.host, "localhost")
        self.assertEqual(settings.port, 6379)
        self.assertEqual(settings.db, 0)
        self.assertEqual(settings.prefix, "cache:")
        self.assertEqual(settings.ttl, 3600)
        self.assertIsNone(settings.password)
        self.assertEqual(settings.timeout, 4)
    
    def test_cache_settings_valkey(self):
        """Test Valkey cache settings."""
        os.environ["CACHE_PROVIDER"] = "valkey"
        os.environ["VALKEY_HOST"] = "valkey.example.com"
        os.environ["VALKEY_PORT"] = "6380"
        os.environ["VALKEY_PASSWORD"] = "valkey-secret"
        
        settings = CacheSettings.from_env(load_dotenv=False)
        
        self.assertEqual(settings.provider, "valkey")
        self.assertEqual(settings.host, "valkey.example.com")
        self.assertEqual(settings.port, 6380)
        self.assertEqual(settings.password, "valkey-secret")
    
    def test_cache_validation(self):
        """Test cache settings validation."""
        # Invalid port
        settings = CacheSettings(host="localhost", port=0)
        self.assertFalse(settings.is_valid)
        self.assertIn("Port must be between 1 and 65535", settings.validation_errors[0])
        
        # Invalid TTL
        settings = CacheSettings(host="localhost", port=6379, ttl=-1)
        self.assertFalse(settings.is_valid)
        self.assertIn("TTL must be positive", settings.validation_errors[0])


class TestTracingSettings(unittest.TestCase):
    """Test tracing-specific settings."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        tracing_vars = [
            "LANGFUSE_TRACING_ENABLED", "TRACING_ENABLED", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
            "LANGFUSE_HOST", "APP_NAME", "APP_VERSION"
        ]
        for var in tracing_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_tracing_settings_defaults(self):
        """Test tracing settings with defaults."""
        settings = TracingSettings.from_env(load_dotenv=False)
        
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.service_version, "0.1.0")
        self.assertEqual(settings.langfuse_host, "http://localhost:3000")
        self.assertIsNone(settings.service_name)
    
    def test_tracing_validation_enabled_missing_keys(self):
        """Test validation when tracing enabled but keys missing."""
        settings = TracingSettings(enabled=True)
        self.assertFalse(settings.is_valid)
        self.assertTrue(any("langfuse_public_key" in error for error in settings.validation_errors))


class TestDatabaseSettings(unittest.TestCase):
    """Test database-specific settings."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        database_vars = [
            "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", 
            "POSTGRES_PASSWORD", "POSTGRES_SSLMODE", "DATABASE_HOST", "DATABASE_PORT"
        ]
        for var in database_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_database_settings_defaults(self):
        """Test database settings with defaults."""
        settings = DatabaseSettings.from_env(load_dotenv=False)
        
        self.assertEqual(settings.host, "localhost")
        self.assertEqual(settings.port, 5432)
        self.assertEqual(settings.database, "app-database")
        self.assertEqual(settings.username, "app_user")
        self.assertEqual(settings.password, "app_password")
        self.assertEqual(settings.sslmode, "disable")
        self.assertEqual(settings.pool_size, 10)
        self.assertEqual(settings.max_overflow, 20)
        self.assertEqual(settings.pool_timeout, 30)
    
    def test_database_custom_settings(self):
        """Test database with custom environment variables."""
        os.environ["POSTGRES_HOST"] = "db.example.com"
        os.environ["POSTGRES_PORT"] = "5433"
        os.environ["POSTGRES_DB"] = "myapp"
        os.environ["POSTGRES_USER"] = "myuser"
        os.environ["POSTGRES_PASSWORD"] = "mypass"
        os.environ["POSTGRES_SSLMODE"] = "require"
        
        settings = DatabaseSettings.from_env(load_dotenv=False)
        
        self.assertEqual(settings.host, "db.example.com")
        self.assertEqual(settings.port, 5433)
        self.assertEqual(settings.database, "myapp")
        self.assertEqual(settings.username, "myuser")
        self.assertEqual(settings.password, "mypass")
        self.assertEqual(settings.sslmode, "require")
    
    def test_database_validation(self):
        """Test database settings validation."""
        # Invalid port
        settings = DatabaseSettings(host="localhost", port=0)
        self.assertFalse(settings.is_valid)
        self.assertIn("Database port must be between 1 and 65535", settings.validation_errors[0])
        
        # Empty database name
        settings = DatabaseSettings(host="localhost", port=5432, database="")
        self.assertFalse(settings.is_valid)
        self.assertIn("Database name is required", settings.validation_errors[0])
        
        # Invalid SSL mode
        settings = DatabaseSettings(host="localhost", port=5432, sslmode="invalid")
        self.assertFalse(settings.is_valid)
        self.assertIn("Invalid SSL mode", settings.validation_errors[0])
    
    def test_database_connection_strings(self):
        """Test database connection string generation."""
        settings = DatabaseSettings(
            host="db.example.com",
            port=5432,
            database="myapp",
            username="user",
            password="pass",
            sslmode="require"
        )
        
        # Default connection string
        conn_str = settings.get_connection_string()
        expected = "postgresql://user:pass@db.example.com:5432/myapp?sslmode=require"
        self.assertEqual(conn_str, expected)
        
        # Async connection string
        async_str = settings.get_async_connection_string()
        expected_async = "postgresql+asyncpg://user:pass@db.example.com:5432/myapp?sslmode=require"
        self.assertEqual(async_str, expected_async)
        
        # Sync connection string
        sync_str = settings.get_sync_connection_string()
        expected_sync = "postgresql+psycopg2://user:pass@db.example.com:5432/myapp?sslmode=require"
        self.assertEqual(sync_str, expected_sync)


class TestStandardSettingsInheritance(unittest.TestCase):
    """Test StandardSettings inheritance from ApiSettings."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = dict(os.environ)
        # Clear all relevant env vars more comprehensively
        env_vars = [
            "APP_NAME", "APP_VERSION", "ENVIRONMENT", "LOG_LEVEL",
            "ENABLE_LLM", "ENABLE_EMBEDDINGS", "ENABLE_CACHE", "ENABLE_TRACING", "ENABLE_DATABASE",
            "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE",
            "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_TEMPERATURE",
            "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
            "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY",
            "OLLAMA_HOST", "OLLAMA_BASE_URL",
            "REDIS_HOST", "REDIS_PORT", "VALKEY_HOST",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
            "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER",
            "MCP_SERVER_NAME", "MCP_SERVER_PORT", "FASTAPI_HOST"
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_standard_settings_inherits_api_settings(self):
        """Test that StandardSettings inherits from ApiSettings."""
        from core_lib.config.api_settings import ApiSettings
        assert issubclass(StandardSettings, ApiSettings)
    
    def test_standard_settings_has_api_settings_attributes(self):
        """Test StandardSettings has all ApiSettings attributes."""
        settings = StandardSettings.from_env(load_dotenv=False)
        
        # API settings attributes
        assert hasattr(settings, 'cache')
        assert hasattr(settings, 'tracing')
        assert hasattr(settings, 'mcp_server')
        assert hasattr(settings, 'fastapi_server')
        assert hasattr(settings, 'enable_cache')
        assert hasattr(settings, 'enable_tracing')
        assert hasattr(settings, 'enable_mcp_server')
        assert hasattr(settings, 'enable_fastapi_server')
        
        # StandardSettings-specific attributes
        assert hasattr(settings, 'app_name')
        assert hasattr(settings, 'version')
        assert hasattr(settings, 'llm')
        assert hasattr(settings, 'embeddings')
        assert hasattr(settings, 'database')


class TestStandardSettings(unittest.TestCase):
    """Test StandardSettings comprehensive functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = dict(os.environ)
        # Clear all relevant env vars more comprehensively
        env_vars = [
            "APP_NAME", "APP_VERSION", "ENVIRONMENT", "LOG_LEVEL",
            "ENABLE_LLM", "ENABLE_EMBEDDINGS", "ENABLE_CACHE", "ENABLE_TRACING", "ENABLE_DATABASE",
            "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE",
            "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_TEMPERATURE",
            "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
            "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY",
            "OLLAMA_HOST", "OLLAMA_BASE_URL",
            "REDIS_HOST", "REDIS_PORT", "VALKEY_HOST",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
            "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER",
            "MCP_SERVER_NAME", "MCP_SERVER_PORT", "FASTAPI_HOST"
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_standard_settings_defaults(self):
        """Test StandardSettings with defaults."""
        settings = StandardSettings.from_env(load_dotenv=False)
        
        # app_name is auto-detected from pyproject.toml ("core-lib" in this repo)
        self.assertIsNotNone(settings.app_name)
        # Version comes from pyproject.toml in the actual project
        self.assertIsNotNone(settings.version)
        self.assertEqual(settings.environment, "dev")
        self.assertEqual(settings.log_level, "DEBUG")  # DEBUG in dev
        
        # No services should be enabled by default
        self.assertIsNone(settings.llm)
        self.assertIsNone(settings.embeddings)
        self.assertIsNone(settings.cache)
        self.assertIsNone(settings.tracing)
        self.assertIsNone(settings.database)
    
    def test_standard_settings_production_environment(self):
        """Test StandardSettings in production environment."""
        os.environ["ENVIRONMENT"] = "production"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        self.assertEqual(settings.environment, "production")
        self.assertEqual(settings.log_level, "INFO")  # INFO in prod
    
    def test_standard_settings_auto_enable_llm(self):
        """Test auto-enabling LLM when API key present."""
        os.environ["OPENAI_API_KEY"] = "sk-test"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        self.assertTrue(settings.enable_llm)
        self.assertIsNotNone(settings.llm)
        self.assertEqual(settings.llm.provider, "openai")
        self.assertEqual(settings.llm.api_key, "sk-test")
    
    def test_standard_settings_auto_enable_cache(self):
        """Test auto-enabling cache when Redis host present."""
        os.environ["REDIS_HOST"] = "redis.example.com"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        self.assertTrue(settings.enable_cache)
        self.assertIsNotNone(settings.cache)
        self.assertEqual(settings.cache.host, "redis.example.com")
    
    def test_standard_settings_auto_enable_tracing(self):
        """Test auto-enabling tracing when Langfuse keys present."""
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk_test"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk_test"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        self.assertTrue(settings.enable_tracing)
        self.assertIsNotNone(settings.tracing)
        self.assertEqual(settings.tracing.langfuse_public_key, "pk_test")
    
    def test_standard_settings_auto_enable_database(self):
        """Test auto-enabling database when Postgres host present."""
        os.environ["POSTGRES_HOST"] = "db.example.com"
        os.environ["POSTGRES_DB"] = "myapp"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        self.assertTrue(settings.enable_database)
        self.assertIsNotNone(settings.database)
        self.assertEqual(settings.database.host, "db.example.com")
        self.assertEqual(settings.database.database, "myapp")
    
    def test_standard_settings_explicit_enable_disable(self):
        """Test explicit service enablement/disablement."""
        os.environ["OPENAI_API_KEY"] = "sk-test"  # Would auto-enable
        
        settings = StandardSettings.from_env(
            load_dotenv=False,
            enable_llm=False,  # Explicitly disable
            enable_cache=True   # Explicitly enable (uses default config)
        )
        
        self.assertFalse(settings.enable_llm)
        self.assertIsNone(settings.llm)
        
        # Cache should be enabled with default configuration
        self.assertTrue(settings.enable_cache)
        self.assertIsNotNone(settings.cache)
        self.assertEqual(settings.cache.host, "localhost")  # Default host
    
    def test_standard_settings_backward_compatibility(self):
        """Test backward compatibility methods."""
        # Note: app_name is auto-detected from pyproject.toml, so APP_NAME env var is ignored
        # unless project_root is not set or pyproject.toml doesn't have a name
        
        settings = StandardSettings.from_env(load_dotenv=False)
        app_settings = settings.as_app_settings()
        
        # app_name comes from pyproject.toml auto-detection ("core-lib" in this repo)
        self.assertIsNotNone(app_settings.app_name)
        # Version comes from project detection, just verify it exists
        self.assertIsNotNone(app_settings.version)
    
    def test_get_llm_config_integration(self):
        """Test getting LLM config for existing clients."""
        os.environ["OPENAI_API_KEY"] = "sk-test"
        os.environ["OPENAI_MODEL"] = "gpt-4"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        llm_config = settings.get_llm_config()
        
        # Should return an OpenAIConfig-compatible object
        self.assertEqual(llm_config.provider, "openai")
        self.assertEqual(llm_config.api_key, "sk-test")
        self.assertEqual(llm_config.model, "gpt-4")
    
    def test_get_llm_config_not_configured(self):
        """Test getting LLM config when not configured."""
        # Make sure no LLM-related env vars are set
        for var in ["OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "OLLAMA_HOST", "AZURE_OPENAI_API_KEY"]:
            if var in os.environ:
                del os.environ[var]
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        with self.assertRaises(SettingsError) as cm:
            settings.get_llm_config()
        self.assertIn("LLM not configured", str(cm.exception))
    
    def test_get_database_config_integration(self):
        """Test getting database config."""
        os.environ["POSTGRES_HOST"] = "db.test.com"
        os.environ["POSTGRES_DB"] = "testdb"
        os.environ["POSTGRES_USER"] = "testuser"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        db_config = settings.get_database_config()
        
        self.assertEqual(db_config.host, "db.test.com")
        self.assertEqual(db_config.database, "testdb")
        self.assertEqual(db_config.username, "testuser")
        
        # Test connection string generation
        conn_str = db_config.get_connection_string()
        self.assertIn("db.test.com", conn_str)
        self.assertIn("testdb", conn_str)
        self.assertIn("testuser", conn_str)
    
    def test_get_database_config_not_configured(self):
        """Test getting database config when not configured."""
        # Make sure no database-related env vars are set
        for var in ["POSTGRES_HOST", "POSTGRES_DB", "DATABASE_HOST"]:
            if var in os.environ:
                del os.environ[var]
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        with self.assertRaises(SettingsError) as cm:
            settings.get_database_config()
        self.assertIn("Database not configured", str(cm.exception))


class TestSettingsManager(unittest.TestCase):
    """Test SettingsManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = SettingsManager()
    
    def test_register_and_get_settings(self):
        """Test registering and retrieving settings."""
        settings = CustomTestSettings(name="test", count=42)
        
        self.manager.register("test", settings)
        retrieved = self.manager.get("test")
        
        self.assertEqual(retrieved, settings)
        self.assertEqual(retrieved.name, "test")
        self.assertEqual(retrieved.count, 42)
    
    def test_get_required_success(self):
        """Test get_required with existing settings."""
        settings = CustomTestSettings(name="test")
        self.manager.register("test", settings)
        
        retrieved = self.manager.get_required("test")
        self.assertEqual(retrieved, settings)
    
    def test_get_required_missing(self):
        """Test get_required with missing settings."""
        with self.assertRaises(SettingsError) as cm:
            self.manager.get_required("nonexistent")
        self.assertIn("Settings 'nonexistent' not found", str(cm.exception))
    
    def test_list_names(self):
        """Test listing registered settings names."""
        self.manager.register("test1", CustomTestSettings(name="test1"))
        self.manager.register("test2", CustomTestSettings(name="test2"))
        
        names = self.manager.list_names()
        self.assertEqual(set(names), {"test1", "test2"})
    
    def test_validate_all(self):
        """Test validating all registered settings."""
        valid_settings = CustomTestSettings(name="valid", count=10)
        invalid_settings = CustomTestSettings(name="invalid", count=-1)
        
        self.manager.register("valid", valid_settings)
        # Expect warning when registering invalid settings
        with pytest.warns(UserWarning, match="Registering invalid settings 'invalid'"):
            self.manager.register("invalid", invalid_settings)
        
        errors = self.manager.validate_all()
        
        self.assertEqual(len(errors), 1)
        self.assertIn("invalid", errors)
        self.assertNotIn("valid", errors)
    
    def test_as_dict(self):
        """Test converting all settings to dict."""
        settings1 = CustomTestSettings(name="test1", count=1)
        settings2 = CustomTestSettings(name="test2", count=2)
        
        self.manager.register("settings1", settings1)
        self.manager.register("settings2", settings2)
        
        result = self.manager.as_dict()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result["settings1"]["name"], "test1")
        self.assertEqual(result["settings2"]["name"], "test2")


class TestStandardSettingsLLMAutoDetection(unittest.TestCase):
    """Test StandardSettings LLM auto-detection (beyond ApiSettings)."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = dict(os.environ)
        for var in list(os.environ.keys()):
            if any(x in var for x in ["OPENAI", "AZURE", "GEMINI", "GOOGLE_GENAI", "OLLAMA", "LLM"]):
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_llm_auto_detection_openai(self):
        """Test LLM auto-enabled when OpenAI key is set."""
        os.environ["OPENAI_API_KEY"] = "sk-test"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        assert settings.enable_llm is True
        assert settings.llm is not None
        assert settings.llm.provider == "openai"
    
    def test_llm_auto_detection_azure(self):
        """Test LLM auto-enabled when Azure keys are set."""
        os.environ["AZURE_OPENAI_API_KEY"] = "azure-key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        assert settings.enable_llm is True
        assert settings.llm is not None
        assert settings.llm.provider == "azure"
    
    def test_llm_auto_detection_gemini(self):
        """Test LLM auto-enabled when Gemini key is set."""
        os.environ["GEMINI_API_KEY"] = "gemini-key"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        assert settings.enable_llm is True
        assert settings.llm is not None
        assert settings.llm.provider == "gemini"
    
    def test_llm_explicit_disable_overrides_env(self):
        """Test explicit LLM disable overrides auto-detection."""
        os.environ["OPENAI_API_KEY"] = "sk-test"
        
        settings = StandardSettings.from_env(load_dotenv=False, enable_llm=False)
        
        assert settings.enable_llm is False
        assert settings.llm is None


class TestStandardSettingsEmbeddingsAutoDetection(unittest.TestCase):
    """Test StandardSettings embeddings auto-detection."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = dict(os.environ)
        for var in list(os.environ.keys()):
            if "EMBEDDING" in var or "OPENAI" in var:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_embeddings_auto_detection_by_provider(self):
        """Test embeddings auto-enabled when provider is set."""
        os.environ["EMBEDDING_PROVIDER"] = "openai"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        assert settings.enable_embeddings is True
        assert settings.embeddings is not None
    
    def test_embeddings_auto_detection_by_model(self):
        """Test embeddings auto-enabled when model is set."""
        os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        assert settings.enable_embeddings is True
        assert settings.embeddings is not None
    
    def test_embeddings_explicit_enable(self):
        """Test explicit embeddings enablement."""
        settings = StandardSettings.from_env(load_dotenv=False, enable_embeddings=True)
        
        assert settings.enable_embeddings is True
        assert settings.embeddings is not None


class TestStandardSettingsDatabaseAutoDetection(unittest.TestCase):
    """Test StandardSettings database auto-detection."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = dict(os.environ)
        for var in list(os.environ.keys()):
            if "POSTGRES" in var or "DATABASE" in var:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_database_auto_detection_postgres_host(self):
        """Test database auto-enabled when POSTGRES_HOST is set."""
        os.environ["POSTGRES_HOST"] = "db.example.com"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        assert settings.enable_database is True
        assert settings.database is not None
        assert settings.database.host == "db.example.com"
    
    def test_database_auto_detection_database_host(self):
        """Test database auto-enabled when DATABASE_HOST is set."""
        os.environ["DATABASE_HOST"] = "db.example.com"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        assert settings.enable_database is True
        assert settings.database is not None


class TestStandardSettingsNullSafeProperties(unittest.TestCase):
    """Test StandardSettings null-safe properties for LLM, embeddings, database."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = dict(os.environ)
        for var in list(os.environ.keys()):
            del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_llm_safe_when_not_configured(self):
        """Test llm_safe returns NullConfig when LLM not configured."""
        settings = StandardSettings.from_env(load_dotenv=False)
        llm_safe = settings.llm_safe
        
        assert llm_safe.provider is None
        assert llm_safe.model is None
        assert bool(llm_safe) is False
    
    def test_llm_safe_when_configured(self):
        """Test llm_safe returns actual config when configured."""
        os.environ["OPENAI_API_KEY"] = "sk-test"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        llm_safe = settings.llm_safe
        
        assert llm_safe.provider == "openai"
        assert bool(llm_safe) is True
    
    def test_embeddings_safe_when_not_configured(self):
        """Test embeddings_safe returns NullConfig when not configured."""
        settings = StandardSettings.from_env(load_dotenv=False)
        embeddings_safe = settings.embeddings_safe
        
        assert embeddings_safe.provider is None
        assert bool(embeddings_safe) is False
    
    def test_database_safe_when_not_configured(self):
        """Test database_safe returns NullConfig when not configured."""
        settings = StandardSettings.from_env(load_dotenv=False)
        database_safe = settings.database_safe
        
        assert database_safe.host is None
        assert bool(database_safe) is False


class TestStandardSettingsExtendFromEnv(unittest.TestCase):
    """Test StandardSettings.extend_from_env functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = dict(os.environ)
        for var in list(os.environ.keys()):
            del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_extend_from_env_basic(self):
        """Test extending StandardSettings with custom fields."""
        os.environ["CUSTOM_FIELD"] = "custom_value"
        os.environ["CUSTOM_PORT"] = "8080"
        
        custom_config = {
            "custom_field": {
                "env_vars": ["CUSTOM_FIELD"],
                "default": "default_value",
                "env_type": str
            },
            "custom_port": {
                "env_vars": ["CUSTOM_PORT"],
                "default": 3000,
                "env_type": int
            }
        }
        
        extended = StandardSettings.extend_from_env(
            custom_config=custom_config,
            load_dotenv=False
        )
        
        assert hasattr(extended, 'custom_field')
        assert extended.custom_field == "custom_value"
        assert hasattr(extended, 'custom_port')
        assert extended.custom_port == 8080
    
    def test_extend_from_env_with_standard_settings(self):
        """Test extended settings still has StandardSettings attributes."""
        os.environ["OPENAI_API_KEY"] = "sk-test"
        os.environ["CUSTOM_FIELD"] = "value"
        
        custom_config = {
            "custom_field": {
                "env_vars": ["CUSTOM_FIELD"],
                "default": "default",
                "env_type": str
            }
        }
        
        extended = StandardSettings.extend_from_env(
            custom_config=custom_config,
            load_dotenv=False
        )
        
        # Should have StandardSettings attributes
        assert hasattr(extended, 'app_name')
        assert hasattr(extended, 'llm')
        assert extended.enable_llm is True
        
        # Should have custom attributes
        assert hasattr(extended, 'custom_field')
        assert extended.custom_field == "value"
    
    def test_extend_from_env_required_field_missing(self):
        """Test extend_from_env raises when required field is missing."""
        custom_config = {
            "required_field": {
                "env_vars": ["REQUIRED_VAR"],
                "required": True,
                "env_type": str
            }
        }
        
        with pytest.raises(SettingsError, match="Required custom field"):
            StandardSettings.extend_from_env(
                custom_config=custom_config,
                load_dotenv=False
            )
    
    def test_extend_from_env_as_dict(self):
        """Test extended settings as_dict includes custom fields."""
        os.environ["CUSTOM_FIELD"] = "value"
        
        custom_config = {
            "custom_field": {
                "env_vars": ["CUSTOM_FIELD"],
                "default": "default"
            }
        }
        
        extended = StandardSettings.extend_from_env(
            custom_config=custom_config,
            load_dotenv=False
        )
        
        result = extended.as_dict()
        assert "custom_field" in result
        assert result["custom_field"] == "value"


class TestStandardSettingsAllServicesIntegration(unittest.TestCase):
    """Test StandardSettings with all services enabled (full integration)."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = dict(os.environ)
        for var in list(os.environ.keys()):
            del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_all_services_auto_detected(self):
        """Test all services auto-detected from environment."""
        os.environ.update({
            "OPENAI_API_KEY": "sk-test",
            "EMBEDDING_PROVIDER": "openai",
            "REDIS_HOST": "localhost",
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test",
            "POSTGRES_HOST": "db.example.com",
            "MCP_SERVER_NAME": "test-server",
            "FASTAPI_HOST": "0.0.0.0"
        })
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        # StandardSettings-specific services
        assert settings.enable_llm is True
        assert settings.llm is not None
        assert settings.enable_embeddings is True
        assert settings.embeddings is not None
        assert settings.enable_database is True
        assert settings.database is not None
        
        # ApiSettings-inherited services
        assert settings.enable_cache is True
        assert settings.cache is not None
        assert settings.enable_tracing is True
        assert settings.tracing is not None
        assert settings.enable_mcp_server is True
        assert settings.mcp_server is not None
        assert settings.enable_fastapi_server is True
        assert settings.fastapi_server is not None
    
    def test_all_services_as_dict(self):
        """Test as_dict includes all service configurations."""
        os.environ.update({
            "OPENAI_API_KEY": "sk-test",
            "REDIS_HOST": "localhost",
            "POSTGRES_HOST": "db.example.com"
        })
        
        settings = StandardSettings.from_env(load_dotenv=False)
        result = settings.as_dict()
        
        # App settings
        assert "app_name" in result
        assert "version" in result
        
        # StandardSettings services
        assert "llm" in result
        assert result["llm"] is not None
        
        # ApiSettings services
        assert "cache" in result
        assert result["cache"] is not None


class TestIntegrationWithExistingClasses(unittest.TestCase):
    """Test integration with existing configuration classes."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        # Set up a basic environment for testing - clear all LLM-related vars
        env_vars = [
            "OPENAI_API_KEY", "OPENAI_MODEL", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
            "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "OLLAMA_HOST", "OLLAMA_BASE_URL",
            "LLM_PROVIDER", "REDIS_HOST", "EMBEDDING_PROVIDER"
        ]
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_llm_config_compatibility(self):
        """Test that LLM config is compatible with existing classes."""
        os.environ["OPENAI_API_KEY"] = "sk-test"
        os.environ["OPENAI_MODEL"] = "gpt-3.5-turbo"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        llm_config = settings.get_llm_config()
        
        # Should have the same interface as existing LLM configs
        self.assertTrue(hasattr(llm_config, 'provider'))
        self.assertTrue(hasattr(llm_config, 'model'))
        self.assertTrue(hasattr(llm_config, 'temperature'))
        self.assertTrue(hasattr(llm_config, 'api_key'))
        
        self.assertEqual(llm_config.provider, "openai")
        self.assertEqual(llm_config.model, "gpt-3.5-turbo")
        self.assertEqual(llm_config.api_key, "sk-test")
    
    def test_redis_config_compatibility(self):
        """Test that Redis config is compatible with existing classes."""
        os.environ["REDIS_HOST"] = "localhost"
        os.environ["REDIS_PORT"] = "6379"
        os.environ["REDIS_PASSWORD"] = "secret"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        redis_config = settings.get_redis_config()
        
        # Should have the same interface as existing RedisConfig
        self.assertTrue(hasattr(redis_config, 'host'))
        self.assertTrue(hasattr(redis_config, 'port'))
        self.assertTrue(hasattr(redis_config, 'password'))
        self.assertTrue(hasattr(redis_config, 'ttl'))
        
        self.assertEqual(redis_config.host, "localhost")
        self.assertEqual(redis_config.port, 6379)
        self.assertEqual(redis_config.password, "secret")
    
    def test_embeddings_config_compatibility(self):
        """Test that embeddings config is compatible with existing classes."""
        os.environ["EMBEDDING_PROVIDER"] = "openai"
        os.environ["EMBEDDING_MODEL"] = "text-embedding-3-large"
        os.environ["OPENAI_API_KEY"] = "sk-test"
        
        settings = StandardSettings.from_env(load_dotenv=False)
        embeddings_config = settings.get_embeddings_config()
        
        # Should have the same interface as existing EmbeddingsConfig
        self.assertTrue(hasattr(embeddings_config, 'provider'))
        self.assertTrue(hasattr(embeddings_config, 'model'))
        self.assertTrue(hasattr(embeddings_config, 'api_key'))
        
        self.assertEqual(embeddings_config.provider, "openai")
        self.assertEqual(embeddings_config.model, "text-embedding-3-large")
        self.assertEqual(embeddings_config.api_key, "sk-test")


class TestMCPServerSettings(unittest.TestCase):
    """Test MCP server settings functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear environment variables
        for key in list(os.environ.keys()):
            if key.startswith(("MCP_", "APP_")):
                del os.environ[key]
    
    def test_mcp_server_settings_defaults(self):
        """Test MCP server settings with defaults."""
        from core_lib.config.mcp_settings import MCPServerSettings
        settings = MCPServerSettings.from_env(load_dotenv=False)
        
        self.assertEqual(settings.server_name, "app-server")
        self.assertEqual(settings.version, "0.1.0")
        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 8204)
        self.assertEqual(settings.url, "http://0.0.0.0:8204")
        self.assertEqual(settings.timeout, 30)
        self.assertEqual(settings.transport, "streamable-http")
        self.assertTrue(settings.is_valid)
    
    def test_mcp_server_custom_settings(self):
        """Test MCP server settings with custom values."""
        from core_lib.config.mcp_settings import MCPServerSettings
        os.environ.update({
            "MCP_SERVER_NAME": "custom-server",
            "MCP_SERVER_VERSION": "1.2.3",
            "MCP_SERVER_HOST": "localhost",
            "MCP_SERVER_PORT": "9000",
            "MCP_SERVER_TIMEOUT": "60",
            "MCP_TRANSPORT": "websocket"
        })
        
        settings = MCPServerSettings.from_env(load_dotenv=False)
        
        self.assertEqual(settings.server_name, "custom-server")
        self.assertEqual(settings.version, "1.2.3")
        self.assertEqual(settings.host, "localhost")
        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.url, "http://localhost:9000")
        self.assertEqual(settings.timeout, 60)
        self.assertEqual(settings.transport, "websocket")
    
    def test_mcp_server_validation(self):
        """Test MCP server validation."""
        from core_lib.config.mcp_settings import MCPServerSettings
        
        # Test invalid port
        settings = MCPServerSettings(port=70000)
        self.assertFalse(settings.is_valid)
        errors = settings.validate()
        self.assertIn("Port must be between 1 and 65535", errors)
        
        # Test invalid transport
        settings = MCPServerSettings(transport="invalid")
        self.assertFalse(settings.is_valid)
        errors = settings.validate()
        self.assertIn("Transport must be one of: streamable-http, stdio, websocket", errors)
    
    def test_mcp_server_connection_config(self):
        """Test MCP server connection configuration."""
        from core_lib.config.mcp_settings import MCPServerSettings
        settings = MCPServerSettings(
            host="localhost",
            port=8000,
            transport="streamable-http"
        )
        
        config = settings.get_connection_config()
        self.assertEqual(config["url"], "http://localhost:8000")
        self.assertEqual(config["transport"], "streamable-http")
        self.assertEqual(config["host"], "localhost")
        self.assertEqual(config["port"], 8000)
    
    def test_standard_settings_with_mcp_server(self):
        """Test StandardSettings with MCP server configuration."""
        os.environ.update({
            "MCP_SERVER_NAME": "test-server",
            "MCP_SERVER_PORT": "8500",
        })
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        self.assertTrue(settings.enable_mcp_server)
        self.assertIsNotNone(settings.mcp_server)
        self.assertEqual(settings.mcp_server.server_name, "test-server")
        self.assertEqual(settings.mcp_server.port, 8500)
        
        # Test get_mcp_server_config method
        mcp_config = settings.get_mcp_server_config()
        self.assertEqual(mcp_config.server_name, "test-server")
        self.assertEqual(mcp_config.port, 8500)


if __name__ == "__main__":
    unittest.main()
