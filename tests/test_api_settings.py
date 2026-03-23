"""
Test suite for ApiSettings configuration.

Tests cover:
- ApiSettings from_env functionality
- Service auto-detection (cache, tracing, MCP server, FastAPI server)
- Service enablement flags
- Validation
- Configuration retrieval methods
- Null-safe property accessors
- Integration with service-specific settings
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from core_lib.config.api_settings import ApiSettings
from core_lib.config.base_settings import SettingsError
from core_lib.config.cache_settings import CacheSettings
from core_lib.config.tracing_settings import TracingSettings
from core_lib.config.mcp_settings import MCPServerSettings
from core_lib.config.fastapi_settings import FastAPIServerSettings


class TestApiSettingsBasics:
    """Test basic ApiSettings functionality."""
    
    def test_api_settings_defaults(self):
        """Test ApiSettings with defaults (no services enabled)."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.cache is None
            assert settings.tracing is None
            assert settings.mcp_server is None
            assert settings.fastapi_server is None
            assert settings.enable_cache is False
            assert settings.enable_tracing is False
            assert settings.enable_mcp_server is False
            assert settings.enable_fastapi_server is False
    
    def test_api_settings_as_dict_empty(self):
        """Test as_dict with no services configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            result = settings.as_dict()
            
            assert result["cache"] is None
            assert result["tracing"] is None
            assert result["mcp_server"] is None
            assert result["fastapi_server"] is None
            assert result["enable_cache"] is False
            assert result["enable_tracing"] is False
            assert result["enable_mcp_server"] is False
            assert result["enable_fastapi_server"] is False


class TestCacheAutoDetection:
    """Test cache service auto-detection and configuration."""
    
    def test_redis_auto_detection(self):
        """Test cache auto-enabled when REDIS_HOST is set."""
        env = {
            "REDIS_HOST": "redis.example.com",
            "REDIS_PORT": "6379",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_cache is True
            assert settings.cache is not None
            assert settings.cache.host == "redis.example.com"
            assert settings.cache.port == 6379
    
    def test_valkey_auto_detection(self):
        """Test cache auto-enabled when VALKEY_HOST is set."""
        env = {
            "VALKEY_HOST": "valkey.example.com",
            "VALKEY_PORT": "6380",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_cache is True
            assert settings.cache is not None
    
    def test_cache_explicit_enable(self):
        """Test explicit cache enablement."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False, enable_cache=True)
            
            assert settings.enable_cache is True
            assert settings.cache is not None
            # Should have default Redis configuration
            assert settings.cache.host == "localhost"
            assert settings.cache.port == 6379
    
    def test_cache_explicit_disable_overrides_env(self):
        """Test explicit disable overrides environment auto-detection."""
        env = {"REDIS_HOST": "redis.example.com"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False, enable_cache=False)
            
            assert settings.enable_cache is False
            assert settings.cache is None


class TestTracingAutoDetection:
    """Test tracing service auto-detection and configuration."""
    
    def test_langfuse_auto_detection(self):
        """Test tracing auto-enabled when Langfuse keys are set."""
        env = {
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_tracing is True
            assert settings.tracing is not None
            assert settings.tracing.langfuse_public_key == "pk_test"
            assert settings.tracing.langfuse_secret_key == "sk_test"

    def test_langfuse_auto_detection_uses_test_keys_in_dev(self):
        """Test dev environments prefer LANGFUSE_TEST_* credentials."""
        env = {
            "ENVIRONMENT": "dev",
            "LANGFUSE_TEST_PUBLIC_KEY": "pk_test_env",
            "LANGFUSE_TEST_SECRET_KEY": "sk_test_env",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)

            assert settings.enable_tracing is True
            assert settings.tracing is not None
            assert settings.tracing.langfuse_public_key == "pk_test_env"
            assert settings.tracing.langfuse_secret_key == "sk_test_env"

    def test_langfuse_auto_detection_dev_falls_back_to_default_keys(self):
        """Test dev environments fall back to LANGFUSE_* when test keys are absent."""
        env = {
            "ENVIRONMENT": "dev",
            "LANGFUSE_PUBLIC_KEY": "pk_default",
            "LANGFUSE_SECRET_KEY": "sk_default",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)

            assert settings.enable_tracing is True
            assert settings.tracing is not None
            assert settings.tracing.langfuse_public_key == "pk_default"
            assert settings.tracing.langfuse_secret_key == "sk_default"

    def test_langfuse_auto_detection_uses_prod_keys_in_production(self):
        """Test production environments use LANGFUSE_* credentials."""
        env = {
            "ENVIRONMENT": "production",
            "LANGFUSE_TEST_PUBLIC_KEY": "pk_should_not_use",
            "LANGFUSE_TEST_SECRET_KEY": "sk_should_not_use",
            "LANGFUSE_PUBLIC_KEY": "pk_prod",
            "LANGFUSE_SECRET_KEY": "sk_prod",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)

            assert settings.enable_tracing is True
            assert settings.tracing is not None
            assert settings.tracing.langfuse_public_key == "pk_prod"
            assert settings.tracing.langfuse_secret_key == "sk_prod"
    
    def test_tracing_explicit_enable(self):
        """Test explicit tracing enablement."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False, enable_tracing=True)
            
            assert settings.enable_tracing is True
            assert settings.tracing is not None
    
    def test_tracing_partial_keys_no_auto_enable(self):
        """Test tracing not auto-enabled with only one key."""
        env = {"LANGFUSE_PUBLIC_KEY": "pk_test"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            # Should not auto-enable with partial configuration
            assert settings.enable_tracing is False

    def test_tracing_partial_test_keys_no_auto_enable_in_dev(self):
        """Test tracing not auto-enabled with incomplete LANGFUSE_TEST_* config."""
        env = {
            "ENVIRONMENT": "dev",
            "LANGFUSE_TEST_PUBLIC_KEY": "pk_test_only",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)

            assert settings.enable_tracing is False


class TestMCPServerAutoDetection:
    """Test MCP server auto-detection and configuration."""
    
    def test_mcp_server_auto_detection_by_host(self):
        """Test MCP server auto-enabled when MCP_SERVER_HOST is set."""
        env = {"MCP_SERVER_HOST": "localhost"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_mcp_server is True
            assert settings.mcp_server is not None
            assert settings.mcp_server.host == "localhost"
    
    def test_mcp_server_auto_detection_by_port(self):
        """Test MCP server auto-enabled when MCP_SERVER_PORT is set."""
        env = {"MCP_SERVER_PORT": "8500"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_mcp_server is True
            assert settings.mcp_server is not None
            assert settings.mcp_server.port == 8500
    
    def test_mcp_server_auto_detection_by_name(self):
        """Test MCP server auto-enabled when MCP_SERVER_NAME is set."""
        env = {"MCP_SERVER_NAME": "custom-server"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_mcp_server is True
            assert settings.mcp_server is not None
            assert settings.mcp_server.server_name == "custom-server"
    
    def test_mcp_server_explicit_enable(self):
        """Test explicit MCP server enablement."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False, enable_mcp_server=True)
            
            assert settings.enable_mcp_server is True
            assert settings.mcp_server is not None


class TestFastAPIServerAutoDetection:
    """Test FastAPI server auto-detection and configuration."""
    
    def test_fastapi_server_auto_detection_by_host(self):
        """Test FastAPI server auto-enabled when FASTAPI_HOST is set."""
        env = {"FASTAPI_HOST": "0.0.0.0"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_fastapi_server is True
            assert settings.fastapi_server is not None
    
    def test_fastapi_server_auto_detection_by_port(self):
        """Test FastAPI server auto-enabled when FASTAPI_PORT is set."""
        env = {"FASTAPI_PORT": "8080"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_fastapi_server is True
            assert settings.fastapi_server is not None
    
    def test_fastapi_server_auto_detection_by_auth(self):
        """Test FastAPI server auto-enabled when API_AUTH_ENABLED is set."""
        env = {"API_AUTH_ENABLED": "true"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_fastapi_server is True
            assert settings.fastapi_server is not None
    
    def test_fastapi_server_explicit_enable(self):
        """Test explicit FastAPI server enablement."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False, enable_fastapi_server=True)
            
            assert settings.enable_fastapi_server is True
            assert settings.fastapi_server is not None


class TestMultipleServicesEnabled:
    """Test scenarios with multiple services enabled."""
    
    def test_all_services_enabled(self):
        """Test all services enabled simultaneously."""
        env = {
            "REDIS_HOST": "redis.example.com",
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test",
            "MCP_SERVER_NAME": "test-server",
            "FASTAPI_HOST": "localhost",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_cache is True
            assert settings.cache is not None
            assert settings.enable_tracing is True
            assert settings.tracing is not None
            assert settings.enable_mcp_server is True
            assert settings.mcp_server is not None
            assert settings.enable_fastapi_server is True
            assert settings.fastapi_server is not None
    
    def test_selective_services_enabled(self):
        """Test only some services enabled."""
        env = {
            "REDIS_HOST": "redis.example.com",
            "MCP_SERVER_PORT": "8500",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_cache is True
            assert settings.enable_tracing is False
            assert settings.enable_mcp_server is True
            assert settings.enable_fastapi_server is False


class TestConfigRetrievalMethods:
    """Test configuration retrieval methods."""
    
    def test_get_redis_config_success(self):
        """Test getting Redis configuration when cache is enabled."""
        env = {
            "REDIS_HOST": "redis.example.com",
            "REDIS_PORT": "6379",
            "REDIS_PASSWORD": "secret",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            redis_config = settings.get_redis_config()
            
            assert redis_config.host == "redis.example.com"
            assert redis_config.port == 6379
            assert redis_config.password == "secret"
    
    def test_get_redis_config_not_configured(self):
        """Test get_redis_config raises when cache not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            with pytest.raises(SettingsError, match="Cache not configured"):
                settings.get_redis_config()
    
    def test_get_mcp_server_config_success(self):
        """Test getting MCP server configuration when enabled."""
        env = {
            "MCP_SERVER_NAME": "test-server",
            "MCP_SERVER_PORT": "8500",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            mcp_config = settings.get_mcp_server_config()
            
            assert mcp_config.server_name == "test-server"
            assert mcp_config.port == 8500
    
    def test_get_mcp_server_config_not_configured(self):
        """Test get_mcp_server_config raises when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            with pytest.raises(SettingsError, match="MCP server not configured"):
                settings.get_mcp_server_config()
    
    def test_get_fastapi_server_config_success(self):
        """Test getting FastAPI server configuration when enabled."""
        env = {
            "FASTAPI_HOST": "0.0.0.0",
            "FASTAPI_PORT": "8080",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            fastapi_config = settings.get_fastapi_server_config()
            
            assert fastapi_config is not None
    
    def test_get_fastapi_server_config_not_configured(self):
        """Test get_fastapi_server_config raises when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            with pytest.raises(SettingsError, match="FastAPI server not configured"):
                settings.get_fastapi_server_config()


class TestNullSafeProperties:
    """Test null-safe property accessors."""
    
    def test_cache_safe_when_configured(self):
        """Test cache_safe property when cache is configured."""
        env = {"REDIS_HOST": "localhost"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            cache_safe = settings.cache_safe
            
            assert cache_safe is not None
            assert hasattr(cache_safe, 'host')
            assert cache_safe.host == "localhost"
    
    def test_cache_safe_when_not_configured(self):
        """Test cache_safe property returns NullConfig when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            cache_safe = settings.cache_safe
            
            # NullConfig returns None for any attribute
            assert cache_safe.host is None
            assert cache_safe.port is None
            assert bool(cache_safe) is False
    
    def test_tracing_safe_when_configured(self):
        """Test tracing_safe property when tracing is configured."""
        env = {
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            tracing_safe = settings.tracing_safe
            
            assert tracing_safe is not None
            assert hasattr(tracing_safe, 'langfuse_public_key')
    
    def test_tracing_safe_when_not_configured(self):
        """Test tracing_safe property returns NullConfig when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            tracing_safe = settings.tracing_safe
            
            assert tracing_safe.enabled is None
            assert bool(tracing_safe) is False
    
    def test_mcp_server_safe_when_configured(self):
        """Test mcp_server_safe property when configured."""
        env = {"MCP_SERVER_NAME": "test-server"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            mcp_safe = settings.mcp_server_safe
            
            assert mcp_safe is not None
            assert hasattr(mcp_safe, 'server_name')
    
    def test_mcp_server_safe_when_not_configured(self):
        """Test mcp_server_safe property returns NullConfig when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            mcp_safe = settings.mcp_server_safe
            
            assert mcp_safe.server_name is None
            assert bool(mcp_safe) is False
    
    def test_fastapi_server_safe_when_configured(self):
        """Test fastapi_server_safe property when configured."""
        env = {"FASTAPI_HOST": "localhost"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            fastapi_safe = settings.fastapi_server_safe
            
            assert fastapi_safe is not None
    
    def test_fastapi_server_safe_when_not_configured(self):
        """Test fastapi_server_safe property returns NullConfig when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            fastapi_safe = settings.fastapi_server_safe
            
            assert bool(fastapi_safe) is False


class TestApiSettingsValidation:
    """Test ApiSettings validation."""
    
    def test_validation_success_no_services(self):
        """Test validation passes with no services enabled."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.is_valid is True
            settings.validate()  # Should not raise
    
    def test_validation_success_all_services(self):
        """Test validation passes with all services properly configured."""
        env = {
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test",
            "MCP_SERVER_NAME": "test-server",
            "MCP_SERVER_PORT": "8500",
            "FASTAPI_HOST": "localhost",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.is_valid is True
            settings.validate()  # Should not raise


class TestApiSettingsAsDictIntegration:
    """Test as_dict method with various configurations."""
    
    def test_as_dict_with_cache_enabled(self):
        """Test as_dict includes cache configuration."""
        env = {"REDIS_HOST": "redis.example.com"}
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            result = settings.as_dict()
            
            assert result["enable_cache"] is True
            assert result["cache"] is not None
            assert result["cache"]["host"] == "redis.example.com"
    
    def test_as_dict_with_all_services(self):
        """Test as_dict with all services enabled."""
        env = {
            "REDIS_HOST": "localhost",
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test",
            "MCP_SERVER_NAME": "test-server",
            "FASTAPI_HOST": "0.0.0.0",
        }
        
        with patch.dict(os.environ, env, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            result = settings.as_dict()
            
            assert result["enable_cache"] is True
            assert result["cache"] is not None
            assert result["enable_tracing"] is True
            assert result["tracing"] is not None
            assert result["enable_mcp_server"] is True
            assert result["mcp_server"] is not None
            assert result["enable_fastapi_server"] is True
            assert result["fastapi_server"] is not None


class TestApiSettingsOverrides:
    """Test that overrides work correctly."""
    
    def test_override_enable_flags(self):
        """Test overriding enable flags."""
        env = {"REDIS_HOST": "localhost"}
        
        with patch.dict(os.environ, env, clear=True):
            # Override to disable despite env var
            settings = ApiSettings.from_env(
                load_dotenv=False,
                enable_cache=False,
                enable_tracing=True  # Enable without env vars
            )
            
            assert settings.enable_cache is False
            assert settings.cache is None
            assert settings.enable_tracing is True
            assert settings.tracing is not None
    
    def test_override_service_configs(self):
        """Test overriding service configurations directly."""
        custom_cache = CacheSettings(
            host="custom.redis.com",
            port=6380,
            provider="redis"
        )
        
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(
                load_dotenv=False,
                cache=custom_cache,
                enable_cache=True
            )
            
            assert settings.cache.host == "custom.redis.com"
            assert settings.cache.port == 6380
