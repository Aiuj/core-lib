"""
Integration tests for the settings hierarchy.

Tests the full hierarchy of settings classes:
BaseSettings -> AppSettings -> ApiSettings -> StandardSettings

These tests verify that:
1. Inheritance works correctly across all levels
2. Each level adds its expected functionality
3. Settings can be composed and extended
4. The hierarchy maintains consistency
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from core_lib.config.base_settings import BaseSettings, SettingsError
from core_lib.config.app_settings import AppSettings
from core_lib.config.api_settings import ApiSettings
from core_lib.config.standard_settings import StandardSettings


@pytest.fixture(autouse=True)
def isolate_project_root(tmp_path, monkeypatch):
    """Ensure hierarchy unit tests run in an isolated temp directory without ambient pyproject.toml."""
    monkeypatch.chdir(tmp_path)


class TestSettingsHierarchyInheritance:
    """Test that the settings hierarchy inheritance is correct."""
    
    def test_hierarchy_structure(self):
        """Test the inheritance chain is correct."""
        # BaseSettings is the root
        assert issubclass(AppSettings, BaseSettings)
        
        # ApiSettings inherits from BaseSettings (not AppSettings)
        assert issubclass(ApiSettings, BaseSettings)
        
        # StandardSettings inherits from ApiSettings
        assert issubclass(StandardSettings, ApiSettings)
    
    def test_app_settings_is_base_settings(self):
        """Test AppSettings implements BaseSettings interface."""
        with patch.dict(os.environ, {}, clear=True):
            settings = AppSettings.from_env(load_dotenv=False)
            
            # Should have BaseSettings interface
            assert hasattr(settings, 'from_env')
            assert hasattr(settings, 'validate')
            assert hasattr(settings, 'is_valid')
            assert hasattr(settings, 'as_dict')
            assert hasattr(settings, 'merge')
    
    def test_api_settings_has_base_interface(self):
        """Test ApiSettings implements BaseSettings interface."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            # Should have BaseSettings interface
            assert hasattr(settings, 'from_env')
            assert hasattr(settings, 'validate')
            assert hasattr(settings, 'is_valid')
            assert hasattr(settings, 'as_dict')
    
    def test_standard_settings_has_all_interfaces(self):
        """Test StandardSettings has interfaces from all levels."""
        with patch.dict(os.environ, {}, clear=True):
            settings = StandardSettings.from_env(load_dotenv=False)
            
            # BaseSettings interface
            assert hasattr(settings, 'from_env')
            assert hasattr(settings, 'validate')
            assert hasattr(settings, 'is_valid')
            
            # AppSettings attributes
            assert hasattr(settings, 'app_name')
            assert hasattr(settings, 'version')
            assert hasattr(settings, 'environment')
            
            # ApiSettings attributes
            assert hasattr(settings, 'cache')
            assert hasattr(settings, 'tracing')
            assert hasattr(settings, 'mcp_server')
            assert hasattr(settings, 'fastapi_server')
            
            # StandardSettings-specific attributes
            assert hasattr(settings, 'llm')
            assert hasattr(settings, 'embeddings')
            assert hasattr(settings, 'database')


class TestLevelByLevelFunctionality:
    """Test that each level adds its expected functionality."""
    
    def test_base_settings_provides_core_utilities(self):
        """Test BaseSettings provides core parsing and validation."""
        # AppSettings uses BaseSettings utilities
        with patch.dict(os.environ, {"APP_NAME": "test-app"}, clear=True):
            settings = AppSettings.from_env(load_dotenv=False)
            
            assert settings.app_name == "test-app"
            assert settings.is_valid is True
            assert len(settings.validation_errors) == 0
    
    def test_app_settings_adds_app_configuration(self):
        """Test AppSettings adds application-level configuration."""
        with patch.dict(os.environ, {
            "APP_NAME": "my-app",
            "ENVIRONMENT": "production",
            "LOG_LEVEL": "INFO"
        }, clear=True):
            settings = AppSettings.from_env(load_dotenv=False)
            
            assert settings.app_name == "my-app"
            assert settings.environment == "production"
            assert settings.log_level == "INFO"
            assert settings.is_production is True
    
    def test_api_settings_adds_api_services(self):
        """Test ApiSettings adds API-related services."""
        with patch.dict(os.environ, {
            "REDIS_HOST": "localhost",
            "MCP_SERVER_NAME": "test-server"
        }, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            # API-specific services
            assert settings.enable_cache is True
            assert settings.cache is not None
            assert settings.enable_mcp_server is True
            assert settings.mcp_server is not None
            
            # Can retrieve configs
            redis_config = settings.get_redis_config()
            assert redis_config.host == "localhost"
    
    def test_standard_settings_adds_all_services(self):
        """Test StandardSettings adds LLM, embeddings, and database."""
        with patch.dict(os.environ, {
            "APP_NAME": "full-app",
            "OPENAI_API_KEY": "sk-test",
            "EMBEDDING_PROVIDER": "openai",
            "REDIS_HOST": "localhost",
            "POSTGRES_HOST": "db.example.com"
        }, clear=True):
            settings = StandardSettings.from_env(load_dotenv=False)
            
            # App-level
            assert settings.app_name == "full-app"
            
            # API-level services
            assert settings.enable_cache is True
            assert settings.cache is not None
            
            # StandardSettings services
            assert settings.enable_llm is True
            assert settings.llm is not None
            assert settings.enable_embeddings is True
            assert settings.embeddings is not None
            assert settings.enable_database is True
            assert settings.database is not None


class TestCompositionAndExtension:
    """Test that settings can be composed and extended."""
    
    def test_app_settings_can_be_standalone(self):
        """Test AppSettings can be used standalone."""
        with patch.dict(os.environ, {
            "APP_NAME": "simple-app",
            "ENVIRONMENT": "dev"
        }, clear=True):
            settings = AppSettings.from_env(load_dotenv=False)
            
            assert settings.app_name == "simple-app"
            assert settings.environment == "dev"
            assert settings.is_valid is True
    
    def test_api_settings_can_be_standalone(self):
        """Test ApiSettings can be used without StandardSettings."""
        with patch.dict(os.environ, {
            "REDIS_HOST": "localhost",
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test"
        }, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_cache is True
            assert settings.enable_tracing is True
            # No LLM, embeddings, or database (those are StandardSettings)
            assert not hasattr(settings, 'llm')
    
    def test_standard_settings_combines_all_levels(self):
        """Test StandardSettings properly combines all levels."""
        with patch.dict(os.environ, {
            "APP_NAME": "complete-app",
            "OPENAI_API_KEY": "sk-test",
            "REDIS_HOST": "localhost",
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test"
        }, clear=True):
            settings = StandardSettings.from_env(load_dotenv=False)
            
            # All levels present
            assert settings.app_name == "complete-app"  # App level
            assert settings.enable_cache is True  # API level
            assert settings.enable_tracing is True  # API level
            assert settings.enable_llm is True  # Standard level
    
    def test_standard_settings_extend_from_env(self):
        """Test StandardSettings can be extended with custom fields."""
        with patch.dict(os.environ, {
            "APP_NAME": "extended-app",
            "CUSTOM_FIELD": "custom_value"
        }, clear=True):
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
            
            # Has all StandardSettings functionality
            assert extended.app_name == "extended-app"
            
            # Plus custom field
            assert hasattr(extended, 'custom_field')
            assert extended.custom_field == "custom_value"


class TestValidationAcrossLevels:
    """Test validation works correctly across all levels."""
    
    def test_app_settings_validation(self):
        """Test AppSettings validation."""
        with pytest.raises(ValueError, match="Log level must be one of"):
            AppSettings(
                app_name="test",
                log_level="INVALID"
            )
    
    def test_api_settings_validation_invalid_cache(self):
        """Test ApiSettings validates cache configuration."""
        from core_lib.config.cache_settings import CacheSettings
        
        invalid_cache = CacheSettings(host="localhost", port=0)
        
        # Creating ApiSettings with invalid cache should work,
        # but validation should fail
        with pytest.raises(Exception):  # CacheSettings validation will raise
            ApiSettings(
                cache=invalid_cache,
                enable_cache=True
            )
    
    def test_standard_settings_validation_all_services(self):
        """Test StandardSettings validates all service configurations."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "REDIS_HOST": "localhost"
        }, clear=True):
            settings = StandardSettings.from_env(load_dotenv=False)
            
            # Should validate successfully
            assert settings.is_valid is True
            settings.validate()  # Should not raise


class TestConfigurationRetrieval:
    """Test configuration retrieval across the hierarchy."""
    
    def test_api_settings_config_retrieval(self):
        """Test ApiSettings config retrieval methods."""
        with patch.dict(os.environ, {
            "REDIS_HOST": "redis.example.com",
            "MCP_SERVER_NAME": "test-server"
        }, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            # Should be able to get configs
            redis_config = settings.get_redis_config()
            assert redis_config.host == "redis.example.com"
            
            mcp_config = settings.get_mcp_server_config()
            assert mcp_config.server_name == "test-server"
    
    def test_standard_settings_config_retrieval(self):
        """Test StandardSettings retrieves configs from all levels."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "REDIS_HOST": "localhost",
            "POSTGRES_HOST": "db.example.com"
        }, clear=True):
            settings = StandardSettings.from_env(load_dotenv=False)
            
            # ApiSettings configs
            redis_config = settings.get_redis_config()
            assert redis_config.host == "localhost"
            
            # StandardSettings configs
            llm_config = settings.get_llm_config()
            assert llm_config.provider == "openai"
            
            db_config = settings.get_database_config()
            assert db_config.host == "db.example.com"
    
    def test_config_retrieval_not_configured_raises(self):
        """Test config retrieval raises when service not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = StandardSettings.from_env(load_dotenv=False)
            
            with pytest.raises(SettingsError, match="Cache not configured"):
                settings.get_redis_config()
            
            with pytest.raises(SettingsError, match="LLM not configured"):
                settings.get_llm_config()
            
            with pytest.raises(SettingsError, match="Database not configured"):
                settings.get_database_config()


class TestAsDictAcrossLevels:
    """Test as_dict serialization across all levels."""
    
    def test_app_settings_as_dict(self):
        """Test AppSettings as_dict."""
        with patch.dict(os.environ, {
            "APP_NAME": "test-app",
            "ENVIRONMENT": "production"
        }, clear=True):
            settings = AppSettings.from_env(load_dotenv=False)
            result = settings.as_dict()
            
            assert "app_name" in result
            assert result["app_name"] == "test-app"
            assert "environment" in result
            assert result["environment"] == "production"
    
    def test_api_settings_as_dict(self):
        """Test ApiSettings as_dict."""
        with patch.dict(os.environ, {
            "REDIS_HOST": "localhost"
        }, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            result = settings.as_dict()
            
            assert "cache" in result
            assert result["cache"] is not None
            assert "enable_cache" in result
            assert result["enable_cache"] is True
    
    def test_standard_settings_as_dict_complete(self):
        """Test StandardSettings as_dict includes all levels."""
        with patch.dict(os.environ, {
            "APP_NAME": "complete-app",
            "OPENAI_API_KEY": "sk-test",
            "REDIS_HOST": "localhost"
        }, clear=True):
            settings = StandardSettings.from_env(load_dotenv=False)
            result = settings.as_dict()
            
            # App level
            assert "app_name" in result
            assert result["app_name"] == "complete-app"
            
            # API level
            assert "cache" in result
            assert result["cache"] is not None
            
            # Standard level
            assert "llm" in result
            assert result["llm"] is not None


class TestNullSafeAccessorsHierarchy:
    """Test null-safe accessors work across the hierarchy."""
    
    def test_api_settings_null_safe_accessors(self):
        """Test ApiSettings null-safe accessors."""
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            # Should return NullConfig, not raise
            assert settings.cache_safe.host is None
            assert bool(settings.cache_safe) is False
            
            assert settings.tracing_safe.enabled is None
            assert bool(settings.tracing_safe) is False
    
    def test_standard_settings_null_safe_accessors(self):
        """Test StandardSettings null-safe accessors for all services."""
        with patch.dict(os.environ, {}, clear=True):
            settings = StandardSettings.from_env(load_dotenv=False)
            
            # API-level null-safe accessors
            assert settings.cache_safe.host is None
            assert bool(settings.cache_safe) is False
            
            # Standard-level null-safe accessors
            assert settings.llm_safe.provider is None
            assert bool(settings.llm_safe) is False
            
            assert settings.embeddings_safe.provider is None
            assert bool(settings.embeddings_safe) is False
            
            assert settings.database_safe.host is None
            assert bool(settings.database_safe) is False


class TestRealWorldScenarios:
    """Test real-world usage scenarios of the settings hierarchy."""
    
    def test_minimal_api_server_config(self):
        """Test minimal config for an API server (just cache)."""
        with patch.dict(os.environ, {
            "REDIS_HOST": "localhost"
        }, clear=True):
            settings = ApiSettings.from_env(load_dotenv=False)
            
            assert settings.enable_cache is True
            assert settings.enable_tracing is False
            assert settings.enable_mcp_server is False
            assert settings.enable_fastapi_server is False
            
            # Can use null-safe accessors for optional services
            assert bool(settings.tracing_safe) is False
    
    def test_full_llm_application_config(self):
        """Test full config for an LLM application."""
        with patch.dict(os.environ, {
            "APP_NAME": "llm-app",
            "ENVIRONMENT": "production",
            "OPENAI_API_KEY": "sk-test",
            "EMBEDDING_PROVIDER": "openai",
            "REDIS_HOST": "localhost",
            "LANGFUSE_PUBLIC_KEY": "pk_test",
            "LANGFUSE_SECRET_KEY": "sk_test",
            "POSTGRES_HOST": "db.example.com"
        }, clear=True):
            settings = StandardSettings.from_env(load_dotenv=False)
            
            # All services enabled
            assert settings.enable_llm is True
            assert settings.enable_embeddings is True
            assert settings.enable_cache is True
            assert settings.enable_tracing is True
            assert settings.enable_database is True
            
            # Can retrieve all configs
            llm_config = settings.get_llm_config()
            assert llm_config.provider == "openai"
            
            embeddings_config = settings.get_embeddings_config()
            assert embeddings_config.provider == "openai"
            
            redis_config = settings.get_redis_config()
            assert redis_config.host == "localhost"
            
            db_config = settings.get_database_config()
            assert db_config.host == "db.example.com"
    
    def test_selective_service_enablement(self):
        """Test selectively enabling/disabling services."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "REDIS_HOST": "localhost",
            "POSTGRES_HOST": "db.example.com"
        }, clear=True):
            # Enable only specific services
            settings = StandardSettings.from_env(
                load_dotenv=False,
                enable_llm=True,
                enable_cache=True,
                enable_database=False,  # Explicit disable
                enable_embeddings=False
            )
            
            assert settings.enable_llm is True
            assert settings.enable_cache is True
            assert settings.enable_database is False
            assert settings.enable_embeddings is False
