"""Tests for LoggerSettings and OVH LDP integration."""

import os
import pytest
from unittest.mock import MagicMock, patch

from core_lib.config import LoggerSettings
from core_lib.config.base_settings import SettingsError


class TestLoggerSettings:
    """Tests for LoggerSettings configuration."""
    
    def test_logger_settings_defaults(self):
        """Test default LoggerSettings values."""
        settings = LoggerSettings()
        
        assert settings.log_level == "INFO"
        assert settings.file_logging is False
        assert settings.ovh_ldp_enabled is False
        assert settings.ovh_ldp_port == 12202
        assert settings.ovh_ldp_protocol == "gelf_tcp"
        assert settings.ovh_ldp_use_tls is True
        assert settings.ovh_ldp_compress is True
    
    def test_logger_settings_from_env(self, monkeypatch):
        """Test loading LoggerSettings from environment variables."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_CONSOLE_COLORS", "true")
        monkeypatch.setenv("LOG_FILE_ENABLED", "true")
        monkeypatch.setenv("LOG_FILE_PATH", "logs/test.log")
        monkeypatch.setenv("OVH_LDP_ENABLED", "true")
        monkeypatch.setenv("OVH_LDP_TOKEN", "test-token")
        monkeypatch.setenv("OVH_LDP_ENDPOINT", "gra1.logs.ovh.com")
        monkeypatch.setenv("OVH_LDP_PORT", "12345")
        monkeypatch.setenv("OVH_LDP_PROTOCOL", "gelf_udp")
        
        settings = LoggerSettings.from_env(load_dotenv=False)
        
        assert settings.log_level == "DEBUG"
        assert settings.console_colors is True
        assert settings.file_logging is True
        assert settings.file_path == "logs/test.log"
        assert settings.ovh_ldp_enabled is True
        assert settings.ovh_ldp_token == "test-token"
        assert settings.ovh_ldp_endpoint == "gra1.logs.ovh.com"
        assert settings.ovh_ldp_port == 12345
        assert settings.ovh_ldp_protocol == "gelf_udp"
    
    def test_logger_settings_additional_fields_parsing(self, monkeypatch):
        """Test parsing of additional fields from JSON."""
        monkeypatch.setenv("OVH_LDP_ADDITIONAL_FIELDS", '{"env": "prod", "version": "1.0"}')
        
        settings = LoggerSettings.from_env(load_dotenv=False)
        
        assert settings.ovh_ldp_additional_fields == {"env": "prod", "version": "1.0"}
    
    def test_logger_settings_invalid_json_additional_fields(self, monkeypatch):
        """Test handling of invalid JSON in additional fields."""
        monkeypatch.setenv("OVH_LDP_ADDITIONAL_FIELDS", "invalid-json")
        
        settings = LoggerSettings.from_env(load_dotenv=False)
        
        # Should default to empty dict on invalid JSON
        assert settings.ovh_ldp_additional_fields == {}
    
    def test_logger_settings_validation_invalid_log_level(self):
        """Test validation rejects invalid log level."""
        with pytest.raises(SettingsError, match="Invalid log level"):
            settings = LoggerSettings(log_level="INVALID")
    
    def test_logger_settings_validation_missing_token(self):
        """Test validation requires token when OVH LDP is enabled."""
        with pytest.raises(SettingsError, match="ovh_ldp_token not provided"):
            settings = LoggerSettings(
                ovh_ldp_enabled=True,
                ovh_ldp_endpoint="gra1.logs.ovh.com"
            )
    
    def test_logger_settings_validation_missing_endpoint(self):
        """Test validation requires endpoint when OVH LDP is enabled."""
        with pytest.raises(SettingsError, match="ovh_ldp_endpoint not provided"):
            settings = LoggerSettings(
                ovh_ldp_enabled=True,
                ovh_ldp_token="token"
            )
    
    def test_logger_settings_validation_invalid_protocol(self):
        """Test validation rejects invalid protocol."""
        with pytest.raises(SettingsError, match="Invalid OVH LDP protocol"):
            settings = LoggerSettings(
                ovh_ldp_enabled=True,
                ovh_ldp_token="token",
                ovh_ldp_endpoint="gra1.logs.ovh.com",
                ovh_ldp_protocol="invalid"
            )
    
    def test_logger_settings_validation_invalid_port(self):
        """Test validation rejects invalid port."""
        with pytest.raises(SettingsError, match="Invalid port"):
            settings = LoggerSettings(
                ovh_ldp_enabled=True,
                ovh_ldp_token="token",
                ovh_ldp_endpoint="gra1.logs.ovh.com",
                ovh_ldp_port=99999
            )
    
    def test_logger_settings_validation_invalid_timeout(self):
        """Test validation rejects invalid timeout."""
        with pytest.raises(SettingsError, match="timeout must be at least 1"):
            settings = LoggerSettings(
                ovh_ldp_enabled=True,
                ovh_ldp_token="token",
                ovh_ldp_endpoint="gra1.logs.ovh.com",
                ovh_ldp_timeout=0
            )
    
    def test_logger_settings_as_dict_masks_token(self):
        """Test that as_dict masks the token."""
        settings = LoggerSettings(
            ovh_ldp_enabled=True,
            ovh_ldp_token="secret-token",
            ovh_ldp_endpoint="gra1.logs.ovh.com"
        )
        
        settings_dict = settings.as_dict()
        
        assert settings_dict["ovh_ldp_token"] == "***"
        assert settings_dict["ovh_ldp_endpoint"] == "gra1.logs.ovh.com"
    
    def test_logger_settings_otlp_defaults(self):
        """Test default OTLP settings values."""
        settings = LoggerSettings()
        
        assert settings.otlp_enabled is False
        assert settings.otlp_endpoint == "http://localhost:4318/v1/logs"
        assert settings.otlp_timeout == 10
        assert settings.otlp_insecure is False
        assert settings.otlp_service_name is None
        assert settings.otlp_instance_id is None
        assert settings.otlp_log_channel is None
    
    def test_logger_settings_otlp_from_env(self, monkeypatch):
        """Test loading OTLP settings from environment variables."""
        monkeypatch.setenv("OTLP_ENABLED", "true")
        monkeypatch.setenv("OTLP_ENDPOINT", "http://otel-collector:4318/v1/logs")
        monkeypatch.setenv("OTLP_HEADERS", '{"Authorization": "Bearer token123"}')
        monkeypatch.setenv("OTLP_TIMEOUT", "15")
        monkeypatch.setenv("OTLP_SERVICE_NAME", "my-service")
        monkeypatch.setenv("OTLP_SERVICE_VERSION", "1.0.0")
        monkeypatch.setenv("OTLP_INSTANCE_ID", "server-01")
        monkeypatch.setenv("OTLP_LOG_CHANNEL", "myfaq")
        
        settings = LoggerSettings.from_env(load_dotenv=False)
        
        assert settings.otlp_enabled is True
        assert settings.otlp_endpoint == "http://otel-collector:4318/v1/logs"
        assert settings.otlp_headers == {"Authorization": "Bearer token123"}
        assert settings.otlp_timeout == 15
        assert settings.otlp_service_name == "my-service"
        assert settings.otlp_service_version == "1.0.0"
        assert settings.otlp_instance_id == "server-01"
        assert settings.otlp_log_channel == "myfaq"

    def test_logger_settings_otlp_service_name_stays_unset_without_explicit_env(self, monkeypatch):
        """Test OTLP service name remains unset when OTLP_SERVICE_NAME is not provided."""
        monkeypatch.setenv("APP_NAME", "settings-app-name")
        monkeypatch.setenv("OTLP_ENABLED", "true")
        monkeypatch.setenv("OTLP_ENDPOINT", "http://otel-collector:4318/v1/logs")
        monkeypatch.delenv("OTLP_SERVICE_NAME", raising=False)

        settings = LoggerSettings.from_env(load_dotenv=False)

        assert settings.otlp_service_name is None

    def test_logger_settings_validation_empty_otlp_log_channel(self):
        """Test validation rejects an empty OTLP log channel."""
        with pytest.raises(SettingsError, match="OTLP log channel cannot be empty"):
            LoggerSettings(otlp_log_channel="   ")

    def test_logger_settings_validation_empty_otlp_instance_id(self):
        """Test validation rejects an empty OTLP instance id."""
        with pytest.raises(SettingsError, match="OTLP instance id cannot be empty"):
            LoggerSettings(otlp_instance_id="   ")


class TestLoggerIntegration:
    """Tests for logger integration with OVH LDP."""
    
    @patch('core_lib.tracing.handlers.gelf_handler.GELFTCPHandler')
    def test_setup_logging_with_ovh_ldp(self, mock_gelf_handler):
        """Test setup_logging with OVH LDP enabled."""
        from core_lib.tracing.logger import setup_logging
        
        logger_settings = LoggerSettings(
            log_level="INFO",
            ovh_ldp_enabled=True,
            ovh_ldp_token="test-token",
            ovh_ldp_endpoint="gra1.logs.ovh.com",
            ovh_ldp_port=12202,
            ovh_ldp_protocol="gelf_tcp",
            ovh_ldp_use_tls=True,
            ovh_ldp_compress=True,
            ovh_ldp_additional_fields={"env": "test"},
            ovh_ldp_timeout=10,
        )
        
        logger = setup_logging(
            app_name="test_app",
            logger_settings=logger_settings,
            force=True
        )
        
        # Verify GELF handler was created with correct params
        mock_gelf_handler.assert_called_once_with(
            host="gra1.logs.ovh.com",
            port=12202,
            token="test-token",
            use_tls=True,
            compress=True,
            additional_fields={"env": "test"},
            timeout=10,
        )
        
        assert logger is not None
    
    def test_setup_logging_without_ovh_ldp(self):
        """Test setup_logging works without OVH LDP."""
        from core_lib.tracing.logger import setup_logging
        
        logger_settings = LoggerSettings(
            log_level="INFO",
            ovh_ldp_enabled=False
        )
        
        logger = setup_logging(
            app_name="test_app",
            logger_settings=logger_settings,
            force=True
        )
        
        assert logger is not None
    
    def test_setup_logging_uses_logger_settings_level(self):
        """Test that logger settings level takes precedence."""
        from core_lib.tracing.logger import setup_logging
        
        logger_settings = LoggerSettings(log_level="DEBUG")
        
        logger = setup_logging(
            app_name="test_app",
            logger_settings=logger_settings,
            force=True
        )
        
        # Logger should be at DEBUG level
        assert logger.level == 10  # DEBUG = 10
    
    @patch('core_lib.tracing.handlers.otlp_handler.OTLPHandler')
    def test_setup_logging_with_otlp(self, mock_otlp_handler):
        """Test setup_logging with OTLP enabled."""
        from core_lib.tracing.logger import setup_logging
        
        # Create a mock handler instance
        mock_handler_instance = MagicMock()
        mock_otlp_handler.return_value = mock_handler_instance
        
        logger_settings = LoggerSettings(
            log_level="INFO",
            otlp_enabled=True,
            otlp_endpoint="http://localhost:4318/v1/logs",
            otlp_service_name="test-service",
            otlp_instance_id="server-42",
            otlp_log_channel="faciliter",
        )
        
        logger = setup_logging(
            app_name="test_app",
            logger_settings=logger_settings,
            force=True
        )
        
        # Verify OTLPHandler was created with correct params
        mock_otlp_handler.assert_called_once()
        call_kwargs = mock_otlp_handler.call_args[1]
        assert call_kwargs["endpoint"] == "http://localhost:4318/v1/logs"
        assert call_kwargs["service_name"] == "test-service"
        assert call_kwargs["service_instance_id"] == "server-42"
        assert call_kwargs["log_channel"] == "faciliter"
        
        # Verify handler was started
        mock_handler_instance.start.assert_called_once()
        mock_handler_instance.setLevel.assert_called_once()
    
    def test_setup_logging_without_otlp(self):
        """Test that OTLP handler is not loaded when disabled."""
        from core_lib.tracing.logger import setup_logging
        
        logger_settings = LoggerSettings(
            log_level="INFO",
            otlp_enabled=False,
        )
        
        logger = setup_logging(
            app_name="test_app",
            logger_settings=logger_settings,
            force=True
        )
        
        # Should not raise any errors
        assert logger is not None

    @patch('core_lib.tracing.handlers.otlp_handler.OTLPHandler')
    def test_setup_logging_defaults_otlp_service_name_from_app_name(self, mock_otlp_handler):
        """Test OTLP service name falls back to setup_logging app_name when unset."""
        from core_lib.tracing.logger import setup_logging

        mock_handler_instance = MagicMock()
        mock_otlp_handler.return_value = mock_handler_instance

        logger_settings = LoggerSettings(
            log_level="INFO",
            otlp_enabled=True,
            otlp_endpoint="http://localhost:4318/v1/logs",
        )

        setup_logging(
            app_name="test_app",
            logger_settings=logger_settings,
            force=True,
        )

        call_kwargs = mock_otlp_handler.call_args[1]
        assert call_kwargs["service_name"] == "test_app"


class TestOTLPHandler:
    """Tests for OTLP handler resource attributes."""

    @patch('core_lib.tracing.handlers.otlp_handler.requests.post')
    def test_otlp_handler_sends_log_channel_resource_attribute(self, mock_post):
        """Test OTLP handler includes faciliter.log_channel in resource attributes."""
        from core_lib.tracing.handlers.otlp_handler import _OTLPWorkerHandler

        mock_post.return_value = MagicMock(status_code=200, text="ok")

        handler = _OTLPWorkerHandler(
            endpoint="http://localhost:4318/v1/logs",
            headers={"Content-Type": "application/json"},
            timeout=10,
            insecure=False,
            service_name="test-service",
            service_version="1.0.0",
            service_instance_id="server-01",
            host_name="host-01",
            log_channel="myfaq",
        )

        record = MagicMock(
            levelno=20,
            created=1_700_000_000.0,
            name="test.logger",
            pathname="test.py",
            lineno=42,
            funcName="test_func",
        )
        record.getMessage.return_value = "hello"

        handler.emit(record)
        handler.flush()

        assert mock_post.called
        payload = mock_post.call_args.kwargs["json"]
        resource_attrs = payload["resourceLogs"][0]["resource"]["attributes"]
        assert {"key": "faciliter.log_channel", "value": {"stringValue": "myfaq"}} in resource_attrs
        assert {"key": "service.instance.id", "value": {"stringValue": "server-01"}} in resource_attrs
        assert {"key": "host.name", "value": {"stringValue": "host-01"}} in resource_attrs


class TestStandardSettingsWithLogger:
    """Tests for StandardSettings integration with LoggerSettings."""
    
    def test_standard_settings_auto_detects_logger(self, monkeypatch):
        """Test that StandardSettings auto-detects logger config."""
        monkeypatch.setenv("OVH_LDP_ENABLED", "true")
        monkeypatch.setenv("OVH_LDP_TOKEN", "token")
        monkeypatch.setenv("OVH_LDP_ENDPOINT", "gra1.logs.ovh.com")
        
        from core_lib.config import StandardSettings
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        assert settings.enable_logger is True
        assert settings.logger is not None
        assert settings.logger.ovh_ldp_enabled is True
        assert settings.logger.ovh_ldp_token == "token"
    
    def test_standard_settings_logger_disabled_by_default(self):
        """Test that logger is not enabled by default."""
        from core_lib.config import StandardSettings
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        # Should be disabled if no OVH LDP vars set
        assert settings.enable_logger is False or settings.logger is None
    
    def test_standard_settings_logger_safe_property(self, monkeypatch):
        """Test logger_safe property returns NullConfig when logger is None."""
        from core_lib.config import StandardSettings
        
        settings = StandardSettings.from_env(load_dotenv=False)
        
        # Should not raise even if logger is None
        logger_safe = settings.logger_safe
        assert logger_safe is not None
        
        # NullConfig returns None for any attribute
        assert logger_safe.ovh_ldp_enabled is None

    @patch('core_lib.tracing.handlers.otlp_handler.OTLPHandler')
    def test_standard_settings_setup_logging_uses_settings_app_name_when_otlp_service_name_unset(self, mock_otlp_handler, monkeypatch):
        """Test StandardSettings path uses resolved settings.app_name when OTLP_SERVICE_NAME is unset."""
        from core_lib.config import StandardSettings
        from core_lib.tracing.logger import setup_logging

        mock_handler_instance = MagicMock()
        mock_otlp_handler.return_value = mock_handler_instance

        monkeypatch.delenv("APP_NAME", raising=False)
        monkeypatch.delenv("OTLP_SERVICE_NAME", raising=False)
        monkeypatch.setenv("OTLP_ENABLED", "true")
        monkeypatch.setenv("OTLP_ENDPOINT", "http://otel-collector:4318/v1/logs")

        settings = StandardSettings.from_env(load_dotenv=False, app_name="deployed-app")

        setup_logging(
            app_settings=settings,
            logger_settings=settings.logger_safe,
            force=True,
        )

        call_kwargs = mock_otlp_handler.call_args[1]
        assert call_kwargs["service_name"] == "deployed-app"
