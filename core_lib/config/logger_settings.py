"""Logger Configuration Settings.

This module contains configuration classes for logging, including
support for OVH Logs Data Platform (LDP) via GELF protocol.

OVH Logs Data Platform supports multiple protocols:
- GELF (Graylog Extended Log Format) over TCP/UDP
- Syslog over TCP/UDP  
- HTTP/HTTPS endpoints

Environment Variables:
    LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_FILE_ENABLED: Enable file logging (true/false)
    LOG_FILE_PATH: Path to log file (default: logs/<app_name>.log)
    LOG_FILE_MAX_BYTES: Max file size before rotation (default: 1048576)
    LOG_FILE_BACKUP_COUNT: Number of backup files (default: 3)
    
    OVH_LDP_ENABLED: Enable OVH Logs Data Platform integration (true/false)
    OVH_LDP_TOKEN: OVH LDP authentication token (required if enabled)
    OVH_LDP_ENDPOINT: OVH LDP endpoint hostname (e.g., gra1.logs.ovh.com)
    OVH_LDP_PORT: OVH LDP port (default: 12202 for GELF TCP)
    OVH_LDP_PROTOCOL: Protocol to use (gelf_tcp, gelf_udp, syslog_tcp, syslog_udp)
    OVH_LDP_USE_TLS: Use TLS/SSL encryption (true/false, default: true)
    OVH_LDP_FACILITY: Syslog facility for categorization (default: user)
    OVH_LDP_ADDITIONAL_FIELDS: JSON string of additional fields to include
    
    OTLP_ENABLED: Enable OpenTelemetry Protocol (OTLP) logging (true/false)
                  Auto-enabled if ENABLE_LOGGER=true and OTLP_ENDPOINT is defined
    OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4318/v1/logs)
    OTLP_HEADERS: JSON string of HTTP headers for authentication
    OTLP_TIMEOUT: Request timeout in seconds (default: 10)
    OTLP_INSECURE: Skip SSL certificate verification (true/false, default: false)
    OTLP_SERVICE_NAME: Service name for resource attributes. Defaults to APP_NAME,
                       then to the app_name passed to setup_logging(), then core-lib.
    OTLP_SERVICE_VERSION: Service version for resource attributes (default: from pyproject.toml)
    OTLP_LOG_CHANNEL: Optional log routing channel. When set, core-lib sends
                      faciliter.log_channel=<value> as an OTLP resource attribute.
    OTLP_LOG_LEVEL: Log level for OTLP handler only (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    If not set, inherits from LOG_LEVEL
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base_settings import BaseSettings, SettingsError, EnvParser

# Try to import tomllib (Python 3.11+) or fallback to tomli
try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LoggerSettings(BaseSettings):
    """Logger configuration settings with OVH Logs Data Platform support.
    
    Supports standard file/console logging plus optional OVH LDP integration
    via GELF (Graylog Extended Log Format) protocol.
    """
    
    # Standard logging settings
    log_level: str = "INFO"
    file_logging: bool = False
    file_path: Optional[str] = None
    file_max_bytes: int = 1_048_576  # 1 MB
    file_backup_count: int = 3
    
    # OVH Logs Data Platform settings
    ovh_ldp_enabled: bool = False
    ovh_ldp_token: Optional[str] = None
    ovh_ldp_endpoint: Optional[str] = None
    ovh_ldp_port: int = 12202  # Default GELF TCP port
    ovh_ldp_protocol: str = "gelf_tcp"  # gelf_tcp, gelf_udp, syslog_tcp, syslog_udp
    ovh_ldp_use_tls: bool = True
    ovh_ldp_facility: str = "user"  # Syslog facility
    ovh_ldp_additional_fields: Dict[str, str] = field(default_factory=dict)
    ovh_ldp_timeout: int = 10  # Connection timeout in seconds
    ovh_ldp_compress: bool = True  # Compress GELF messages
    
    # OpenTelemetry Protocol (OTLP) settings
    otlp_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4318/v1/logs"
    otlp_headers: Dict[str, str] = field(default_factory=dict)
    otlp_timeout: int = 10
    otlp_insecure: bool = False  # Skip SSL verification
    otlp_service_name: Optional[str] = None
    otlp_service_version: Optional[str] = None
    otlp_log_channel: Optional[str] = None
    otlp_log_level: Optional[str] = None  # Independent log level for OTLP handler (if None, inherits from log_level)
    
    @staticmethod
    def _read_pyproject_version() -> Optional[str]:
        """Read version from pyproject.toml in project root.
        
        Returns:
            Version string from pyproject.toml or None if not found/parseable.
        """
        if tomllib is None:
            return None
        
        try:
            # Walk upwards from CWD looking for pyproject.toml
            cwd = Path(os.getcwd()).resolve()
            for parent in [cwd, *cwd.parents]:
                pyproject_path = parent / "pyproject.toml"
                if pyproject_path.exists():
                    with pyproject_path.open("rb") as f:
                        data = tomllib.load(f)
                    project = data.get("project") or {}
                    version = project.get("version")
                    if isinstance(version, str) and version.strip():
                        return version.strip()
                    break  # Found pyproject.toml but no version
        except Exception:
            pass
        
        return None
    
    @classmethod
    def from_env(
        cls,
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides
    ) -> "LoggerSettings":
        """Create logger settings from environment variables.
        
        Auto-enables OTLP when:
        - ENABLE_LOGGER=true (or LOG_FILE_ENABLED=true) AND
        - OTLP_ENDPOINT is defined AND
        - OTLP_ENABLED is not explicitly set to false
        
        Auto-sets service name/version:
                - otlp_service_name defaults to APP_NAME env or "core-lib"
                    and setup_logging(app_name=...) is used for programmatic defaults when unset
        - otlp_service_version defaults to version from pyproject.toml
        """
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        # Parse additional fields from JSON if provided
        additional_fields_raw = EnvParser.get_env("OVH_LDP_ADDITIONAL_FIELDS")
        additional_fields = {}
        if additional_fields_raw:
            try:
                additional_fields = json.loads(additional_fields_raw)
                if not isinstance(additional_fields, dict):
                    additional_fields = {}
            except json.JSONDecodeError:
                pass  # Silently ignore invalid JSON
        
        # Parse OTLP headers from JSON if provided
        otlp_headers_raw = EnvParser.get_env("OTLP_HEADERS")
        otlp_headers = {}
        if otlp_headers_raw:
            try:
                otlp_headers = json.loads(otlp_headers_raw)
                if not isinstance(otlp_headers, dict):
                    otlp_headers = {}
            except json.JSONDecodeError:
                pass  # Silently ignore invalid JSON
        
        # Auto-detect OTLP enablement
        # Enable OTLP if:
        # 1. ENABLE_LOGGER or LOG_FILE_ENABLED is true AND
        # 2. OTLP_ENDPOINT is defined AND
        # 3. OTLP_ENABLED is not explicitly false
        logger_enabled = (
            EnvParser.get_env("ENABLE_LOGGER", default=False, env_type=bool) or
            EnvParser.get_env("LOG_FILE_ENABLED", default=False, env_type=bool)
        )
        otlp_endpoint_defined = EnvParser.get_env("OTLP_ENDPOINT") is not None
        otlp_explicitly_disabled = EnvParser.get_env("OTLP_ENABLED") == "false"
        
        # Auto-enable if conditions met, unless explicitly disabled
        auto_enable_otlp = logger_enabled and otlp_endpoint_defined and not otlp_explicitly_disabled
        
        # Get explicit OTLP_ENABLED setting (None if not set)
        otlp_enabled_explicit = EnvParser.get_env("OTLP_ENABLED", env_type=bool) if EnvParser.get_env("OTLP_ENABLED") is not None else None
        
        # Use explicit setting if provided, otherwise use auto-detection
        otlp_enabled = otlp_enabled_explicit if otlp_enabled_explicit is not None else auto_enable_otlp
        
        # Get service name and version defaults
        default_service_name = EnvParser.get_env("APP_NAME", default="core-lib")
        default_service_version = cls._read_pyproject_version()
        
        settings_dict = {
            # Standard logging
            "log_level": EnvParser.get_env("LOG_LEVEL", default="INFO"),
            "file_logging": EnvParser.get_env("LOG_FILE_ENABLED", default=False, env_type=bool),
            "file_path": EnvParser.get_env("LOG_FILE_PATH"),
            "file_max_bytes": EnvParser.get_env("LOG_FILE_MAX_BYTES", default=1_048_576, env_type=int),
            "file_backup_count": EnvParser.get_env("LOG_FILE_BACKUP_COUNT", default=3, env_type=int),
            
            # OVH LDP
            "ovh_ldp_enabled": EnvParser.get_env("OVH_LDP_ENABLED", default=False, env_type=bool),
            "ovh_ldp_token": EnvParser.get_env("OVH_LDP_TOKEN", "OVH_LOGS_TOKEN"),
            "ovh_ldp_endpoint": EnvParser.get_env("OVH_LDP_ENDPOINT", "OVH_LOGS_ENDPOINT"),
            "ovh_ldp_port": EnvParser.get_env("OVH_LDP_PORT", default=12202, env_type=int),
            "ovh_ldp_protocol": EnvParser.get_env("OVH_LDP_PROTOCOL", default="gelf_tcp"),
            "ovh_ldp_use_tls": EnvParser.get_env("OVH_LDP_USE_TLS", default=True, env_type=bool),
            "ovh_ldp_facility": EnvParser.get_env("OVH_LDP_FACILITY", default="user"),
            "ovh_ldp_additional_fields": additional_fields,
            "ovh_ldp_timeout": EnvParser.get_env("OVH_LDP_TIMEOUT", default=10, env_type=int),
            "ovh_ldp_compress": EnvParser.get_env("OVH_LDP_COMPRESS", default=True, env_type=bool),
            
            # OTLP
            "otlp_enabled": otlp_enabled,
            "otlp_endpoint": EnvParser.get_env("OTLP_ENDPOINT", default="http://localhost:4318/v1/logs"),
            "otlp_headers": otlp_headers,
            "otlp_timeout": EnvParser.get_env("OTLP_TIMEOUT", default=10, env_type=int),
            "otlp_insecure": EnvParser.get_env("OTLP_INSECURE", default=False, env_type=bool),
            "otlp_service_name": EnvParser.get_env("OTLP_SERVICE_NAME", default=default_service_name),
            "otlp_service_version": EnvParser.get_env("OTLP_SERVICE_VERSION", default=default_service_version),
            "otlp_log_channel": EnvParser.get_env("OTLP_LOG_CHANNEL"),
            "otlp_log_level": EnvParser.get_env("OTLP_LOG_LEVEL"),  # Optional: independent OTLP log level
        }
        
        settings_dict.update(overrides)
        return cls(**settings_dict)
    
    def validate(self) -> None:
        """Validate logger configuration."""
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise SettingsError(
                f"Invalid log level '{self.log_level}'. Must be one of: {', '.join(valid_log_levels)}"
            )
        
        # Validate OVH LDP settings
        if self.ovh_ldp_enabled:
            if not self.ovh_ldp_token:
                raise SettingsError("OVH LDP enabled but ovh_ldp_token not provided")
            if not self.ovh_ldp_endpoint:
                raise SettingsError("OVH LDP enabled but ovh_ldp_endpoint not provided")
            
            valid_protocols = ["gelf_tcp", "gelf_udp", "syslog_tcp", "syslog_udp"]
            if self.ovh_ldp_protocol.lower() not in valid_protocols:
                raise SettingsError(
                    f"Invalid OVH LDP protocol '{self.ovh_ldp_protocol}'. "
                    f"Must be one of: {', '.join(valid_protocols)}"
                )
            
            if self.ovh_ldp_port < 1 or self.ovh_ldp_port > 65535:
                raise SettingsError(f"Invalid port {self.ovh_ldp_port}. Must be between 1 and 65535")
            
            if self.ovh_ldp_timeout < 1:
                raise SettingsError("OVH LDP timeout must be at least 1 second")
        
        # Validate OTLP log level if specified
        if self.otlp_log_level is not None:
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self.otlp_log_level.upper() not in valid_log_levels:
                raise SettingsError(
                    f"Invalid OTLP log level '{self.otlp_log_level}'. Must be one of: {', '.join(valid_log_levels)}"
                )

        if self.otlp_log_channel is not None and not self.otlp_log_channel.strip():
            raise SettingsError("OTLP log channel cannot be empty when provided")
    
    def as_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "log_level": self.log_level,
            "file_logging": self.file_logging,
            "file_path": self.file_path,
            "file_max_bytes": self.file_max_bytes,
            "file_backup_count": self.file_backup_count,
            "ovh_ldp_enabled": self.ovh_ldp_enabled,
            "ovh_ldp_token": "***" if self.ovh_ldp_token else None,  # Mask token
            "ovh_ldp_endpoint": self.ovh_ldp_endpoint,
            "ovh_ldp_port": self.ovh_ldp_port,
            "ovh_ldp_protocol": self.ovh_ldp_protocol,
            "ovh_ldp_use_tls": self.ovh_ldp_use_tls,
            "ovh_ldp_facility": self.ovh_ldp_facility,
            "ovh_ldp_additional_fields": self.ovh_ldp_additional_fields,
            "ovh_ldp_timeout": self.ovh_ldp_timeout,
            "ovh_ldp_compress": self.ovh_ldp_compress,
            "otlp_enabled": self.otlp_enabled,
            "otlp_endpoint": self.otlp_endpoint,
            "otlp_headers": self.otlp_headers,
            "otlp_timeout": self.otlp_timeout,
            "otlp_insecure": self.otlp_insecure,
            "otlp_service_name": self.otlp_service_name,
            "otlp_service_version": self.otlp_service_version,
            "otlp_log_channel": self.otlp_log_channel,
            "otlp_log_level": self.otlp_log_level,
        }
