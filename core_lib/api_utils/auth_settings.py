"""Authentication settings configuration.

Configuration for time-based authentication using private keys
and/or static API keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, List, Optional, Set, Union

from ..config.base_settings import BaseSettings, EnvParser, SettingsError


def _parse_keys(keys_str: Optional[str]) -> FrozenSet[str]:
    """Parse comma-separated keys into a frozen set.
    
    Args:
        keys_str: Comma-separated string of keys, or None
        
    Returns:
        Frozen set of non-empty stripped keys
    """
    if not keys_str:
        return frozenset()
    return frozenset(k.strip() for k in keys_str.split(",") if k.strip())


@dataclass(frozen=True)
class AuthSettings(BaseSettings):
    """Authentication settings supporting both time-based HMAC and static API keys.
    
    This configuration supports two authentication methods:
    1. Time-based HMAC: Generates keys valid for a 3-hour window
    2. Static API keys: Pre-defined keys for simple authentication
    
    Multiple keys are supported for both methods to enable key rotation.
    
    Environment Variables:
        AUTH_ENABLED: Whether authentication is enabled (default: False)
        AUTH_PRIVATE_KEY: Secret key(s) for time-based auth (comma-separated for rotation)
        AUTH_STATIC_KEYS: Static API key(s) that are always valid (comma-separated)
        AUTH_KEY_HEADER_NAME: HTTP header name for auth key (default: x-auth-key)
    
    Example:
        ```python
        # From environment
        settings = AuthSettings.from_env()
        
        # With both time-based and static keys
        # AUTH_ENABLED=true
        # AUTH_PRIVATE_KEY=hmac-secret-1,hmac-secret-2
        # AUTH_STATIC_KEYS=static-key-1,static-key-2,static-key-3
        
        settings = AuthSettings.from_env(
            auth_enabled=True,
            auth_private_key="my-secret-key",
            auth_static_keys="api-key-1,api-key-2"
        )
        ```
    """
    
    auth_enabled: bool = False
    auth_private_key: Optional[str] = None  # Primary private key (first if multiple)
    auth_private_keys: FrozenSet[str] = field(default_factory=frozenset)  # All private keys
    auth_static_keys: FrozenSet[str] = field(default_factory=frozenset)  # Static API keys
    auth_key_header_name: str = "x-auth-key"
    
    @classmethod
    def from_env(
        cls,
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides
    ) -> "AuthSettings":
        """Create authentication settings from environment variables.
        
        Args:
            load_dotenv: Whether to load .env files
            dotenv_paths: Custom paths to search for .env files
            **overrides: Direct value overrides
            
        Returns:
            AuthSettings instance
        """
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        # Get raw private key string (may contain multiple comma-separated keys)
        raw_private_key = EnvParser.get_env("AUTH_PRIVATE_KEY", default=None)
        private_keys = _parse_keys(raw_private_key)
        
        # First key is the primary (used for generation)
        primary_key = None
        if raw_private_key:
            first_key = raw_private_key.split(",")[0].strip()
            primary_key = first_key if first_key else None
        
        # Get static API keys
        raw_static_keys = EnvParser.get_env("AUTH_STATIC_KEYS", default=None)
        static_keys = _parse_keys(raw_static_keys)
        
        settings_dict = {
            "auth_enabled": EnvParser.get_env(
                "AUTH_ENABLED",
                default=False,
                env_type=bool
            ),
            "auth_private_key": primary_key,
            "auth_private_keys": private_keys,
            "auth_static_keys": static_keys,
            "auth_key_header_name": EnvParser.get_env(
                "AUTH_KEY_HEADER_NAME",
                default="x-auth-key"
            ),
        }
        
        # Handle string overrides for keys (convert to frozenset)
        if "auth_private_key" in overrides and isinstance(overrides["auth_private_key"], str):
            override_keys = _parse_keys(overrides["auth_private_key"])
            overrides["auth_private_keys"] = override_keys
            first = overrides["auth_private_key"].split(",")[0].strip()
            overrides["auth_private_key"] = first if first else None
        
        if "auth_static_keys" in overrides and isinstance(overrides["auth_static_keys"], str):
            overrides["auth_static_keys"] = _parse_keys(overrides["auth_static_keys"])
        
        settings_dict.update(overrides)
        return cls(**settings_dict)
    
    def validate(self) -> None:
        """Validate authentication settings.
        
        Raises:
            SettingsError: If auth is enabled but no valid keys are configured
        """
        if self.auth_enabled:
            has_private_keys = bool(self.auth_private_keys)
            has_static_keys = bool(self.auth_static_keys)
            
            if not has_private_keys and not has_static_keys:
                raise SettingsError(
                    "AUTH_PRIVATE_KEY and/or AUTH_STATIC_KEYS must be set when AUTH_ENABLED is True"
                )
            
            # Validate private key length for security
            for key in self.auth_private_keys:
                if len(key) < 16:
                    raise SettingsError(
                        "AUTH_PRIVATE_KEY entries should be at least 16 characters for security"
                    )
            
            # Validate static key length for security
            for key in self.auth_static_keys:
                if len(key) < 16:
                    raise SettingsError(
                        "AUTH_STATIC_KEYS entries should be at least 16 characters for security"
                    )
        
        if not self.auth_key_header_name or not self.auth_key_header_name.strip():
            raise SettingsError("auth_key_header_name cannot be empty")
    
    def is_valid_static_key(self, key: str) -> bool:
        """Check if the provided key is a valid static API key.
        
        Args:
            key: The key to check
            
        Returns:
            True if the key is in the static keys set
        """
        if not key or not self.auth_static_keys:
            return False
        return key in self.auth_static_keys
    
    def has_hmac_auth(self) -> bool:
        """Check if time-based HMAC authentication is configured."""
        return bool(self.auth_private_keys)
    
    def has_static_auth(self) -> bool:
        """Check if static API key authentication is configured."""
        return bool(self.auth_static_keys)
    
    def as_dict(self) -> dict:
        """Convert to dictionary representation.
        
        Note: Excludes actual keys for security when serializing.
        """
        return {
            "auth_enabled": self.auth_enabled,
            "auth_private_key": "***" if self.auth_private_key else None,
            "auth_private_keys_count": len(self.auth_private_keys),
            "auth_static_keys_count": len(self.auth_static_keys),
            "auth_key_header_name": self.auth_key_header_name,
            "has_hmac_auth": self.has_hmac_auth(),
            "has_static_auth": self.has_static_auth(),
        }
