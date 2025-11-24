"""
base_settings.py

Extensible base settings class for AI projects.

This module provides a flexible foundation for managing application configuration
with support for environment variables, .env files, type conversion, validation,
and easy extension for custom settings.

Key Features:
- Automatic .env file loading with configurable search paths
- Type-safe environment variable parsing with defaults
- Composition-friendly design for combining multiple config classes
- Support for nested configuration objects
- Validation and error handling
- Easy extension for project-specific settings
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_type_hints
import warnings

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

T = TypeVar('T')


class SettingsError(Exception):
    """Base exception for settings-related errors."""
    pass


class EnvironmentVariableError(SettingsError):
    """Raised when required environment variables are missing or invalid."""
    pass


class DotEnvLoader:
    """Handles .env file discovery and loading."""
    
    @staticmethod
    def load_dotenv_files(
        search_paths: Optional[List[Union[str, Path]]] = None,
        filename: str = ".env"
    ) -> bool:
        """Load environment variables from .env files.
        
        Args:
            search_paths: Directories to search for .env files. If None, searches:
                         1. Current working directory
                         2. Project root (if pyproject.toml found)
                         3. User home directory
            filename: Name of the env file (default: ".env")
            
        Returns:
            True if at least one .env file was loaded, False otherwise
        """
        if not HAS_DOTENV:
            return False
            
        if search_paths is None:
            search_paths = DotEnvLoader._get_default_search_paths()
        
        loaded = False
        for path in search_paths:
            env_file = Path(path) / filename
            if env_file.exists():
                load_dotenv(env_file, override=False)  # Don't override existing env vars
                loaded = True
                
        return loaded
    
    @staticmethod
    def _get_default_search_paths() -> List[Path]:
        """Get default search paths for .env files."""
        paths = []
        
        # Current working directory
        paths.append(Path.cwd())
        
        # Project root (walk up looking for pyproject.toml)
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            if (parent / "pyproject.toml").exists():
                paths.append(parent)
                break
        
        # User home directory
        paths.append(Path.home())
        
        return paths


class EnvParser:
    """Utilities for parsing environment variables with type conversion."""
    
    @staticmethod
    def get_env(
        *names: str, 
        default: Any = None, 
        required: bool = False,
        env_type: Type[T] = str
    ) -> Optional[T]:
        """Get environment variable with type conversion and fallback.
        
        Args:
            *names: Environment variable names to try in order
            default: Default value if no env var is found
            required: If True, raise EnvironmentVariableError if no value found
            env_type: Type to convert the value to (str, int, float, bool, list)
            
        Returns:
            Parsed environment variable value or default
            
        Raises:
            EnvironmentVariableError: If required=True and no value found
        """
        value = None
        for name in names:
            value = os.getenv(name)
            if value is not None:
                break
                
        if value is None:
            if required:
                raise EnvironmentVariableError(
                    f"Required environment variable not found: {', '.join(names)}"
                )
            return default
            
        return EnvParser._convert_type(value, env_type, names[0])
    
    @staticmethod
    def _convert_type(value: str, target_type: Type[T], var_name: str) -> T:
        """Convert string value to target type."""
        if target_type == str:
            return value  # type: ignore
        elif target_type == int:
            try:
                # Handle potential inline comments (e.g. "86400 # 1 day")
                clean_value = value.split('#')[0].strip()
                return int(clean_value)  # type: ignore
            except ValueError:
                raise EnvironmentVariableError(
                    f"Invalid integer value for {var_name}: {value}"
                )
        elif target_type == float:
            try:
                # Handle potential inline comments
                clean_value = value.split('#')[0].strip()
                return float(clean_value)  # type: ignore
            except ValueError:
                raise EnvironmentVariableError(
                    f"Invalid float value for {var_name}: {value}"
                )
        elif target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')  # type: ignore
        elif target_type == list:
            # Split by comma, strip whitespace
            return [item.strip() for item in value.split(',') if item.strip()]  # type: ignore
        else:
            # For other types, try direct conversion
            try:
                return target_type(value)  # type: ignore
            except (ValueError, TypeError):
                raise EnvironmentVariableError(
                    f"Cannot convert {var_name} value '{value}' to {target_type.__name__}"
                )


@dataclass(frozen=True)
class BaseSettings(ABC):
    """Abstract base class for application settings.
    
    This class provides a foundation for building configuration classes that:
    - Load from environment variables with type safety
    - Support .env files with automatic discovery
    - Allow easy composition and extension
    - Provide validation and error handling
    
    Subclasses should:
    1. Define their fields as dataclass fields with defaults
    2. Implement `from_env()` to populate from environment variables
    3. Optionally override `validate()` for custom validation
    4. Use `get_env()` helper for environment variable parsing
    """
    
    # Metadata fields (not part of configuration)
    _env_loaded: bool = field(default=False, init=False, repr=False)
    _validation_errors: List[str] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        """Post-initialization validation."""
        object.__setattr__(self, '_validation_errors', [])
        try:
            self.validate()
        except Exception as e:
            object.__setattr__(self, '_validation_errors', [str(e)])
    
    @classmethod
    @abstractmethod
    def from_env(
        cls, 
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides
    ) -> "BaseSettings":
        """Create instance from environment variables.
        
        Args:
            load_dotenv: Whether to load .env files before reading env vars
            dotenv_paths: Custom paths to search for .env files
            **overrides: Direct value overrides (bypass environment variables)
            
        Returns:
            Configured settings instance
        """
        raise NotImplementedError
    
    def validate(self) -> None:
        """Validate configuration values.
        
        Override this method to add custom validation logic.
        Raise SettingsError or subclass for validation failures.
        """
        pass
    
    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self._validation_errors) == 0
    
    @property
    def validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_errors.copy()
    
    def get_env(self, *names: str, **kwargs) -> Any:
        """Convenience method for environment variable parsing."""
        return EnvParser.get_env(*names, **kwargs)
    
    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary, excluding metadata fields."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Exclude metadata fields
                if hasattr(value, 'as_dict'):
                    result[key] = value.as_dict()
                else:
                    result[key] = value
        return result
    
    def merge(self, **overrides) -> "BaseSettings":
        """Create a new instance with specified overrides.
        
        Args:
            **overrides: Field values to override
            
        Returns:
            New settings instance with overrides applied
        """
        current_dict = self.as_dict()
        current_dict.update(overrides)
        return type(self)(**current_dict)
    
    @classmethod
    def _load_dotenv_if_requested(
        cls, 
        load_dotenv: bool, 
        dotenv_paths: Optional[List[Union[str, Path]]]
    ) -> bool:
        """Load .env files if requested."""
        if load_dotenv:
            return DotEnvLoader.load_dotenv_files(dotenv_paths)
        return False


class SettingsManager:
    """Manages multiple settings instances and provides unified access."""
    
    def __init__(self):
        self._settings: Dict[str, BaseSettings] = {}
    
    def register(self, name: str, settings: BaseSettings) -> None:
        """Register a settings instance."""
        if not settings.is_valid:
            warnings.warn(
                f"Registering invalid settings '{name}': {settings.validation_errors}"
            )
        self._settings[name] = settings
    
    def get(self, name: str) -> Optional[BaseSettings]:
        """Get settings by name."""
        return self._settings.get(name)
    
    def get_required(self, name: str) -> BaseSettings:
        """Get settings by name, raising if not found."""
        settings = self._settings.get(name)
        if settings is None:
            raise SettingsError(f"Settings '{name}' not found")
        return settings
    
    def list_names(self) -> List[str]:
        """List all registered settings names."""
        return list(self._settings.keys())
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all registered settings and return errors."""
        errors = {}
        for name, settings in self._settings.items():
            if not settings.is_valid:
                errors[name] = settings.validation_errors
        return errors
    
    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return all settings as nested dictionary."""
        return {name: settings.as_dict() for name, settings in self._settings.items()}


# Global settings manager instance
settings_manager = SettingsManager()


class NullConfig:
    """Null-object for configuration sections.

    - Any attribute access returns None
    - Falsy in boolean context
    - `as_dict()` returns None
    """

    def __getattr__(self, name: str):  # noqa: D401
        return None

    def __bool__(self) -> bool:
        return False

    def as_dict(self):
        return None