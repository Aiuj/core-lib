"""Application Core Configuration Settings.

This module contains the core application settings that extend
the base AppSettings with BaseSettings compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ImportError:
    tomllib = None  # type: ignore[assignment]

from .base_settings import BaseSettings


@dataclass(frozen=True)
class AppSettings(BaseSettings):
    """Core application settings with BaseSettings compatibility.
    
    Extends the base AppSettings with environment variable parsing
    and validation capabilities from BaseSettings.
    """
    
    app_name: str = "app"
    version: str = "0.2.8"
    environment: str = "dev"
    log_level: str = "DEBUG"
    project_root: Optional[Path] = None
    
    @classmethod
    def from_env(
        cls,
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides
    ) -> "AppSettings":
        """Create app settings from environment variables."""
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)
        
        # Resolve project root
        project_root = overrides.get("project_root")
        explicit_project_root = project_root is not None
        resolved_root = cls._resolve_project_root(project_root)
        
        # Resolve app name
        app_name = overrides.get("app_name")
        if app_name is None:
            # Try pyproject.toml
            if explicit_project_root and resolved_root:
                pyproject = resolved_root / "pyproject.toml"
                if pyproject.exists():
                    data = cls._read_pyproject_data(pyproject)
                    if data and data.get("name"):
                        app_name = data["name"]
            
            # Fallback to env var or default
            if app_name is None:
                app_name = os.getenv("APP_NAME", "app")
        
        # Resolve version
        version = overrides.get("version")
        if version is None:
            version = cls._resolve_version(resolved_root)
            
        # Resolve environment
        environment = overrides.get("environment")
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "dev").lower()
            
        # Resolve log level
        log_level = overrides.get("log_level")
        if log_level is None:
            default_level = "DEBUG" if environment == "dev" else "INFO"
            log_level = os.getenv("LOG_LEVEL", default_level).upper()
            
        return cls(
            app_name=app_name,
            version=version,
            environment=environment,
            log_level=log_level,
            project_root=resolved_root
        )

    @staticmethod
    def _resolve_project_root(project_root: Union[str, Path, None]) -> Optional[Path]:
        """Resolve the project root path."""
        if project_root is not None:
            root = Path(project_root).resolve()
            return root if root.exists() else None

        # Walk upwards from CWD looking for pyproject.toml
        try:
            cwd = Path(os.getcwd()).resolve()
            for parent in [cwd, *cwd.parents]:
                if (parent / "pyproject.toml").exists():
                    return parent
        except Exception:
            pass
        return None

    @staticmethod
    def _read_pyproject_data(pyproject_path: Path) -> Optional[dict]:
        """Read project metadata from a pyproject.toml file."""
        if tomllib is None:
            return None
        try:
            with pyproject_path.open("rb") as f:
                data = tomllib.load(f)
            project = data.get("project") or {}
            return {
                "name": project.get("name", "").strip() if isinstance(project.get("name"), str) else "",
                "version": project.get("version", "").strip() if isinstance(project.get("version"), str) else "",
            }
        except Exception:
            return None

    @classmethod
    def _resolve_version(cls, project_root: Optional[Path]) -> str:
        """Resolve application version."""
        if project_root is not None:
            pyproject = project_root / "pyproject.toml"
            if pyproject.exists():
                data = cls._read_pyproject_data(pyproject)
                if data and data.get("version"):
                    return data["version"]

        return os.getenv("APP_VERSION", "0.1.0")
    
    def validate(self) -> None:
        """Validate app settings."""
        if not self.app_name or not self.app_name.strip():
            raise ValueError("App name cannot be empty")
        
        valid_environments = ["dev", "development", "staging", "prod", "production", "test", "testing"]
        if self.environment.lower() not in valid_environments:
            # Don't raise error, just warn - environment can be custom
            pass
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_log_levels)}")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ("dev", "development")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() in ("prod", "production")
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment.lower() in ("test", "testing")
    
    def as_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "environment": self.environment,
            "log_level": self.log_level,
            "project_root": str(self.project_root) if self.project_root else None,
        }
