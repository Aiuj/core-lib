import os
from pathlib import Path

from core_lib.config.app_settings import AppSettings


def test_app_settings_reads_version_from_pyproject(tmp_path: Path, monkeypatch):
    # Create a fake project with a pyproject.toml
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        """
[project]
name = "dummy"
version = "9.8.7"
""".strip()
    )

    # Even if APP_VERSION is set, pyproject should take precedence
    monkeypatch.setenv("APP_VERSION", "1.2.3-env")

    s = AppSettings.from_env(app_name="Dummy", project_root=project_dir, load_dotenv=False)
    assert s.version == "9.8.7"


def test_app_settings_reads_name_from_pyproject(tmp_path: Path, monkeypatch):
    # Create a fake project with a pyproject.toml
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        """
[project]
name = "my-awesome-app"
version = "1.2.3"
""".strip()
    )

    # When app_name is None, should read from pyproject.toml
    monkeypatch.delenv("APP_NAME", raising=False)
    s = AppSettings.from_env(app_name=None, project_root=project_dir, load_dotenv=False)
    assert s.app_name == "my-awesome-app"
    assert s.version == "1.2.3"
    
    # Even if APP_NAME env var is set, pyproject should take precedence
    monkeypatch.setenv("APP_NAME", "env-app")
    s2 = AppSettings.from_env(app_name=None, project_root=project_dir, load_dotenv=False)
    assert s2.app_name == "my-awesome-app"
    
    # But explicit app_name parameter should override everything
    s3 = AppSettings.from_env(app_name="explicit-app", project_root=project_dir, load_dotenv=False)

    assert s3.app_name == "explicit-app"


def test_app_settings_env_and_defaults(monkeypatch, tmp_path: Path):
    # Use an explicit root so an ancestor of the temp folder cannot be detected.
    monkeypatch.chdir(tmp_path)

    # No project_root pyproject.toml -> fallback to env, then default
    monkeypatch.delenv("APP_VERSION", raising=False)
    s_default = AppSettings.from_env(app_name="NoProj", project_root=tmp_path, load_dotenv=False)
    assert s_default.version == "0.1.0"

    monkeypatch.setenv("APP_VERSION", "2.0.0")
    s_env = AppSettings.from_env(app_name="NoProj", project_root=tmp_path, load_dotenv=False)
    assert s_env.version == "2.0.0"


def test_app_settings_log_level(monkeypatch, tmp_path: Path):
    # Prepare pyproject
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='0.0.1'\n")

    # ENVIRONMENT=dev -> default DEBUG
    monkeypatch.setenv("ENVIRONMENT", "dev")
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    s_dev = AppSettings.from_env(app_name="x", project_root=tmp_path, load_dotenv=False)
    assert s_dev.log_level == "DEBUG"

    # ENVIRONMENT=prod -> default INFO unless overridden
    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    s_prod = AppSettings.from_env(app_name="x", project_root=tmp_path, load_dotenv=False)
    assert s_prod.log_level == "INFO"

    # Override
    monkeypatch.setenv("LOG_LEVEL", "warning")
    s_warn = AppSettings.from_env(app_name="x", project_root=tmp_path, load_dotenv=False)
    assert s_warn.log_level == "WARNING"
