"""LLM Payload Capture Configuration Settings.

Configuration for optionally persisting full LLM request/response payloads
(prompt + completion text) to S3-compatible object storage, so they can be
inspected later (e.g. via saas-admin's LLM Call Inspector) without paying to
keep that large text content in OpenSearch or the primary database.

Disabled by default. Reuses the same AWS_S3_* credentials/env vars already
used elsewhere for media storage, so enabling this in an environment that
already has S3 configured requires no duplicate credential setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_settings import BaseSettings, EnvParser, SettingsError


@dataclass(frozen=True)
class PayloadCaptureSettings(BaseSettings):
    """Settings for capturing LLM prompt/response payloads to object storage."""

    enabled: bool = False
    s3_bucket: str = ""
    s3_endpoint_url: Optional[str] = None
    s3_region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    retention_days: int = 30
    max_chars: int = 200_000

    @classmethod
    def from_env(
        cls,
        load_dotenv: bool = True,
        dotenv_paths: Optional[List[Union[str, Path]]] = None,
        **overrides,
    ) -> "PayloadCaptureSettings":
        """Create payload capture settings from environment variables."""
        cls._load_dotenv_if_requested(load_dotenv, dotenv_paths)

        settings_dict = {
            "enabled": EnvParser.get_env("LLM_PAYLOAD_CAPTURE_ENABLED", default=False, env_type=bool),
            "s3_bucket": EnvParser.get_env("LLM_PAYLOAD_S3_BUCKET", default=""),
            # Fall back to the shared AWS_S3_* config already used for media storage.
            "s3_endpoint_url": EnvParser.get_env("LLM_PAYLOAD_S3_ENDPOINT_URL", "AWS_S3_ENDPOINT_URL"),
            "s3_region": EnvParser.get_env("LLM_PAYLOAD_S3_REGION_NAME", "AWS_S3_REGION_NAME"),
            "aws_access_key_id": EnvParser.get_env("LLM_PAYLOAD_AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": EnvParser.get_env("LLM_PAYLOAD_AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
            "retention_days": EnvParser.get_env("LLM_PAYLOAD_RETENTION_DAYS", default=30, env_type=int),
            "max_chars": EnvParser.get_env("LLM_PAYLOAD_MAX_CHARS", default=200_000, env_type=int),
        }

        settings_dict.update(overrides)
        return cls(**settings_dict)

    def validate(self) -> None:
        """Validate payload capture configuration."""
        if self.enabled and not self.s3_bucket:
            raise SettingsError("LLM_PAYLOAD_S3_BUCKET must be set when LLM_PAYLOAD_CAPTURE_ENABLED is true")
        if self.retention_days <= 0:
            raise SettingsError("Retention days must be positive")
        if self.max_chars <= 0:
            raise SettingsError("Max chars must be positive")

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "enabled": self.enabled,
            "s3_bucket": self.s3_bucket,
            "s3_endpoint_url": self.s3_endpoint_url,
            "s3_region": self.s3_region,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": "***" if self.aws_secret_access_key else None,
            "retention_days": self.retention_days,
            "max_chars": self.max_chars,
        }
