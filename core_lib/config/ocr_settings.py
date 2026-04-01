"""OCR Service Configuration Settings.

Configuration for document OCR services. Vision-capable LLM providers
(via llm_providers.yaml with ``usage: [vision, ocr]``) are the primary
OCR backend. dots-ocr is an optional high-performance alternative that
is activated only when ``DOTS_OCR_BASE_URL`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base_settings import BaseSettings, EnvParser


@dataclass(frozen=True)
class OcrSettings(BaseSettings):
    """OCR service configuration settings.

    Vision-capable LLM providers (configured via ``llm_providers.yaml``
    with ``usage: [vision, ocr]``) are the primary OCR backend.

    dots-ocr (a vLLM-hosted vision model) is an optional alternative
    backend: set ``DOTS_OCR_BASE_URL`` to activate it. In the default
    service implementation it is used as a fallback when the vision
    LLM is unavailable or fails, rather than being called after a
    successful vision LLM response.
    """

    # dots-ocr service configuration (optional — None means not configured)
    dots_ocr_base_url: Optional[str] = None
    dots_ocr_api_key: Optional[str] = None
    dots_ocr_model: str = "dots-ocr"
    dots_ocr_timeout: int = 120  # VLM inference is slow
    dots_ocr_max_tokens: int = 32768

    # Vision LLM timeout — higher than the default LLM timeout because
    # local models process images slowly (large base64 payloads).
    # When set, overrides the vision LLM client's configured timeout.
    vision_llm_timeout: int = 180  # seconds

    # Image filtering thresholds — skip icons, logos, and decorative images
    min_image_width: int = 200
    min_image_height: int = 200
    min_image_bytes: int = 20480  # 20 KB

    # Vision LLM output cap — limits the number of tokens the vision model
    # can generate per image.  Prevents small local models from producing
    # excessively long output (or looping) with complex prompts such as
    # the enriched OCR prompt.  0 = no limit (provider default).
    vision_max_output_tokens: int = 2048

    # Image optimisation — resize and compress before sending to vision LLM
    # to reduce base64 payload size and speed up local model inference.
    max_image_dimension: int = 1568   # px, longest side; 0 = no resize
    ocr_jpeg_quality: int = 85        # JPEG quality for PNG→JPEG conversion

    # OCR generation settings
    ocr_temperature: float = 0.1

    # Cache settings for deduplication (seconds)
    ocr_cache_ttl: int = 86400  # 24 hours

    @classmethod
    def from_env(cls, load_dotenv: bool = False, **overrides) -> "OcrSettings":
        """Create OCR settings from environment variables."""
        if load_dotenv:
            cls._load_dotenv_if_requested(load_dotenv)

        settings: Dict[str, Any] = {}

        dots_url = EnvParser.get_env("DOTS_OCR_BASE_URL") or EnvParser.get_env("DOTS_OCR_URL")
        if dots_url:
            settings["dots_ocr_base_url"] = dots_url

        dots_key = EnvParser.get_env("DOTS_OCR_API_KEY")
        if dots_key:
            settings["dots_ocr_api_key"] = dots_key

        dots_model = EnvParser.get_env("DOTS_OCR_MODEL")
        if dots_model:
            settings["dots_ocr_model"] = dots_model

        dots_timeout = EnvParser.get_env("DOTS_OCR_TIMEOUT", env_type=int)
        if dots_timeout is not None:
            settings["dots_ocr_timeout"] = dots_timeout

        dots_max_tokens = EnvParser.get_env("DOTS_OCR_MAX_TOKENS", env_type=int)
        if dots_max_tokens is not None:
            settings["dots_ocr_max_tokens"] = dots_max_tokens

        min_w = EnvParser.get_env("OCR_MIN_IMAGE_WIDTH", env_type=int)
        if min_w is not None:
            settings["min_image_width"] = min_w

        min_h = EnvParser.get_env("OCR_MIN_IMAGE_HEIGHT", env_type=int)
        if min_h is not None:
            settings["min_image_height"] = min_h

        min_b = EnvParser.get_env("OCR_MIN_IMAGE_BYTES", env_type=int)
        if min_b is not None:
            settings["min_image_bytes"] = min_b

        vision_timeout = EnvParser.get_env("OCR_VISION_LLM_TIMEOUT", env_type=int)
        if vision_timeout is not None:
            settings["vision_llm_timeout"] = vision_timeout

        temp = EnvParser.get_env("OCR_TEMPERATURE", env_type=float)
        if temp is not None:
            settings["ocr_temperature"] = temp

        vision_max_tokens = EnvParser.get_env("OCR_VISION_MAX_OUTPUT_TOKENS", env_type=int)
        if vision_max_tokens is not None:
            settings["vision_max_output_tokens"] = vision_max_tokens

        max_dim = EnvParser.get_env("OCR_MAX_IMAGE_DIMENSION", env_type=int)
        if max_dim is not None:
            settings["max_image_dimension"] = max_dim

        jpeg_q = EnvParser.get_env("OCR_JPEG_QUALITY", env_type=int)
        if jpeg_q is not None:
            settings["ocr_jpeg_quality"] = jpeg_q

        cache_ttl = EnvParser.get_env("OCR_CACHE_TTL", env_type=int)
        if cache_ttl is not None:
            settings["ocr_cache_ttl"] = cache_ttl

        settings.update(overrides)
        return cls(**settings)

    def validate(self) -> None:
        """Validate OCR settings."""
        if self.dots_ocr_timeout < 1:
            raise ValueError("dots_ocr_timeout must be >= 1")
        if self.min_image_width < 1 or self.min_image_height < 1:
            raise ValueError("min_image_width and min_image_height must be >= 1")
        if self.min_image_bytes < 0:
            raise ValueError("min_image_bytes must be >= 0")
