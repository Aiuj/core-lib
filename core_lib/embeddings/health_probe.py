"""Per-provider health probes for embedding providers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from core_lib import get_module_logger

from ..config.embeddings_settings import EmbeddingsSettings
from ..config.provider_chain_utils import build_provider_chain
from .factory import EmbeddingFactory

logger = get_module_logger()


@dataclass
class EmbeddingProviderHealthResult:
    provider: str
    model: str
    priority: int
    healthy: bool
    error: Optional[str]
    latency_ms: Optional[float]


def _close_client(client) -> None:
    close_method = getattr(client, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception:
            pass


def _probe_single_provider(config: dict) -> str | None:
    client = None
    try:
        provider = config.get("provider")
        model = config.get("model")
        kwargs = {k: v for k, v in config.items() if k not in {"provider", "model"}}
        client = EmbeddingFactory.create(provider=provider, model=model, **kwargs)
        return None if client.health_check() else "health_check returned False"
    except Exception as exc:
        return str(exc)
    finally:
        if client is not None:
            _close_client(client)


def _config_from_settings(settings: EmbeddingsSettings) -> dict:
    config = {
        "provider": settings.provider,
        "model": settings.model,
    }

    if settings.embedding_dimension is not None:
        config["embedding_dim"] = settings.embedding_dimension

    if settings.provider == "openai":
        if settings.api_key:
            config["api_key"] = settings.api_key
        if settings.base_url:
            config["base_url"] = settings.base_url
        if settings.organization:
            config["organization"] = settings.organization
        if settings.project:
            config["project"] = settings.project
    elif settings.provider in {"google_genai", "google", "gemini"}:
        if settings.google_api_key:
            config["api_key"] = settings.google_api_key
        if settings.task_type:
            config["task_type"] = settings.task_type
        if settings.title:
            config["title"] = settings.title
    elif settings.provider == "infinity":
        if settings.infinity_url:
            config["base_url"] = settings.infinity_url
        if settings.infinity_timeout is not None:
            config["timeout"] = settings.infinity_timeout
        if settings.infinity_token:
            config["token"] = settings.infinity_token
        if settings.infinity_wake_on_lan:
            config["wake_on_lan"] = settings.infinity_wake_on_lan
    elif settings.provider == "ollama":
        if settings.ollama_url:
            config["base_url"] = settings.ollama_url
        elif settings.ollama_host:
            config["base_url"] = settings.ollama_host
        if settings.ollama_timeout is not None:
            config["timeout"] = settings.ollama_timeout
    elif settings.provider in {"local", "huggingface"}:
        config["device"] = settings.device
        if settings.cache_dir:
            config["cache_dir"] = settings.cache_dir
        config["trust_remote_code"] = settings.trust_remote_code
        config["use_sentence_transformers"] = settings.use_sentence_transformers

    return config


def check_embedding_providers_health(
    provider_configs: Iterable[dict] | None = None,
) -> List[EmbeddingProviderHealthResult]:
    """Probe configured embedding providers and return per-provider health results."""
    if provider_configs is None:
        try:
            settings = EmbeddingsSettings.from_env(load_dotenv=False)
        except Exception as exc:
            logger.warning("Embedding health check: failed to load settings: %s", exc)
            return []

        provider_configs = list(getattr(settings, "provider_configs", ()) or ())
        if not provider_configs:
            start = time.monotonic()
            error = _probe_single_provider(_config_from_settings(settings))
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            return [
                EmbeddingProviderHealthResult(
                    provider=settings.provider,
                    model=settings.model,
                    priority=100,
                    healthy=error is None,
                    error=error,
                    latency_ms=elapsed_ms,
                )
            ]

    raw_configs = list(provider_configs)
    sanitized_configs = build_provider_chain(raw_configs)
    results: List[EmbeddingProviderHealthResult] = []

    for raw_config, config in zip(raw_configs, sanitized_configs):
        start = time.monotonic()
        error = _probe_single_provider(config)
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        results.append(
            EmbeddingProviderHealthResult(
                provider=str(config.get("provider", raw_config.get("provider", ""))),
                model=str(config.get("model", raw_config.get("model", ""))),
                priority=int(raw_config.get("priority", 100)),
                healthy=error is None,
                error=error,
                latency_ms=elapsed_ms,
            )
        )

    results.sort(key=lambda result: result.priority)
    return results