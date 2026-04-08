"""Per-provider health probes for reranker providers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from core_lib import get_module_logger

from ..config.provider_chain_utils import build_provider_chain
from .factory import RerankerFactory
from .reranker_config import RerankerSettings

logger = get_module_logger()


@dataclass
class RerankerProviderHealthResult:
    provider: str
    model: str
    priority: int
    healthy: bool
    error: Optional[str]
    latency_ms: Optional[float]
    url: Optional[str] = None


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
        client = RerankerFactory.create(provider=provider, model=model, **kwargs)
        return None if client.health_check() else "health_check returned False"
    except Exception as exc:
        return str(exc)
    finally:
        if client is not None:
            _close_client(client)


def _probe_settings(settings: RerankerSettings) -> str | None:
    client = None
    try:
        client = RerankerFactory.from_config(settings)
        return None if client.health_check() else "health_check returned False"
    except Exception as exc:
        return str(exc)
    finally:
        if client is not None:
            _close_client(client)


def check_reranker_providers_health(
    provider_configs: Iterable[dict] | None = None,
) -> List[RerankerProviderHealthResult]:
    """Probe configured reranker providers and return per-provider health results."""
    if provider_configs is None:
        try:
            settings = RerankerSettings.from_env(load_dotenv=False)
        except Exception as exc:
            logger.warning("Reranker health check: failed to load settings: %s", exc)
            return []

        provider_configs = list(getattr(settings, "provider_configs", ()) or ())
        if not provider_configs:
            start = time.monotonic()
            error = _probe_settings(settings)
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            resolved_url = getattr(settings, "infinity_url", None) or getattr(settings, "base_url", None) or None
            return [
                RerankerProviderHealthResult(
                    provider=settings.provider,
                    model=settings.model,
                    priority=100,
                    healthy=error is None,
                    error=error,
                    latency_ms=elapsed_ms,
                    url=resolved_url,
                )
            ]

    raw_configs = list(provider_configs)
    sanitized_configs = build_provider_chain(raw_configs)
    results: List[RerankerProviderHealthResult] = []

    for raw_config, config in zip(raw_configs, sanitized_configs):
        start = time.monotonic()
        error = _probe_single_provider(config)
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        resolved_url = (
            config.get("base_url")
            or raw_config.get("base_url")
            or None
        )
        results.append(
            RerankerProviderHealthResult(
                provider=str(config.get("provider", raw_config.get("provider", ""))),
                model=str(config.get("model", raw_config.get("model", ""))),
                priority=int(raw_config.get("priority", 100)),
                healthy=error is None,
                error=error,
                latency_ms=elapsed_ms,
                url=resolved_url,
            )
        )

    results.sort(key=lambda result: result.priority)
    return results