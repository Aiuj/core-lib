"""Startup preflight checks for configured LLM providers.

This module provides a lightweight, non-blocking validation pass that services
can call at startup to detect obvious provider/model misconfiguration before the
first user request.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from core_lib import get_module_logger

from .provider_registry import ProviderRegistry

logger = get_module_logger()


# Known model aliases that frequently fail in some provider/region combinations.
_MODEL_HINTS = {
    "gemini-flash-lite-latest": "gemini-2.5-flash-lite",
    "gemini-flash-latest": "gemini-2.5-flash",
    "gemma-3-4b-it": "gemini-2.5-flash-lite",
}


@dataclass
class StartupValidationSummary:
    configured_providers: int
    static_warnings: int
    live_checks_attempted: int
    live_checks_failed: int


@dataclass
class ProviderHealthResult:
    """Health probe result for a single LLM provider/model."""
    provider: str
    model: str
    tier: str
    priority: int
    healthy: bool
    error: Optional[str]
    latency_ms: Optional[float]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "LLM startup preflight: invalid integer for %s=%r, using default=%d",
            name,
            raw,
            default,
        )
        return default


def _iter_unique_models(providers: Iterable) -> list:
    unique = []
    seen = set()
    for provider in providers:
        key = (provider.provider, provider.model)
        if key in seen:
            continue
        seen.add(key)
        unique.append(provider)
    return unique


def _run_static_checks(providers: Iterable) -> int:
    warnings = 0
    for provider in providers:
        model = (provider.model or "").strip()
        replacement = _MODEL_HINTS.get(model)
        if replacement:
            logger.warning(
                "LLM startup preflight: model '%s' on provider '%s' is often unavailable for some region/project routes. "
                "Consider using '%s'.",
                model,
                provider.provider,
                replacement,
            )
            warnings += 1
        elif model.endswith("-latest"):
            logger.warning(
                "LLM startup preflight: model alias '%s' may not be available in all regions/projects. "
                "Prefer a pinned model version where possible.",
                model,
            )
            warnings += 1
    return warnings


def _probe_provider(provider) -> str | None:
    """Probe a single provider/model.

    Returns None on success, otherwise a short error string.
    """
    client = None
    try:
        client = provider.to_client()
        response = client.chat([{"role": "user", "content": "Reply with OK."}])
        if isinstance(response, dict) and response.get("error"):
            return str(response["error"])
        return None
    except Exception as exc:
        return str(exc)
    finally:
        if client is not None:
            close_method = getattr(client, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    # Ignore cleanup errors; only startup signal is relevant.
                    pass


async def run_llm_startup_preflight() -> StartupValidationSummary:
    """Run startup checks for configured providers.

    This function never raises: it logs warnings/info only so startup is not
    blocked by transient provider issues.

    Environment flags:
    - LLM_STARTUP_LIVE_PROBE: enable/disable live probes (default: true)
    - LLM_STARTUP_LIVE_PROBE_MAX: max distinct provider/model probes (default: 3)
    """
    try:
        registry = ProviderRegistry.from_env()
    except Exception as exc:
        logger.warning("LLM startup preflight: failed to load provider registry: %s", exc)
        return StartupValidationSummary(0, 0, 0, 0)

    providers = list(getattr(registry, "providers", []) or [])
    if not providers:
        logger.warning("LLM startup preflight: no providers configured in registry.")
        return StartupValidationSummary(0, 0, 0, 0)

    static_warnings = _run_static_checks(providers)

    live_enabled = _env_bool("LLM_STARTUP_LIVE_PROBE", True)
    max_live_probes = max(0, _env_int("LLM_STARTUP_LIVE_PROBE_MAX", 3))
    live_attempted = 0
    live_failed = 0

    if live_enabled:
        for provider in _iter_unique_models(providers):
            if live_attempted >= max_live_probes:
                break

            live_attempted += 1
            provider_label = f"{provider.provider}:{provider.model}"
            error = await asyncio.to_thread(_probe_provider, provider)
            if error:
                live_failed += 1
                if "404" in error and "NOT_FOUND" in error:
                    logger.warning(
                        "LLM startup preflight: provider %s failed live probe with 404 NOT_FOUND. "
                        "This often indicates an unavailable model/version for the active project/region.",
                        provider_label,
                    )
                else:
                    logger.warning(
                        "LLM startup preflight: provider %s failed live probe: %s",
                        provider_label,
                        error,
                    )
            else:
                logger.info("LLM startup preflight: provider %s passed live probe.", provider_label)

    logger.info(
        "LLM startup preflight summary: providers=%d, static_warnings=%d, live_checks=%d, live_failures=%d",
        len(providers),
        static_warnings,
        live_attempted,
        live_failed,
    )

    return StartupValidationSummary(
        configured_providers=len(providers),
        static_warnings=static_warnings,
        live_checks_attempted=live_attempted,
        live_checks_failed=live_failed,
    )


def check_llm_providers_health(providers: Iterable | None = None) -> List[ProviderHealthResult]:
    """Probe every configured LLM provider and return per-provider health results.

    This is a **synchronous** function suitable for REST health endpoints.  It
    runs probes sequentially; each probe sends a single minimal chat message to
    the provider and records latency + any error.

    Args:
        providers: Optional iterable of ``ProviderConfig`` objects.  When
            *None* (the default) the registry is loaded from the environment
            via ``ProviderRegistry.from_env()``.

    Returns:
        A list of :class:`ProviderHealthResult` — one per configured provider
        entry, sorted by priority ascending.
    """
    if providers is None:
        try:
            registry = ProviderRegistry.from_env()
            providers = list(getattr(registry, "providers", []) or [])
        except Exception as exc:
            logger.warning("LLM health check: failed to load provider registry: %s", exc)
            return []

    configured = list(providers)
    results: List[ProviderHealthResult] = []

    for provider in configured:
        start = time.monotonic()
        error = _probe_provider(provider)
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)

        results.append(
            ProviderHealthResult(
                provider=provider.provider,
                model=provider.model,
                tier=getattr(provider, "tier", "standard"),
                priority=getattr(provider, "priority", 100),
                healthy=error is None,
                error=error,
                latency_ms=elapsed_ms,
            )
        )

    results.sort(key=lambda r: r.priority)
    return results
