"""Startup preflight checks for configured LLM providers.

This module provides a lightweight, non-blocking validation pass that services
can call at startup to detect obvious provider/model misconfiguration before the
first user request.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core_lib import get_module_logger

from .provider_registry import ProviderRegistry

logger = get_module_logger()


_provider_registry_cache: ProviderRegistry | None = None


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
    url: Optional[str] = None
    location: Optional[str] = None


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
        # Cap output tokens for health probes so thinking-capable models don't
        # burn excessive tokens.  We only override when the provider has no
        # explicit max_tokens configured (to respect intentional limits).
        probe_config = provider
        try:
            import dataclasses as _dc
            if not getattr(provider, "max_tokens", None):
                probe_config = _dc.replace(provider, max_tokens=32)
        except Exception:
            pass  # dataclasses.replace failed (e.g. not a dataclass); use original

        client = probe_config.to_client()
        response = client.chat(
            [{"role": "user", "content": "Reply with OK."}],
            thinking_enabled=False,
        )
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


def _resolve_provider_endpoint_and_region(provider) -> tuple[Optional[str], Optional[str]]:
    """Resolve provider URL/region using effective config (includes env defaults)."""
    url = getattr(provider, "host", None) or None
    region = getattr(provider, "location", None) or None

    # ProviderConfig.to_llm_config applies provider/env defaults that may not be
    # present on the raw ProviderConfig object.
    try:
        resolved = provider.to_llm_config()
    except Exception:
        resolved = None

    if resolved is not None:
        if not url:
            url = (
                getattr(resolved, "base_url", None)
                or getattr(resolved, "azure_endpoint", None)
                or getattr(resolved, "host", None)
                or None
            )
        if not region:
            region = getattr(resolved, "location", None) or None

    return url, region


def get_cached_llm_provider_registry(
    *,
    force_reload: bool = False,
    env_var: str = "LLM_PROVIDERS",
    file_env_var: str = "LLM_PROVIDERS_FILE",
) -> ProviderRegistry:
    """Load and cache the LLM provider registry.

    This is a lightweight alternative to startup preflight for services that
    want their provider configuration parsed and cached at startup without
    sending live token-consuming probe requests.
    """
    global _provider_registry_cache

    if force_reload or _provider_registry_cache is None:
        _provider_registry_cache = ProviderRegistry.from_env(
            env_var=env_var,
            file_env_var=file_env_var,
        )

    return _provider_registry_cache


def reset_cached_llm_provider_registry() -> None:
    """Clear the cached LLM provider registry."""
    global _provider_registry_cache
    _provider_registry_cache = None


def warm_llm_provider_registry(*, force_reload: bool = False) -> int:
    """Warm the cached LLM provider registry without running live probes.

    Returns the number of enabled/configured providers available after loading.
    """
    registry = get_cached_llm_provider_registry(force_reload=force_reload)
    provider_count = len(getattr(registry, "providers", []) or [])
    logger.info("LLM provider registry loaded at startup: providers=%d", provider_count)
    return provider_count


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
        resolved_url, resolved_region = _resolve_provider_endpoint_and_region(provider)

        results.append(
            ProviderHealthResult(
                provider=provider.provider,
                model=provider.model,
                tier=getattr(provider, "tier", "standard"),
                priority=getattr(provider, "priority", 100),
                healthy=error is None,
                error=error,
                latency_ms=elapsed_ms,
                url=resolved_url,
                location=resolved_region,
            )
        )

    results.sort(key=lambda r: r.priority)
    return results


# ---------------------------------------------------------------------------
# Lightweight connectivity checks (no tokens consumed)
# ---------------------------------------------------------------------------

@dataclass
class ConnectivityResult:
    """Result of a no-token connectivity probe for a single LLM provider/model."""
    provider: str
    model: str
    tier: str
    priority: int
    min_intelligence_level: int
    max_intelligence_level: int
    status: str          # "ok", "down", "not_configured", "unknown"
    details: Optional[str] = None
    error: Optional[str] = None


def _http_get(url: str, headers: dict, timeout: int = 8) -> tuple[int, bytes]:
    """Simple HTTP GET; returns (status_code, body_bytes). Raises URLError on network failure."""
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def _parse_http_error(exc: HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode()).get("error", {}).get("message", "")
    except Exception:
        body = exc.reason
    return f"HTTP {exc.code}: {body or exc.reason}"


def _check_ollama(host: str, model: str) -> tuple[str, Optional[str], Optional[str]]:
    """status, details, error — checks /api/tags for model availability."""
    url = f"{host.rstrip('/')}/api/tags"
    try:
        _, body = _http_get(url, {}, timeout=5)
        data = json.loads(body.decode())
        models = data.get("models", [])
        names = [m.get("name", "") for m in models]
        available = any(n == model or n.startswith(model + ":") or model in n for n in names)
        if available:
            return "ok", f"{len(models)} model(s) loaded", None
        preview = ", ".join(names[:4]) if names else "none"
        return "down", None, f"Model '{model}' not found. Available: {preview}"
    except HTTPError as exc:
        return "down", None, _parse_http_error(exc)
    except URLError as exc:
        return "down", None, f"Connection failed: {exc.reason}"
    except Exception as exc:
        return "down", None, str(exc)


def _check_gemini(model: str, api_key: str) -> tuple[str, Optional[str], Optional[str]]:
    """status, details, error — uses model metadata endpoint, no tokens consumed."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}?key={api_key}"
    try:
        _, body = _http_get(url, {}, timeout=8)
        display_name = json.loads(body.decode()).get("displayName", "")
        return "ok", display_name or None, None
    except HTTPError as exc:
        msg = _parse_http_error(exc)
        if exc.code == 404:
            return "down", None, f"Model not found: {msg}"
        if exc.code in (401, 403):
            return "down", None, f"Auth failed: {msg}"
        return "down", None, msg
    except URLError as exc:
        return "down", None, f"Connection failed: {exc.reason}"
    except Exception as exc:
        return "down", None, str(exc)


def _check_openai_compatible(model: str, api_key: str, base_url: str) -> tuple[str, Optional[str], Optional[str]]:
    """status, details, error — GET /models/{model} (single model lookup)."""
    url = f"{base_url.rstrip('/')}/models/{model}"
    try:
        _http_get(url, {"Authorization": f"Bearer {api_key}"}, timeout=8)
        return "ok", None, None
    except HTTPError as exc:
        msg = _parse_http_error(exc)
        if exc.code == 404:
            return "down", None, f"Model '{model}' not found: {msg}"
        if exc.code in (401, 403):
            return "down", None, f"Auth failed: {msg}"
        return "down", None, msg
    except URLError as exc:
        return "down", None, f"Connection failed: {exc.reason}"
    except Exception as exc:
        return "down", None, str(exc)


def _check_openrouter(model: str, api_key: str, base_url: str) -> tuple[str, Optional[str], Optional[str]]:
    """status, details, error.

    OpenRouter model IDs contain '/' (e.g. nvidia/nemotron-...) which breaks
    per-model URL path construction.  Uses GET /models (list) instead.
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        _, body = _http_get(url, {"Authorization": f"Bearer {api_key}"}, timeout=10)
        ids = {m.get("id", "") for m in json.loads(body.decode()).get("data", [])}
        if model in ids:
            return "ok", f"{len(ids)} model(s) available", None
        return "down", None, f"Model '{model}' not found on OpenRouter"
    except HTTPError as exc:
        msg = _parse_http_error(exc)
        if exc.code in (401, 403):
            return "down", None, f"Auth failed: {msg}"
        return "down", None, msg
    except URLError as exc:
        return "down", None, f"Connection failed: {exc.reason}"
    except Exception as exc:
        return "down", None, str(exc)


def _check_azure(model: str, api_key: str, endpoint: str, api_version: str) -> tuple[str, Optional[str], Optional[str]]:
    """status, details, error — deployment info endpoint."""
    url = f"{endpoint.rstrip('/')}/openai/deployments/{model}?api-version={api_version}"
    try:
        _http_get(url, {"api-key": api_key}, timeout=8)
        return "ok", None, None
    except HTTPError as exc:
        msg = _parse_http_error(exc)
        if exc.code == 404:
            return "down", None, f"Deployment '{model}' not found"
        if exc.code in (401, 403):
            return "down", None, f"Auth failed: {msg}"
        return "down", None, msg
    except URLError as exc:
        return "down", None, f"Connection failed: {exc.reason}"
    except Exception as exc:
        return "down", None, str(exc)


def _probe_connectivity(provider) -> tuple[str, Optional[str], Optional[str]]:
    """Dispatch a no-token connectivity check based on provider type.

    Returns (status, details, error) where status is one of:
    "ok", "down", "not_configured", "unknown".
    """
    if not provider.is_configured():
        return "not_configured", None, "Missing required credentials"

    p = provider.provider

    if p == "ollama":
        return _check_ollama(provider.host or "http://localhost:11434", provider.model)

    if p in ("gemini", "vertex"):
        api_key = (
            provider.api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_GENAI_API_KEY")
            or ""
        )
        if not api_key:
            return "not_configured", None, "No API key"
        return _check_gemini(provider.model, api_key)

    if p == "openrouter":
        api_key = provider.api_key or os.getenv("OPENROUTER_API_KEY") or ""
        if not api_key:
            return "not_configured", None, "No API key"
        return _check_openrouter(provider.model, api_key, provider.host)

    if p in ("openai", "openai-responses"):
        api_key = (
            provider.api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or ""
        )
        if not api_key:
            return "not_configured", None, "No API key"
        return _check_openai_compatible(provider.model, api_key, provider.host or "https://api.openai.com/v1")

    if p == "azure-openai":
        api_key = provider.api_key or ""
        endpoint = provider.azure_endpoint or ""
        if not api_key or not endpoint:
            return "not_configured", None, "Missing API key or endpoint"
        api_version = provider.azure_api_version or "2024-08-01-preview"
        return _check_azure(provider.model, api_key, endpoint, api_version)

    return "unknown", None, f"No connectivity check implemented for provider '{p}'"


def check_llm_connectivity(providers: Optional[Iterable] = None) -> List[ConnectivityResult]:
    """Probe every configured LLM provider without consuming tokens.

    Each check verifies credentials and endpoint reachability only (e.g. model
    metadata or model-list endpoints), not chat completions.  Probes run in
    parallel threads with a 15-second total timeout.

    Args:
        providers: Optional iterable of ``ProviderConfig`` objects.  When
            *None* (the default) the registry is loaded from the environment
            via ``ProviderRegistry.from_env()``.

    Returns:
        A list of :class:`ConnectivityResult` — one per configured provider
        entry, sorted by priority ascending.
    """
    import concurrent.futures

    if providers is None:
        try:
            registry = ProviderRegistry.from_env()
            providers = list(getattr(registry, "all_providers", []) or [])
        except Exception as exc:
            logger.warning("LLM connectivity check: failed to load provider registry: %s", exc)
            return []

    configured = [p for p in providers if getattr(p, "enabled", True)]
    if not configured:
        return []

    def _run(provider) -> ConnectivityResult:
        try:
            status, details, error = _probe_connectivity(provider)
        except Exception as exc:
            status, details, error = "down", None, str(exc)
        return ConnectivityResult(
            provider=provider.provider,
            model=provider.model,
            tier=getattr(provider, "tier", "standard"),
            priority=getattr(provider, "priority", 100),
            min_intelligence_level=getattr(provider, "min_intelligence_level", 0),
            max_intelligence_level=getattr(provider, "max_intelligence_level", 10),
            status=status,
            details=details,
            error=error,
        )

    results: List[ConnectivityResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(configured), 6)) as executor:
        futures = {executor.submit(_run, p): p for p in configured}
        try:
            for future in concurrent.futures.as_completed(futures, timeout=15):
                try:
                    results.append(future.result())
                except Exception as exc:
                    p = futures[future]
                    results.append(ConnectivityResult(
                        provider=p.provider, model=p.model,
                        tier=getattr(p, "tier", "standard"),
                        priority=getattr(p, "priority", 100),
                        min_intelligence_level=getattr(p, "min_intelligence_level", 0),
                        max_intelligence_level=getattr(p, "max_intelligence_level", 10),
                        status="down", error=str(exc),
                    ))
        except concurrent.futures.TimeoutError:
            for future, p in futures.items():
                if not future.done():
                    results.append(ConnectivityResult(
                        provider=p.provider, model=p.model,
                        tier=getattr(p, "tier", "standard"),
                        priority=getattr(p, "priority", 100),
                        min_intelligence_level=getattr(p, "min_intelligence_level", 0),
                        max_intelligence_level=getattr(p, "max_intelligence_level", 10),
                        status="unknown", error="Connectivity check timed out",
                    ))

    results.sort(key=lambda r: r.priority)
    return results
