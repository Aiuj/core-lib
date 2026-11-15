"""Shared provider diagnostics for FastAPI health endpoints.

This module centralises the LLM/embedding/reranker probe logic so that every
service (mcp-doc-qa, agent-rfx, …) can call a single function instead of
maintaining separate copies of the same boilerplate.

Usage::

    from core_lib.api_utils import check_providers_health

    components = check_providers_health(
        check_llm=True,
        wait_for_wol=True,
        llm_providers_file="/path/to/llm_providers.yaml",  # optional
    )
    # components is a dict with keys:
    #   "llm_providers", "embedding_providers", "reranker_providers"

The dict entries follow the same schema already used by both services, so
callers can merge the result directly into their own response dict.

Notes
-----
*Conditional probing for agent-rfx*: agent-rfx does not use embeddings or
rerankers, so calling the embedding/reranker probe would always fail.  Pass
``probe_embeddings=False`` / ``probe_rerankers=False`` to skip those sections.
By default (``None``) the function auto-detects whether those providers are
configured via the YAML file.  Passing an explicit ``bool`` overrides the
auto-detection.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()


def _build_llm_section(llm_results: List[Any]) -> Dict[str, Any]:
    return {
        "healthy": all(r.healthy for r in llm_results) if llm_results else False,
        "degraded": any(getattr(r, "status", "ok") == "degraded" for r in llm_results),
        "providers": [
            {
                "provider": r.provider,
                "model": r.model,
                "tier": r.tier,
                "priority": r.priority,
                "healthy": r.healthy,
                "error": r.error,
                "latency_ms": r.latency_ms,
                "status": getattr(r, "status", "ok" if r.healthy else "down"),
                "latency_threshold_ms": getattr(r, "latency_threshold_ms", None),
                **({"url": r.url} if r.url else {}),
                **({"region": r.location} if r.location else {}),
                **({"project": getattr(r, "project", None)} if getattr(r, "project", None) else {}),
                **({"credential_project": getattr(r, "credential_project", None)} if getattr(r, "credential_project", None) else {}),
                **({"service_account_email": getattr(r, "service_account_email", None)} if getattr(r, "service_account_email", None) else {}),
                **({"credential_type": getattr(r, "credential_type", None)} if getattr(r, "credential_type", None) else {}),
            }
            for r in llm_results
        ],
    }


def _build_embedding_section(embedding_results: List[Any]) -> Dict[str, Any]:
    return {
        "healthy": all(r.healthy for r in embedding_results) if embedding_results else False,
        "providers": [
            {
                "provider": r.provider,
                "model": r.model,
                "priority": r.priority,
                "healthy": r.healthy,
                "error": r.error,
                "latency_ms": r.latency_ms,
                **({"url": r.url} if r.url else {}),
            }
            for r in embedding_results
        ],
    }


def _build_reranker_section(reranker_results: List[Any]) -> Dict[str, Any]:
    return {
        "healthy": all(r.healthy for r in reranker_results) if reranker_results else False,
        "providers": [
            {
                "provider": r.provider,
                "model": r.model,
                "priority": r.priority,
                "healthy": r.healthy,
                "error": r.error,
                "latency_ms": r.latency_ms,
                **({"url": r.url} if r.url else {}),
            }
            for r in reranker_results
        ],
    }


def _detect_configured_providers(providers_file: Optional[str]) -> tuple[bool, bool]:
    """Return (embedding_configured, reranker_configured) by inspecting the YAML file."""
    if not providers_file or not os.path.isfile(providers_file):
        return False, False
    try:
        import yaml

        with open(providers_file) as fh:
            data = yaml.safe_load(fh) or {}
        embedding_configured = bool(
            data.get("embedding_providers")
            or data.get("embeddings_providers")
            or data.get("embeddings")
        )
        reranker_configured = bool(
            data.get("reranker_providers")
            or data.get("rerankers")
            or data.get("reranker")
        )
        return embedding_configured, reranker_configured
    except Exception as exc:
        logger.warning("providers_health: could not inspect YAML file %s: %s", providers_file, exc)
        return False, False


def check_providers_health(
    *,
    check_llm: bool = True,
    wait_for_wol: bool = False,
    enable_wol: Optional[bool] = None,
    probe_embeddings: Optional[bool] = None,
    probe_rerankers: Optional[bool] = None,
    llm_providers_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Probe LLM, embedding, and reranker providers and return a components dict.

    Parameters
    ----------
    check_llm:
        When *False* the function returns an empty dict immediately (no-op).
    wait_for_wol:
        Passed through to :func:`check_llm_providers_health`.  When *True*
        the probe waits for Wake-on-LAN warmup before declaring a sleeping
        host as degraded.
    enable_wol:
        Whether provider probes may trigger Wake-on-LAN or a configured
        wakeup service.  If omitted, it follows ``wait_for_wol``.  Set it to
        ``True`` while keeping ``wait_for_wol=False`` for non-blocking wakeup.
    probe_embeddings:
        Whether to probe embedding providers.  *None* (default) means
        auto-detect from ``llm_providers_file``; *True*/*False* overrides.
    probe_rerankers:
        Whether to probe reranker providers.  Same semantics as
        ``probe_embeddings``.
    llm_providers_file:
        Path to the ``llm_providers.yaml`` file used to auto-detect which
        optional provider types are configured.  Falls back to the
        ``LLM_PROVIDERS_FILE`` environment variable when not supplied.

    Returns
    -------
    dict
        Keys: ``"llm_providers"``, ``"embedding_providers"``,
        ``"reranker_providers"``.  Each value is a dict with at least a
        ``"healthy"`` key.  If a provider type was skipped, the value is
        ``{"configured": False}``.  On unexpected errors the value is
        ``{"healthy": False, "error": "<message>"}``.

        The dict is empty (``{}``) when *check_llm* is *False*.
    """
    if not check_llm:
        return {}

    if enable_wol is None:
        enable_wol = wait_for_wol

    # Resolve optional providers file for auto-detection
    resolved_file = llm_providers_file or os.environ.get("LLM_PROVIDERS_FILE")

    # Auto-detect if not explicitly overridden
    if probe_embeddings is None or probe_rerankers is None:
        auto_embed, auto_rerank = _detect_configured_providers(resolved_file)
        if probe_embeddings is None:
            probe_embeddings = auto_embed
        if probe_rerankers is None:
            probe_rerankers = auto_rerank

    components: Dict[str, Any] = {}

    try:
        from core_lib.llm import check_llm_providers_health

        llm_results = check_llm_providers_health(
            wait_for_wol=wait_for_wol,
            enable_wol=enable_wol,
        )
        components["llm_providers"] = _build_llm_section(llm_results)

        if probe_embeddings:
            try:
                from core_lib.embeddings import check_embedding_providers_health

                embedding_results = check_embedding_providers_health(allow_wol=enable_wol)
                components["embedding_providers"] = _build_embedding_section(embedding_results)
            except Exception as exc:
                components["embedding_providers"] = {"healthy": False, "error": str(exc)}
        else:
            components["embedding_providers"] = {"configured": False}

        if probe_rerankers:
            try:
                from core_lib.reranker import check_reranker_providers_health

                reranker_results = check_reranker_providers_health(allow_wol=enable_wol)
                components["reranker_providers"] = _build_reranker_section(reranker_results)
            except Exception as exc:
                components["reranker_providers"] = {"healthy": False, "error": str(exc)}
        else:
            components["reranker_providers"] = {"configured": False}

    except Exception as exc:
        components = {
            "llm_providers": {"healthy": False, "error": str(exc)},
            "embedding_providers": {"configured": False},
            "reranker_providers": {"configured": False},
        }

    return components


def is_providers_healthy(components: Dict[str, Any]) -> bool:
    """Return *True* when all probed provider sections report healthy.

    Skipped sections (``{"configured": False}``) are treated as healthy
    so that services without embeddings/rerankers are not falsely degraded.
    """
    for section in ("llm_providers", "embedding_providers", "reranker_providers"):
        data = components.get(section, {})
        if "configured" in data and not data.get("healthy", True):
            # Explicitly not-configured sections are not counted as failures
            continue
        if not data.get("healthy", True):
            return False
    return True


def check_downstream_service_health(
    url: Optional[str],
    *,
    auth_mode: Optional[str] = None,
    jwt_token: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_header_name: str = "x-api-key",
    health_endpoint: str = "/health",
    auth_endpoint: Optional[str] = None,
    auth_error: Optional[str] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Probe a downstream HTTP service's health and optional authentication.

    Parameters
    ----------
    url:
        Base URL of the downstream service. Returns ``{"configured": False}``
        when empty or *None*.
    auth_mode:
        Configured authentication mode (e.g. ``"jwt"``, ``"legacy"``, ``"none"``).
    jwt_token:
        Bearer JWT token to send when probing ``auth_endpoint``.
    api_key:
        Legacy API key to send when ``jwt_token`` is not set.
    api_key_header_name:
        Header name for legacy API key (default: ``"x-api-key"``).
    health_endpoint:
        Unauthenticated liveness path on the downstream service.
    auth_endpoint:
        Authenticated endpoint path to verify credentials (e.g.
        ``"/v1/documents?company_id=00000000-0000-0000-0000-000000000000"``).
        Skipped when *None* or when ``auth_mode == "none"``.
    auth_error:
        Pre-existing credential-setup error (e.g. missing ``JWT_SECRET``).
    timeout:
        HTTP request timeout in seconds.
    """
    if not url:
        return {"configured": False}

    import httpx

    base = url.rstrip("/")
    result: Dict[str, Any] = {"configured": True, "url": url}
    if auth_mode:
        result["auth_mode"] = auth_mode

    start = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            health_url = f"{base}/{health_endpoint.lstrip('/')}"
            resp = client.get(health_url)
            result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
            if resp.status_code != 200:
                result["healthy"] = False
                result["error"] = f"Health endpoint returned HTTP {resp.status_code}"
                return result

            if auth_error:
                result["healthy"] = False
                result["auth_verified"] = False
                result["error"] = auth_error
                return result

            if auth_endpoint and (auth_mode or "").lower() != "none":
                headers: Dict[str, str] = {}
                if jwt_token:
                    headers["Authorization"] = f"Bearer {jwt_token}"
                elif api_key:
                    headers[api_key_header_name] = api_key

                auth_url = f"{base}/{auth_endpoint.lstrip('/')}"
                auth_resp = client.get(auth_url, headers=headers)
                if auth_resp.status_code in (401, 403):
                    result["healthy"] = False
                    result["auth_verified"] = False
                    result["error"] = f"Authentication failed (HTTP {auth_resp.status_code})"
                    return result
                if auth_resp.status_code >= 500:
                    result["healthy"] = False
                    result["auth_verified"] = None
                    result["error"] = f"Auth probe returned HTTP {auth_resp.status_code}"
                    return result
                result["auth_verified"] = True

            result["healthy"] = True
            return result
    except Exception as exc:
        result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        result["healthy"] = False
        if auth_endpoint and (auth_mode or "").lower() != "none":
            result["auth_verified"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
