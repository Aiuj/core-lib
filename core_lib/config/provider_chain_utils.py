"""Utilities for shared provider-chain handling across services.

This module centralizes provider metadata sanitization and chain preparation
logic reused by embeddings/reranker factories.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROVIDER_METADATA_KEYS = {
    "priority",
    "enabled",
    "min_level",
    "max_level",
    "min_intelligence_level",
    "max_intelligence_level",
    "tier",
    "usage",
    "service",
    "services",
}


def sanitize_provider_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove non-constructor metadata keys from a provider config dict."""
    return {k: v for k, v in config.items() if k not in PROVIDER_METADATA_KEYS}


def build_provider_chain(
    provider_configs: Iterable[Dict[str, Any]],
    model: Optional[str] = None,
    model_key: str = "model",
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return sanitized provider configs with optional common overrides applied.

    Args:
        provider_configs: Raw provider config dictionaries.
        model: Optional model override applied to every config.
        model_key: Key name used for model field.
        overrides: Optional key/value overrides applied to every config when
            value is not None.
    """
    chain = [sanitize_provider_config(dict(config)) for config in provider_configs]

    if model is not None:
        for config in chain:
            config[model_key] = model

    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            for config in chain:
                config[key] = value

    return chain


def load_runtime_settings_if_needed(
    provider: Optional[str],
    loader: Callable[[], Any],
) -> Optional[Any]:
    """Load runtime settings only when provider is auto-detected."""
    if provider is not None:
        return None
    return loader()


def create_client_from_runtime_chain(
    provider: Optional[str],
    runtime_settings: Optional[Any],
    model: Optional[str],
    create_single: Callable[[str, Optional[str], Dict[str, Any]], Any],
    create_fallback: Callable[[List[Dict[str, Any]]], Any],
    chain_overrides: Optional[Dict[str, Any]] = None,
    default_provider: Optional[str] = None,
    model_key: str = "model",
) -> Optional[Any]:
    """Create a client from runtime provider chain settings when available.

    Returns None when no chain-based routing should be applied.
    """
    if provider is not None or runtime_settings is None:
        return None

    provider_configs = getattr(runtime_settings, "provider_configs", None)
    if not provider_configs:
        return None

    chain = build_provider_chain(
        provider_configs,
        model=model,
        model_key=model_key,
        overrides=chain_overrides,
    )

    if len(chain) > 1:
        return create_fallback(chain)

    single_cfg = chain[0]
    provider_name = single_cfg.pop(
        "provider",
        default_provider or getattr(runtime_settings, "provider", None),
    )
    model_name = single_cfg.pop(model_key, model)
    return create_single(provider_name, model_name, single_cfg)


def resolve_provider_url_from_env(
    provider_name: str,
    provider_env_map: Dict[str, List[str]],
    default_env_vars: Optional[List[str]] = None,
    extra_candidates: Optional[List[Optional[str]]] = None,
) -> Optional[str]:
    """Resolve first non-empty URL candidate from env/runtime for a provider."""
    env_names: List[str] = []
    env_names.extend(provider_env_map.get(provider_name, []))
    if default_env_vars:
        env_names.extend(default_env_vars)

    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value

    if extra_candidates:
        for value in extra_candidates:
            if value:
                return value

    return None


def get_multi_url_value(
    provider_name: str,
    provider_env_map: Dict[str, List[str]],
    default_env_vars: Optional[List[str]] = None,
    extra_candidates: Optional[List[Optional[str]]] = None,
) -> Optional[str]:
    """Return resolved URL value only when it is a multi-URL CSV string."""
    value = resolve_provider_url_from_env(
        provider_name=provider_name,
        provider_env_map=provider_env_map,
        default_env_vars=default_env_vars,
        extra_candidates=extra_candidates,
    )
    if value and "," in value:
        return value
    return None


def build_kwargs_from_config(
    config: Any,
    field_specs: Sequence[Tuple[str, str, str]],
) -> Dict[str, Any]:
    """Build provider kwargs from config object using declarative field specs.

    Args:
        config: settings/config object.
        field_specs: sequence of (source_attr, target_key, mode)
            where mode is:
            - "truthy": set only when source value is truthy
            - "exists": set whenever attribute exists (even if False/None)

    Returns:
        Dictionary ready to pass to factory/provider constructors.
    """
    kwargs: Dict[str, Any] = {}

    for source_attr, target_key, mode in field_specs:
        if not hasattr(config, source_attr):
            continue

        value = getattr(config, source_attr)
        if mode == "truthy":
            if value:
                kwargs[target_key] = value
        elif mode == "exists":
            kwargs[target_key] = value
        else:
            raise ValueError(f"Unknown field spec mode: {mode}")

    return kwargs
