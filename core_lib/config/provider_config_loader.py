"""Shared provider config loader for YAML/JSON files.

This module provides reusable loading/filtering logic for service-specific
provider lists (LLM, embeddings, reranker) with environment variable
substitution and priority-based selection.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in a value.

    Supports patterns:
        ${VAR_NAME}           - Required variable
        ${VAR_NAME:-default}  - Variable with default value
    """
    if isinstance(value, str):
        def replace_match(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2)
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            if default_value is not None:
                return default_value
            logger.warning("Environment variable %s not set and no default provided", var_name)
            return ""

        return _ENV_VAR_PATTERN.sub(replace_match, value)

    if isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [substitute_env_vars(item) for item in value]

    return value


def _to_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_config_file_path(config_file: str) -> Optional[str]:
    """Resolve config path from absolute/relative path and common locations."""
    if not config_file:
        return None

    if os.path.isabs(config_file) and os.path.exists(config_file):
        return config_file

    candidates = [
        os.path.expanduser(config_file),
        os.path.join(os.getcwd(), config_file),
        os.path.join(os.path.dirname(__file__), config_file),
        os.path.join("/etc/llm", config_file),
        os.path.expanduser(os.path.join("~/.config/llm", config_file)),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def load_provider_config_file(path: str, substitute_env: bool = True) -> Optional[Any]:
    """Load provider configuration data from JSON/YAML file."""
    if not path:
        return None

    file_path = resolve_config_file_path(path)
    if not file_path:
        logger.warning("Provider config file not found: %s", path)
        return None

    content = Path(file_path).read_text(encoding="utf-8")

    if file_path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed, cannot load YAML provider config")
            return None
        data = yaml.safe_load(content) or {}
    else:
        data = json.loads(content)

    if substitute_env:
        data = substitute_env_vars(data)

    return data


def _extract_configs_from_data(data: Any, service: str) -> List[Dict[str, Any]]:
    """Extract service-specific provider configs from loaded config data."""
    if not data:
        return []

    require_service_tag = False

    if isinstance(data, list):
        # For plain list configs, only accept explicitly service-tagged rows.
        candidates = data
        require_service_tag = True
    else:
        service_aliases = {
            "embedding": ["embedding_providers", "embeddings_providers", "embeddings", "embedding"],
            "reranker": ["reranker_providers", "rerankers", "reranker"],
        }

        candidates = []
        has_service_specific_section = False
        for key in service_aliases.get(service, []):
            section = data.get(key)
            if isinstance(section, list):
                candidates.extend(section)
                has_service_specific_section = True
            elif isinstance(section, dict):
                nested = section.get("providers")
                if isinstance(nested, list):
                    candidates.extend(nested)
                    has_service_specific_section = True

        # Optional generic section support only when no service-specific section
        # exists. Generic rows must be explicitly tagged with service/services.
        if not has_service_specific_section:
            generic = data.get("providers", [])
            if isinstance(generic, list):
                candidates.extend(generic)
            require_service_tag = True

    extracted: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue

        service_tag = item.get("service")
        services_tag = item.get("services")
        if service_tag is not None or services_tag is not None:
            accepted = {service, f"{service}s", "all", "*"}

            service_values: List[str] = []
            if isinstance(service_tag, str):
                service_values = [service_tag.lower()]
            elif isinstance(services_tag, list):
                service_values = [str(v).lower() for v in services_tag]
            elif isinstance(services_tag, str):
                service_values = [services_tag.lower()]

            if not any(v in accepted for v in service_values):
                continue
        elif require_service_tag:
            continue

        extracted.append(item.copy())

    return extracted


def _matches_level(config: Dict[str, Any], intelligence_level: Optional[int]) -> bool:
    if intelligence_level is None:
        return True

    min_level = config.get("min_intelligence_level", config.get("min_level", 0))
    max_level = config.get("max_intelligence_level", config.get("max_level", 10))

    min_level_int = _to_int(min_level, 0)
    max_level_int = _to_int(max_level, 10)

    return min_level_int <= intelligence_level <= max_level_int


def _matches_usage(config: Dict[str, Any], usage: Optional[str]) -> bool:
    if not usage:
        return True

    requested = usage.strip().lower()
    configured = (
        config.get("usage")
        or config.get("use_case")
        or config.get("usecase")
        or config.get("purpose")
        or config.get("task")
    )

    if configured is None:
        return True

    if isinstance(configured, str):
        return configured.strip().lower() in {requested, "*", "all"}

    if isinstance(configured, list):
        values = {str(v).strip().lower() for v in configured}
        return bool(values.intersection({requested, "*", "all"}))

    return True


def get_service_provider_configs(
    service: str,
    config_file_path: Optional[str],
    intelligence_level: Optional[int] = None,
    usage: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load and filter service provider configs from a YAML/JSON file.

    Returns entries sorted by ascending priority.
    """
    if not config_file_path:
        return []

    data = load_provider_config_file(config_file_path, substitute_env=True)
    configs = _extract_configs_from_data(data, service=service)

    filtered: List[Dict[str, Any]] = []
    for config in configs:
        if not _to_bool(config.get("enabled", True), default=True):
            continue
        if not _matches_level(config, intelligence_level):
            continue
        if not _matches_usage(config, usage):
            continue

        config["priority"] = _to_int(config.get("priority", 100), 100)
        filtered.append(config)

    filtered.sort(key=lambda c: c.get("priority", 100))
    return filtered
