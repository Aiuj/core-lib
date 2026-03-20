"""Machine identity helpers for telemetry resource attributes.

This module resolves stable host-level identifiers for OpenTelemetry resource
attributes without relying on external network calls.
"""

from __future__ import annotations

import os
import socket
from typing import Optional, Tuple


def _clean(value: Optional[str]) -> Optional[str]:
    """Normalize string values by trimming whitespace and dropping empties."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def resolve_machine_identity(explicit_instance_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Resolve machine identity for telemetry resource attributes.

    Resolution order:
    1. Explicit instance id (function argument)
    2. ``OTLP_INSTANCE_ID`` environment variable
    3. Hostname (``HOSTNAME``/``COMPUTERNAME``/``socket.gethostname``)

    Returns:
        Tuple of (instance_id, host_name). Values can be None when unavailable.
    """
    host_name = _clean(os.getenv("HOSTNAME")) or _clean(os.getenv("COMPUTERNAME"))
    if host_name is None:
        try:
            host_name = _clean(socket.gethostname())
        except Exception:
            host_name = None

    instance_id = _clean(explicit_instance_id) or _clean(os.getenv("OTLP_INSTANCE_ID")) or host_name
    return instance_id, host_name
