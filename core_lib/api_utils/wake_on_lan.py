"""Wake-on-LAN utilities for recovering sleeping hosts before failover.

This module is intentionally independent from request logic so callers can keep
network wake behavior configurable and optional per target host.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()


@dataclass(frozen=True)
class WakeTarget:
    """Wake configuration for one host."""

    host: str
    mac_address: str
    port: int = 9
    broadcast_ip: str = "255.255.255.255"
    wait_seconds: float = 20.0
    retry_timeout_seconds: Optional[float] = None
    max_attempts: int = 1


@dataclass(frozen=True)
class WakeResult:
    """Outcome of a wake attempt."""

    attempted: bool
    succeeded: bool
    retry_timeout_seconds: Optional[float] = None


class WakeOnLanStrategy:
    """Optional wake strategy for a subset of configured Infinity hosts.

    Supported config shape:
    {
      "enabled": true,
      "initial_timeout_seconds": 2,
      "targets": [
        {
          "host": "powerspec",
          "mac_address": "FC:34:97:9E:C8:AF",
          "port": 7777,
          "wait_seconds": 20,
          "retry_timeout_seconds": 8,
          "max_attempts": 1
        }
      ]
    }

    A shorthand single-target form is also supported by providing host/mac at
    the top level without "targets".
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", bool(cfg)))
        configured_initial_timeout = self._to_optional_float(
            cfg.get("initial_timeout_seconds")
            or cfg.get("initial_timeout")
        )
        self.initial_timeout_seconds: Optional[float] = (
            configured_initial_timeout
            if configured_initial_timeout is not None
            else (2.0 if self.enabled else None)
        )
        self._targets = self._parse_targets(cfg)
        self._woken_hosts: set[str] = set()

    @staticmethod
    def _to_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
            return parsed if parsed > 0 else default
        except (TypeError, ValueError):
            return default

    @classmethod
    def _normalize_mac(cls, mac_address: str) -> Optional[str]:
        if not mac_address:
            return None
        compact = str(mac_address).replace(":", "").replace("-", "").strip().lower()
        if len(compact) != 12:
            return None
        if any(c not in "0123456789abcdef" for c in compact):
            return None
        return compact

    @classmethod
    def _build_target(cls, source: Dict[str, Any]) -> Optional[WakeTarget]:
        host = str(source.get("host") or source.get("match_host") or "").strip().lower()
        mac_raw = source.get("mac_address") or source.get("mac")
        mac_address = cls._normalize_mac(str(mac_raw or ""))

        if not mac_address:
            return None

        # Optional host: when omitted, this target applies to the API client's base_url host.
        # We represent it as wildcard and resolve at lookup time.
        if not host:
            host = "*"

        wait_seconds = cls._to_optional_float(source.get("wait_seconds"))
        if wait_seconds is None or wait_seconds < 0:
            wait_seconds = 20.0

        return WakeTarget(
            host=host,
            mac_address=mac_address,
            port=cls._to_positive_int(source.get("port"), default=9),
            broadcast_ip=str(source.get("broadcast_ip") or "255.255.255.255"),
            wait_seconds=wait_seconds,
            retry_timeout_seconds=cls._to_optional_float(
                source.get("retry_timeout_seconds")
                or source.get("post_wake_timeout_seconds")
                or source.get("retry_timeout")
            ),
            max_attempts=cls._to_positive_int(source.get("max_attempts"), default=1),
        )

    @classmethod
    def _parse_targets(cls, cfg: Dict[str, Any]) -> List[WakeTarget]:
        targets_raw = cfg.get("targets")

        targets: List[WakeTarget] = []
        if isinstance(targets_raw, Sequence) and not isinstance(targets_raw, (str, bytes)):
            for item in targets_raw:
                if isinstance(item, dict):
                    target = cls._build_target(item)
                    if target:
                        targets.append(target)
        else:
            target = cls._build_target(cfg)
            if target:
                targets.append(target)

        return targets

    @staticmethod
    def _extract_host(base_url: str) -> str:
        parsed = urlparse(base_url)
        return (parsed.hostname or "").strip().lower()

    def _find_target(self, base_url: str) -> Optional[WakeTarget]:
        host = self._extract_host(base_url)
        if not host:
            return None

        wildcard_target: Optional[WakeTarget] = None
        for target in self._targets:
            if target.host == host:
                return target
            if target.host == "*":
                wildcard_target = target
        return wildcard_target

    @staticmethod
    def _build_magic_packet(mac_hex: str) -> bytes:
        mac_bytes = bytes.fromhex(mac_hex)
        return b"\xff" * 6 + mac_bytes * 16

    def _send_magic_packet(self, target: WakeTarget) -> None:
        packet = self._build_magic_packet(target.mac_address)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(packet, (target.broadcast_ip, target.port))
        finally:
            sock.close()

    def maybe_get_initial_timeout(self, base_url: str, default_timeout: float) -> float:
        """Return a short initial timeout for configured sleeping hosts."""
        if not self.enabled:
            return default_timeout

        target = self._find_target(base_url)
        if not target:
            return default_timeout

        if base_url in self._woken_hosts:
            return default_timeout

        if self.initial_timeout_seconds and self.initial_timeout_seconds > 0:
            return self.initial_timeout_seconds

        return default_timeout

    def maybe_wake(self, base_url: str, error: Exception) -> WakeResult:
        """Wake host once on connection/timeout failures.

        Returns:
            WakeResult describing whether wake was attempted and successful.
        """
        if not self.enabled:
            return WakeResult(attempted=False, succeeded=False)

        target = self._find_target(base_url)
        if not target:
            return WakeResult(attempted=False, succeeded=False)

        target_host = self._extract_host(base_url) if target.host == "*" else target.host

        if base_url in self._woken_hosts:
            return WakeResult(attempted=False, succeeded=False)

        logger.warning(
            f"Infinity host appears unavailable ({base_url}): {error}. "
            f"Attempting Wake-on-LAN for host '{target_host}'"
        )

        for attempt in range(1, target.max_attempts + 1):
            try:
                self._send_magic_packet(target)
                logger.info(
                    f"Sent WoL magic packet to {target_host} on UDP {target.port} "
                    f"(attempt {attempt}/{target.max_attempts})"
                )
                break
            except Exception as exc:
                logger.error(
                    f"Failed to send WoL packet to {target_host} "
                    f"(attempt {attempt}/{target.max_attempts}): {exc}"
                )
        else:
            return WakeResult(attempted=True, succeeded=False)

        if target.wait_seconds > 0:
            logger.info(
                f"Waiting {target.wait_seconds:.1f}s for host '{target_host}' to wake"
            )
            time.sleep(target.wait_seconds)

        self._woken_hosts.add(base_url)
        return WakeResult(
            attempted=True,
            succeeded=True,
            retry_timeout_seconds=target.retry_timeout_seconds,
        )
