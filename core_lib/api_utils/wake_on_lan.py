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
    """Wake configuration for one host.

    ``broadcast_ip`` is the UDP destination for the magic packet.  It can be:
    - A plain unicast IP (e.g. a router's WAN IP) — works from containers and
      remote servers (``SO_BROADCAST`` is NOT set).
    - A directed-broadcast (e.g. ``192.168.1.255``) — LAN-only.
    - ``255.255.255.255`` — limited broadcast, LAN-only, blocked by routers
      and container bridge networks.
    """

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
    warmup_seconds: Optional[float] = None
    """When > 0, the caller should route to secondary servers for this many seconds
    instead of retrying the waking host directly.  Set when *non-blocking* WoL mode
    is active (``warmup_seconds`` configured on the strategy).
    """


class WakeOnLanStrategy:
    """Optional wake strategy for a subset of configured Infinity hosts.

    Supported config shape — **shorthand** (target fields flat, single server):
    {
      "mac_address": "FC:34:97:9E:C8:AF",   # enabled defaults to true when block is present
      "broadcast_ip": "82.66.214.52",
      "port": 7777,
      "wait_seconds": 20,
      "retry_timeout_seconds": 8
    }

    **targets list** form (identical result; required for multiple machines):
    {
      "enabled": true,
      "initial_timeout_seconds": 2,  # optional, default 2 s when enabled
      "targets": [
        {
          "mac_address": "FC:34:97:9E:C8:AF",
          "port": 7777,
          "wait_seconds": 20,
          "retry_timeout_seconds": 8,
          "max_attempts": 1           # optional, default 1
        }
      ]
    }

    ``initial_timeout_seconds`` is optional.  When omitted (or set to ``None``)
    and WoL is enabled, it defaults to **2 s** so that sleeping hosts
    are detected quickly without the caller having to repeat the value.
    Set it explicitly to override the default (e.g. ``5`` for slower networks).

    ``enabled`` defaults to ``True`` whenever a non-empty ``wake_on_lan`` block
    is provided, so you can omit ``enabled: true`` entirely.  Set
    ``enabled: false`` explicitly to disable WoL while keeping the config in
    place (e.g. for local development).
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
        # Non-blocking mode: how long (seconds) to route to secondary servers after
        # a WoL packet is sent, instead of blocking until the host wakes.
        # Set to 0 or omit to keep the original blocking wait behaviour.
        self.warmup_seconds: Optional[float] = self._to_optional_float(
            cfg.get("warmup_seconds")
        )
        # Timestamps (host URL → epoch seconds) recording when a WoL was triggered,
        # used to decide when the warmup window has elapsed.
        self._waking_timestamps: Dict[str, float] = {}

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
        mac_raw = source.get("mac_address") or source.get("mac")
        mac_address = cls._normalize_mac(str(mac_raw or ""))

        if not mac_address:
            return None

        wait_seconds = cls._to_optional_float(source.get("wait_seconds"))
        if wait_seconds is None or wait_seconds < 0:
            wait_seconds = 20.0

        raw_target_ip = source.get("target_ip") or source.get("ip")
        target_ip: Optional[str] = str(raw_target_ip).strip() if raw_target_ip else None

        return WakeTarget(
            mac_address=mac_address,
            port=cls._to_positive_int(source.get("port"), default=9),
            broadcast_ip=target_ip or str(source.get("broadcast_ip") or "255.255.255.255"),
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

    def _find_target(self) -> Optional[WakeTarget]:
        return self._targets[0] if self._targets else None

    @staticmethod
    def _build_magic_packet(mac_hex: str) -> bytes:
        mac_bytes = bytes.fromhex(mac_hex)
        return b"\xff" * 6 + mac_bytes * 16

    @staticmethod
    def _is_broadcast(ip: str) -> bool:
        """Return True when ip looks like a broadcast address (last octet is 255)."""
        return ip.endswith(".255") or ip == "255.255.255.255"

    def _send_magic_packet(self, target: WakeTarget) -> None:
        packet = self._build_magic_packet(target.mac_address)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if self._is_broadcast(target.broadcast_ip):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(packet, (target.broadcast_ip, target.port))
        finally:
            sock.close()

    def is_in_warmup(self, base_url: str) -> bool:
        """Return True while a WoL warmup window is active for *base_url*.

        After sending a WoL packet in non-blocking mode, the waking host is
        unavailable for ``warmup_seconds``.  During this window callers should
        skip the host and route requests to secondary providers instead.
        Once the window elapses the host is assumed to be operational and will
        be tried again.

        Returns False when warmup mode is disabled (``warmup_seconds`` not set)
        or when no packet has been sent for this URL yet.
        """
        if not self.warmup_seconds or self.warmup_seconds <= 0:
            return False
        wake_time = self._waking_timestamps.get(base_url)
        if wake_time is None:
            return False
        remaining = self.warmup_seconds - (time.time() - wake_time)
        if remaining > 0:
            logger.debug(
                f"WoL warmup active for {base_url}: {remaining:.1f}s remaining "
                f"(routing to secondary)"
            )
            return True
        return False

    def maybe_get_initial_timeout(self, base_url: str, default_timeout: float) -> float:
        """Return a short initial timeout for configured sleeping hosts."""
        if not self.enabled:
            return default_timeout

        target = self._find_target()
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

        target = self._find_target()
        if not target:
            return WakeResult(attempted=False, succeeded=False)

        target_host = self._extract_host(base_url)

        if base_url in self._woken_hosts:
            return WakeResult(attempted=False, succeeded=False)

        is_warmup_mode = bool(self.warmup_seconds and self.warmup_seconds > 0)
        # In warmup mode the host being asleep is expected — only warn on
        # classic blocking mode where the caller is about to stall.
        if not is_warmup_mode:
            logger.warning(
                f"Infinity host appears unavailable ({base_url}): {error}. "
                f"Attempting Wake-on-LAN for host '{target_host}'"
            )
        else:
            logger.debug(
                f"Infinity host appears unavailable ({base_url}): {error}. "
                f"Attempting Wake-on-LAN for host '{target_host}'"
            )

        for attempt in range(1, target.max_attempts + 1):
            try:
                self._send_magic_packet(target)
                if is_warmup_mode:
                    logger.debug(
                        f"Sent WoL magic packet to {target_host} on UDP {target.port} "
                        f"(attempt {attempt}/{target.max_attempts})"
                    )
                else:
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

        if self.warmup_seconds and self.warmup_seconds > 0:
            # Non-blocking mode: record wake timestamp and let the caller route
            # requests to secondary servers for the warmup window instead of
            # sleeping here and retrying the same host.
            self._waking_timestamps[base_url] = time.time()
            logger.info(
                f"WoL sent to '{target_host}' — warmup active for "
                f"{self.warmup_seconds:.0f}s, routing to secondary"
            )
        else:
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
            warmup_seconds=self.warmup_seconds if self.warmup_seconds else None,
        )
