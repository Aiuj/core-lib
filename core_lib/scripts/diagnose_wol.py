#!/usr/bin/env python3
"""Diagnose and test Wake-on-LAN behaviour for Infinity embedding hosts.

Loads settings from a .env file (in the current directory or specified via
--env-file) exactly the way a calling application would, then runs one of
several diagnostic modes.

Modes (pick one or more, executed in order):

  --show-config       Print the resolved WoL config read from the environment.
  --health            Probe /health on every configured Infinity URL and report
                      status, applying the WoL initial_timeout when relevant.
  --dry-run           Walk through the full WoL decision logic (no packets sent
                      and no real HTTP requests).
  --send-wol          Immediately send a magic packet to a target host.
  --probe             Fire a real /embeddings request so the full retry/WoL path
                      can be exercised live.

Usage examples:
  diagnose-wol --show-config
  diagnose-wol --health
  diagnose-wol --dry-run
  diagnose-wol --send-wol --host powerspec
  diagnose-wol --probe
  diagnose-wol --probe --url http://powerspec:7997
  diagnose-wol --show-config --env-file /path/to/.env
  diagnose-wol --probe --url http://h1:7997,http://h2:7997 \\
      --mac FC:34:97:9E:C8:AF --host h1 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"


def _color(text: str, code: str, use_color: bool = True) -> str:
    return f"{code}{text}{_RESET}" if use_color else text


def _ok(msg: str, color: bool = True) -> str:
    return _color(f"✓ {msg}", _GREEN, color)


def _warn(msg: str, color: bool = True) -> str:
    return _color(f"⚠ {msg}", _YELLOW, color)


def _err(msg: str, color: bool = True) -> str:
    return _color(f"✗ {msg}", _RED, color)


def _info(msg: str, color: bool = True) -> str:
    return _color(f"• {msg}", _CYAN, color)


def _head(msg: str, color: bool = True) -> str:
    separator = "─" * (len(msg) + 4)
    return f"\n{_color(separator, _BOLD, color)}\n  {_color(msg, _BOLD, color)}\n{_color(separator, _BOLD, color)}"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Wake-on-LAN configuration and behaviour for Infinity embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              diagnose-wol --show-config
              diagnose-wol --health
              diagnose-wol --dry-run
              diagnose-wol --send-wol --host powerspec
              diagnose-wol --probe
        """),
    )

    # Mode flags
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved Wake-on-LAN config from the environment.",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Probe /health on every configured Infinity URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate WoL decisions without sending packets or HTTP requests.",
    )
    parser.add_argument(
        "--send-wol",
        action="store_true",
        help="Send a magic packet to the target host immediately.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Fire a real /embeddings request and exercise the full retry/WoL path.",
    )

    # Target overrides
    parser.add_argument(
        "--url",
        default=None,
        help="Override Infinity URL(s), comma-separated. Falls back to INFINITY_BASE_URL / EMBEDDING_BASE_URL.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Target host name for --send-wol (matches against WoL targets config). "
             "Defaults to the first configured WoL target host.",
    )
    parser.add_argument(
        "--mac",
        default=None,
        help="Override or supply a MAC address (format: FC:34:97:9E:C8:AF).",
    )
    parser.add_argument(
        "--ip",
        default=None,
        help="Destination IP for the magic packet. Can be a WAN/unicast IP (e.g. your router's "
             "public IP) or a directed-broadcast (e.g. 192.168.1.255). Use a unicast IP when "
             "sending from inside a container or a remote VPS. Overrides broadcast_ip in config "
             "when --mac is also supplied.",
    )
    parser.add_argument(
        "--wol-port",
        type=int,
        default=None,
        help="UDP port for magic packet (default: 9, or value from config).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for --probe (defaults to EMBEDDING_MODEL or BAAI/bge-small-en-v1.5).",
    )

    # Environment
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env file to load. Defaults to the nearest .env up from cwd.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug-level log output.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def _load_env(env_file: Optional[str]) -> str:
    """Load dotenv and return the resolved path for reporting."""
    from dotenv import load_dotenv, find_dotenv  # type: ignore[import]

    if env_file:
        path = Path(env_file).resolve()
        if not path.exists():
            print(f"  ERROR: --env-file not found: {path}", file=sys.stderr)
            sys.exit(1)
        load_dotenv(dotenv_path=str(path), override=False)
        return str(path)

    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(dotenv_path=found, override=False)
        return found
    return "(no .env found – using OS env)"


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _resolve_wol_config(url_override: Optional[str]) -> Dict[str, Any]:
    """Read settings and return a normalised config dict.

    Returns:
      {
        "urls":          [str, ...],
        "timeout":       int,
        "token":         str | None,
        "wake_on_lan":   dict | None,
        "source":        str,
      }
    """
    # Try to resolve via EmbeddingsSettings if available
    infinity_wake_on_lan: Optional[Dict[str, Any]] = None
    resolved_url: Optional[str] = None
    resolved_token: Optional[str] = None
    resolved_timeout: int = 30
    source = "environment variables"

    try:
        from core_lib.config.embeddings_settings import EmbeddingsSettings

        settings = EmbeddingsSettings.from_env(load_dotenv=False)
        infinity_wake_on_lan = settings.infinity_wake_on_lan
        resolved_url = settings.infinity_url or settings.base_url
        resolved_token = settings.infinity_token
        resolved_timeout = settings.infinity_timeout or settings.timeout or 30

        # If provider_configs is populated, check per-entry WoL too
        if not infinity_wake_on_lan and settings.provider_configs:
            for cfg in settings.provider_configs:
                if cfg.get("wake_on_lan"):
                    infinity_wake_on_lan = cfg["wake_on_lan"]
                    source = "YAML provider config"
                    break

    except Exception as exc:
        print(f"  {_warn('Could not load EmbeddingsSettings')}: {exc}")

    # Raw env-var fallback: INFINITY_WAKE_ON_LAN as JSON (not read by EmbeddingsSettings itself)
    if not infinity_wake_on_lan:
        raw_wol = os.getenv("INFINITY_WAKE_ON_LAN")
        if raw_wol:
            try:
                infinity_wake_on_lan = json.loads(raw_wol)
                source = "INFINITY_WAKE_ON_LAN env var (JSON)"
            except json.JSONDecodeError as exc:
                print(f"  {_warn(f'INFINITY_WAKE_ON_LAN is set but not valid JSON: {exc}')}")

    # --url override wins
    if url_override:
        resolved_url = url_override

    # Last resort: raw env vars
    if not resolved_url:
        resolved_url = (
            os.getenv("INFINITY_BASE_URL")
            or os.getenv("EMBEDDING_BASE_URL")
            or "http://localhost:7997"
        )

    if not resolved_token:
        resolved_token = os.getenv("INFINITY_TOKEN") or os.getenv("EMBEDDING_TOKEN") or None

    # Parse URLs
    urls = [u.strip().rstrip("/") for u in resolved_url.split(",") if u.strip()]

    return {
        "urls": urls,
        "timeout": resolved_timeout,
        "token": resolved_token,
        "wake_on_lan": infinity_wake_on_lan,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Mode: --show-config
# ---------------------------------------------------------------------------

def cmd_show_config(cfg: Dict[str, Any], color: bool) -> None:
    print(_head("Resolved Wake-on-LAN Configuration", color))

    print(f"\n  Source         : {cfg['source']}")
    print(f"  Infinity URL(s): {', '.join(cfg['urls'])}")
    print(f"  Timeout        : {cfg['timeout']}s")
    print(f"  Token          : {'<set>' if cfg['token'] else '(none)'}")

    wol = cfg.get("wake_on_lan")
    if not wol:
        print(f"\n  {_warn('No wake_on_lan config found.', color)}")
        print("  Set INFINITY_WAKE_ON_LAN as JSON in your .env or provide a YAML provider config.")
        return

    print(f"\n  Wake-on-LAN config:\n")
    print(textwrap.indent(json.dumps(wol, indent=2), "    "))

    # Parse it with the strategy and report derived values
    try:
        from core_lib.api_utils.wake_on_lan import WakeOnLanStrategy

        strategy = WakeOnLanStrategy(wol)
        print(f"\n  Parsed targets  : {len(strategy._targets)}")
        for i, t in enumerate(strategy._targets):
            kind = "broadcast" if t.broadcast_ip.endswith(".255") or t.broadcast_ip == "255.255.255.255" else "unicast"
            print(f"    [{i}] dest={t.broadcast_ip}  mac={t.mac_address}  port={t.port}  "
                  f"({kind})  wait={t.wait_seconds}s  "
                  f"retry_timeout={t.retry_timeout_seconds}s  max_attempts={t.max_attempts}")
        print(f"  initial_timeout : {strategy.initial_timeout_seconds}s")
        print(f"  enabled         : {strategy.enabled}")

    except Exception as exc:
        print(f"  {_warn('Could not parse WoL strategy', color)}: {exc}")


# ---------------------------------------------------------------------------
# Mode: --health
# ---------------------------------------------------------------------------

def cmd_health(cfg: Dict[str, Any], color: bool) -> None:
    print(_head("Infinity Host Health Check", color))

    try:
        import requests as req
    except ImportError:
        print(_err("'requests' is not installed. Run: uv pip install requests", color))
        return

    wol = cfg.get("wake_on_lan")
    strategy = None
    if wol:
        try:
            from core_lib.api_utils.wake_on_lan import WakeOnLanStrategy
            strategy = WakeOnLanStrategy(wol)
        except Exception:
            pass

    for url in cfg["urls"]:
        effective_timeout = (
            strategy.maybe_get_initial_timeout(url, cfg["timeout"])
            if strategy
            else cfg["timeout"]
        )

        headers = {}
        if cfg["token"]:
            headers["Authorization"] = f"Bearer {cfg['token']}"

        print(f"\n  URL   : {url}")
        print(f"  Using timeout {effective_timeout}s"
              + (" (WoL initial timeout)" if effective_timeout != cfg["timeout"] else ""))

        start = time.time()
        try:
            resp = req.get(f"{url}/health", headers=headers, timeout=effective_timeout)
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                print(f"  {_ok(f'Healthy  ({elapsed:.0f}ms)', color)}")
            else:
                print(f"  {_warn(f'HTTP {resp.status_code}  ({elapsed:.0f}ms)', color)}")
        except req.exceptions.Timeout:
            elapsed = (time.time() - start) * 1000
            print(f"  {_err(f'Timeout after {effective_timeout}s', color)}")
            if strategy and strategy._find_target():
                print(f"  {_info('Host is in WoL targets – would send magic packet before retry.', color)}")
        except req.exceptions.ConnectionError as exc:
            print(f"  {_err(f'Connection refused/unreachable: {exc}', color)}")
            if strategy and strategy._find_target():
                print(f"  {_info('Host is in WoL targets – would send magic packet before retry.', color)}")
        except Exception as exc:
            print(f"  {_err(str(exc), color)}")

    print()


# ---------------------------------------------------------------------------
# Mode: --dry-run
# ---------------------------------------------------------------------------

def cmd_dry_run(cfg: Dict[str, Any], color: bool) -> None:
    print(_head("Wake-on-LAN Dry Run (no packets sent)", color))

    wol = cfg.get("wake_on_lan")
    if not wol:
        print(f"\n  {_warn('No wake_on_lan config present – nothing to simulate.', color)}")
        return

    try:
        from core_lib.api_utils.wake_on_lan import WakeOnLanStrategy
    except ImportError as exc:
        print(_err(f"Import failed: {exc}", color))
        return

    strategy = WakeOnLanStrategy(wol)

    print(f"\n  WoL enabled          : {strategy.enabled}")
    print(f"  Initial timeout      : {strategy.initial_timeout_seconds}s  "
          f"(applies to sleeping hosts before sending packet)")
    warmup = strategy.warmup_seconds
    if warmup:
        print(f"  Warmup window        : {warmup:.0f}s  (non-blocking — routes to secondary after WoL)")
    print(f"  Number of targets    : {len(strategy._targets)}")

    for url in cfg["urls"]:
        target = strategy._find_target()
        initial_t = strategy.maybe_get_initial_timeout(url, cfg["timeout"])

        print(f"\n  URL: {url}")
        if target:
            print(f"    {_ok('Matched WoL target', color)}")
            kind = "broadcast" if target.broadcast_ip.endswith(".255") or target.broadcast_ip == "255.255.255.255" else "unicast"
            print(f"    destination      : {target.broadcast_ip} ({kind})  port {target.port}")
            print(f"    mac_address      : {target.mac_address}")
            print(f"    initial_timeout  : {initial_t}s  (vs normal {cfg['timeout']}s)")
            if warmup:
                print(f"    warmup_seconds   : {warmup:.0f}s")
            else:
                print(f"    wait_after_wake  : {target.wait_seconds}s")
            print(f"    retry_timeout    : {target.retry_timeout_seconds or cfg['timeout']}s")
            print(f"    max_attempts     : {target.max_attempts}")
            print(f"\n    Simulated request flow:")
            print(f"      1. POST /embeddings  timeout={initial_t}s")
            print(f"      2. On Timeout/ConnError → send magic packet to {target.mac_address} UDP {target.port}")
            if warmup:
                print(f"      3. Route to secondary for {warmup:.0f}s warmup window")
                print(f"      4. After {warmup:.0f}s → retry this URL normally")
            else:
                if target.wait_seconds:
                    print(f"      3. Wait {target.wait_seconds}s for host to boot")
                print(f"      {'4' if target.wait_seconds else '3'}. Retry POST /embeddings  timeout={target.retry_timeout_seconds or cfg['timeout']}s")
                print(f"      {'5' if target.wait_seconds else '4'}. On failure → fail over to next URL (if any)")
        else:
            print(f"    {_warn('No WoL target for this host – normal failover only.', color)}")
            print(f"    initial_timeout  : {initial_t}s  (== normal timeout)")

    print()


# ---------------------------------------------------------------------------
# Mode: --send-wol
# ---------------------------------------------------------------------------

def cmd_send_wol(
    cfg: Dict[str, Any],
    host_override: Optional[str],
    mac_override: Optional[str],
    ip_override: Optional[str],
    port_override: Optional[int],
    color: bool,
) -> None:
    print(_head("Send Magic Packet (WoL)", color))

    try:
        from core_lib.api_utils.wake_on_lan import WakeOnLanStrategy, WakeTarget
    except ImportError as exc:
        print(_err(f"Import failed: {exc}", color))
        return

    wol = cfg.get("wake_on_lan") or {}

    # Allow fully ad-hoc: override mac at top level
    if mac_override:
        wol = dict(wol)
        wol["mac_address"] = mac_override
        if host_override:
            wol.pop("targets", None)
            wol["host"] = host_override
        if ip_override:
            wol["broadcast_ip"] = ip_override
        if port_override:
            wol["port"] = port_override
        wol["enabled"] = True

    if not wol:
        print(_err(
            "No WoL config found. Provide --mac (and optionally --host / --wol-port) "
            "or add wake_on_lan to your .env / YAML config.",
            color,
        ))
        return

    strategy = WakeOnLanStrategy(wol)

    if not strategy._targets:
        print(_err("No valid WoL targets could be parsed from config.", color))
        print("  Ensure mac_address is set (format: FC:34:97:9E:C8:AF).")
        return

    # Pick target
    target = strategy._targets[0] if strategy._targets else None

    if target is None:
        print(_err("No WoL target configured.", color))
        return

    print(f"\n  MAC address    : {target.mac_address}")
    print(f"  UDP port       : {port_override or target.port}")
    print(f"  Broadcast IP   : {target.broadcast_ip}")

    # Patch port if override
    if port_override and port_override != target.port:
        from dataclasses import replace as _replace
        target = _replace(target, port=port_override)

    print(f"\n  Sending magic packet …")
    try:
        strategy._send_magic_packet(target)
        print(f"  {_ok('Magic packet sent.', color)}")
        if target.wait_seconds:
            print(f"  Waiting {target.wait_seconds}s for host to boot …")
            time.sleep(target.wait_seconds)
            print(f"  {_ok('Wait complete.', color)}")
    except Exception as exc:
        print(f"  {_err(f'Failed to send packet: {exc}', color)}")
        _suggest_wol_errors(exc, color)

    print()


def _suggest_wol_errors(exc: Exception, color: bool) -> None:
    msg = str(exc).lower()
    if "permission" in msg or "operation not permitted" in msg:
        print(f"  {_info('Hint: Sending WoL packets requires network broadcast access.', color)}")
        print(f"  {_info('On Linux try: sudo diagnose-wol --send-wol', color)}")
        print(f"  {_info('Or configure a directed-broadcast address via --wol-port.', color)}")
    elif "network unreachable" in msg:
        print(f"  {_info('Hint: The broadcast network is unreachable from this host.', color)}")


# ---------------------------------------------------------------------------
# Mode: --probe
# ---------------------------------------------------------------------------

def cmd_probe(
    cfg: Dict[str, Any],
    model_override: Optional[str],
    color: bool,
) -> None:
    print(_head("Live Probe (real /embeddings request)", color))

    try:
        from core_lib.api_utils import InfinityAPIClient
        from core_lib.api_utils.infinity_api import InfinityAPIError
    except ImportError as exc:
        print(_err(f"Import failed: {exc}", color))
        return

    model = model_override or os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
    wol = cfg.get("wake_on_lan")

    client = InfinityAPIClient(
        base_urls=cfg["urls"],
        timeout=cfg["timeout"],
        token=cfg["token"] or None,
        max_retries_per_url=1,
        wake_on_lan=wol,
    )

    print(f"\n  URL(s)  : {', '.join(cfg['urls'])}")
    print(f"  Model   : {model}")
    print(f"  Timeout : {cfg['timeout']}s")
    print(f"  WoL     : {'configured' if wol else 'not configured'}")

    # Show per-URL initial timeout
    if wol:
        from core_lib.api_utils.wake_on_lan import WakeOnLanStrategy
        strategy = WakeOnLanStrategy(wol)
        for url in cfg["urls"]:
            it = strategy.maybe_get_initial_timeout(url, cfg["timeout"])
            suffix = " (WoL initial timeout)" if it != cfg["timeout"] else ""
            print(f"    {url}  →  initial_timeout={it}s{suffix}")

    print(f"\n  Sending: POST /embeddings  input=[\"diagnose wol probe\"]")
    print()

    start = time.time()
    try:
        data, used_url = client.post(
            "/embeddings",
            json={"input": ["diagnose wol probe"], "model": model},
        )
        elapsed = (time.time() - start) * 1000
        vectors = data.get("data", [])
        dim = len(vectors[0]["embedding"]) if vectors else "?"
        print(f"  {_ok(f'Success via {used_url}  ({elapsed:.0f}ms)', color)}")
        print(f"  Returned {len(vectors)} embedding(s), dimension={dim}")
        url_status = client.get_url_status()
        if len(url_status) > 1:
            print(f"\n  {_info('URL failure counts after request:', color)}")
            for s in url_status:
                mark = " ← used" if s["is_current"] else ""
                print(f"    [{s['index']}] {s['url']}  failures={s['failures']}{mark}")
    except InfinityAPIError as exc:
        elapsed = (time.time() - start) * 1000
        print(f"  {_err(f'All URLs failed  ({elapsed:.0f}ms)', color)}")
        print(f"\n  Detail: {exc}")
        _suggest_probe_errors(cfg, color)
    except Exception as exc:
        elapsed = (time.time() - start) * 1000
        print(f"  {_err(f'Unexpected error  ({elapsed:.0f}ms): {exc}', color)}")

    print()


def _suggest_probe_errors(cfg: Dict[str, Any], color: bool) -> None:
    wol = cfg.get("wake_on_lan")
    print(f"\n  {_info('Suggestions:', color)}")
    if not wol:
        print(f"    • No WoL config found – add wake_on_lan to your YAML/env to enable automatic wake.")
    else:
        print(f"    • WoL config is present. Check that:")
        print(f"        - The mac_address is correct for the sleeping host.")
        print(f"        - This machine can broadcast UDP to the target network.")
        print(f"        - retry_timeout_seconds is long enough for the host to fully boot.")
        print(f"    • Run --send-wol then --probe to test the two steps separately.")
    print(f"    • Run --health to see per-host reachability.")
    print(f"    • Run --dry-run to verify the config parsing is correct.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    color = not args.no_color and sys.stdout.isatty()

    # If no mode flag given, default to --show-config + --dry-run
    no_mode = not any([args.show_config, args.health, args.dry_run, args.send_wol, args.probe])
    if no_mode:
        args.show_config = True
        args.dry_run = True

    # Verbose logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    # Load .env
    env_path = _load_env(args.env_file)
    print(f"\n  {_info(f'Loaded env: {env_path}', color)}")

    # Resolve config (may be augmented by --url / --mac)
    cfg = _resolve_wol_config(args.url)

    # Inject CLI-supplied WoL overrides when no config exists
    if args.mac and not cfg.get("wake_on_lan"):
        target_host = args.host or (urlparse(cfg["urls"][0]).hostname if cfg["urls"] else "*")
        cfg["wake_on_lan"] = {
            "enabled": True,
            "targets": [{
                "host": target_host,
                "mac_address": args.mac,
                "port": args.wol_port or 9,
            }],
        }

    # Run modes
    if args.show_config:
        cmd_show_config(cfg, color)

    if args.dry_run:
        cmd_dry_run(cfg, color)

    if args.health:
        cmd_health(cfg, color)

    if args.send_wol:
        cmd_send_wol(cfg, args.host, args.mac, args.ip, args.wol_port, color)

    if args.probe:
        cmd_probe(cfg, args.model, color)


if __name__ == "__main__":
    main()
