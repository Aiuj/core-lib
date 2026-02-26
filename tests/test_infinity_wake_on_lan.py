"""Tests for Infinity Wake-on-LAN recovery behavior."""

import time
from unittest.mock import Mock

import requests

from core_lib.api_utils.infinity_api import InfinityAPIClient
from core_lib.api_utils.wake_on_lan import WakeOnLanStrategy


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_wol_strategy_applies_initial_timeout_when_enabled():
    strategy = WakeOnLanStrategy(
        {
            "enabled": True,
            "initial_timeout_seconds": 2,
            "targets": [
                {
                    "mac_address": "FC:34:97:9E:C8:AF",
                    "port": 7777,
                }
            ],
        }
    )

    assert strategy.maybe_get_initial_timeout("http://powerspec:7997", 30) == 2
    assert strategy.maybe_get_initial_timeout("http://localhost:7997", 30) == 2


def test_wol_strategy_wakes_host_only_once(monkeypatch):
    strategy = WakeOnLanStrategy(
        {
            "enabled": True,
            "targets": [
                {
                    "mac_address": "FC:34:97:9E:C8:AF",
                    "port": 7777,
                    "wait_seconds": 0,
                    "retry_timeout_seconds": 8,
                }
            ],
        }
    )

    sent = {"count": 0}

    def _fake_send(_target):
        sent["count"] += 1

    monkeypatch.setattr(strategy, "_send_magic_packet", _fake_send)

    first = strategy.maybe_wake("http://powerspec:7997", RuntimeError("timeout"))
    second = strategy.maybe_wake("http://powerspec:7997", RuntimeError("timeout"))

    assert first.attempted is True
    assert first.succeeded is True
    assert first.retry_timeout_seconds == 8
    assert second.attempted is False
    assert sent["count"] == 1


def test_infinity_client_retries_after_wol_on_timeout(monkeypatch):
    client = InfinityAPIClient(
        base_urls=["http://powerspec:7997", "http://127.0.0.1:7997"],
        timeout=30,
        max_retries_per_url=1,
        wake_on_lan={
            "enabled": True,
            "initial_timeout_seconds": 2,
            "targets": [
                {
                    "mac_address": "FC:34:97:9E:C8:AF",
                    "port": 7777,
                    "wait_seconds": 0,
                    "retry_timeout_seconds": 8,
                }
            ],
        },
    )

    post_mock = Mock(
        side_effect=[
            requests.exceptions.Timeout("simulated timeout"),
            _Response({"data": [{"index": 0, "embedding": [0.1]}]}),
        ]
    )
    monkeypatch.setattr("core_lib.api_utils.infinity_api.requests.post", post_mock)

    data, used_url = client.post("/embeddings", json={"input": ["hello"], "model": "x"})

    assert used_url == "http://powerspec:7997"
    assert data["data"][0]["embedding"] == [0.1]
    assert post_mock.call_count == 2
    assert post_mock.call_args_list[0].kwargs["timeout"] == 2
    assert post_mock.call_args_list[1].kwargs["timeout"] == 8


def test_infinity_client_fails_over_without_wol_when_disabled(monkeypatch):
    client = InfinityAPIClient(
        base_urls=["http://powerspec:7997", "http://127.0.0.1:7997"],
        timeout=30,
        max_retries_per_url=1,
        wake_on_lan={"enabled": False},
    )

    post_mock = Mock(
        side_effect=[
            requests.exceptions.Timeout("simulated timeout"),
            _Response({"data": [{"index": 0, "embedding": [0.2]}]}),
        ]
    )
    monkeypatch.setattr("core_lib.api_utils.infinity_api.requests.post", post_mock)

    data, used_url = client.post("/embeddings", json={"input": ["hello"], "model": "x"})

    assert used_url == "http://127.0.0.1:7997"
    assert data["data"][0]["embedding"] == [0.2]
    assert post_mock.call_count == 2
    assert post_mock.call_args_list[0].kwargs["timeout"] == 30
    assert post_mock.call_args_list[1].kwargs["timeout"] == 30


def test_wol_defaults_host_from_base_url_and_port_9(monkeypatch):
    strategy = WakeOnLanStrategy(
        {
            "enabled": True,
            "mac_address": "FC:34:97:9E:C8:AF",
            "wait_seconds": 0,
        }
    )

    captured = {}

    def _fake_send(target):
        captured["port"] = target.port

    monkeypatch.setattr(strategy, "_send_magic_packet", _fake_send)

    timeout_value = strategy.maybe_get_initial_timeout("http://powerspec:7997", 30)
    wake_result = strategy.maybe_wake("http://powerspec:7997", RuntimeError("timeout"))

    assert timeout_value == 2
    assert wake_result.succeeded is True
    assert captured["port"] == 9


# ---------------------------------------------------------------------------
# Non-blocking warmup tests
# ---------------------------------------------------------------------------

def test_wol_is_in_warmup_false_before_wake():
    """is_in_warmup() must return False before any WoL has been sent."""
    strategy = WakeOnLanStrategy(
        {
            "enabled": True,
            "warmup_seconds": 30,
            "targets": [{"mac_address": "FC:34:97:9E:C8:AF"}],
        }
    )
    assert strategy.is_in_warmup("http://powerspec:7997") is False


def test_wol_is_in_warmup_true_during_window(monkeypatch):
    """is_in_warmup() must return True immediately after a WoL packet is sent."""
    strategy = WakeOnLanStrategy(
        {
            "enabled": True,
            "warmup_seconds": 30,
            "targets": [
                {
                    "mac_address": "FC:34:97:9E:C8:AF",
                    "wait_seconds": 0,
                }
            ],
        }
    )
    monkeypatch.setattr(strategy, "_send_magic_packet", lambda _t: None)

    url = "http://powerspec:7997"
    result = strategy.maybe_wake(url, RuntimeError("timeout"))

    assert result.succeeded is True
    assert result.warmup_seconds == 30
    assert strategy.is_in_warmup(url) is True


def test_wol_is_in_warmup_false_after_window(monkeypatch):
    """is_in_warmup() must return False once the warmup window has elapsed."""
    WARMUP = 5  # short window for the test

    strategy = WakeOnLanStrategy(
        {
            "enabled": True,
            "warmup_seconds": WARMUP,
            "targets": [
                {
                    "mac_address": "FC:34:97:9E:C8:AF",
                    "wait_seconds": 0,
                }
            ],
        }
    )
    monkeypatch.setattr(strategy, "_send_magic_packet", lambda _t: None)

    url = "http://powerspec:7997"
    strategy.maybe_wake(url, RuntimeError("timeout"))

    # Back-date the wake timestamp so the window has already expired.
    strategy._waking_timestamps[url] = time.time() - WARMUP - 1

    assert strategy.is_in_warmup(url) is False


def test_wol_is_in_warmup_false_when_warmup_seconds_not_set(monkeypatch):
    """Without warmup_seconds, is_in_warmup() always returns False (old behaviour)."""
    strategy = WakeOnLanStrategy(
        {
            "enabled": True,
            "targets": [
                {
                    "mac_address": "FC:34:97:9E:C8:AF",
                    "wait_seconds": 0,
                }
            ],
        }
    )
    monkeypatch.setattr(strategy, "_send_magic_packet", lambda _t: None)

    url = "http://powerspec:7997"
    strategy.maybe_wake(url, RuntimeError("timeout"))

    assert strategy.is_in_warmup(url) is False


def test_wol_non_blocking_does_not_sleep(monkeypatch):
    """When warmup_seconds is set, maybe_wake() must NOT call time.sleep()."""
    strategy = WakeOnLanStrategy(
        {
            "enabled": True,
            "warmup_seconds": 30,
            "targets": [
                {
                    "mac_address": "FC:34:97:9E:C8:AF",
                    "wait_seconds": 20,  # would block 20 s in classic mode
                }
            ],
        }
    )
    monkeypatch.setattr(strategy, "_send_magic_packet", lambda _t: None)

    sleep_calls = []
    monkeypatch.setattr("core_lib.api_utils.wake_on_lan.time.sleep", lambda s: sleep_calls.append(s))

    strategy.maybe_wake("http://powerspec:7997", RuntimeError("timeout"))

    assert sleep_calls == [], "time.sleep should not be called in non-blocking warmup mode"


def test_infinity_client_routes_to_secondary_during_warmup(monkeypatch):
    """During a WoL warmup window the client must use the secondary server.

    Sequence:
      1. First request → main server times out → WoL sent, warmup starts.
      2. Second request arrives while warmup is active →  client should skip
         main and serve from secondary immediately.
    """
    client = InfinityAPIClient(
        base_urls=["http://main:7997", "http://secondary:7997"],
        timeout=30,
        max_retries_per_url=1,
        wake_on_lan={
            "enabled": True,
            "warmup_seconds": 30,
            "initial_timeout_seconds": 2,
            "targets": [
                {
                    "mac_address": "FC:34:97:9E:C8:AF",
                    "wait_seconds": 0,
                    "retry_timeout_seconds": 8,
                }
            ],
        },
    )

    payload = {"data": [{"index": 0, "embedding": [0.5]}]}

    # req 1: main times out; secondary succeeds (fallback after WoL)
    # req 2: warmup is active → client skips main and goes straight to secondary
    post_mock = Mock(
        side_effect=[
            requests.exceptions.Timeout("main server sleeping"),
            _Response(payload),  # secondary handles req 1 fallback
            _Response(payload),  # secondary handles req 2 (warmup skip)
        ]
    )
    monkeypatch.setattr("core_lib.api_utils.infinity_api.requests.post", post_mock)
    monkeypatch.setattr(client.wake_on_lan, "_send_magic_packet", lambda _t: None)

    # First call: main timeouts, WoL fired, fallback to secondary
    data1, url1 = client.post("/embeddings", json={"input": ["a"], "model": "x"})
    assert url1 == "http://secondary:7997"
    assert client.wake_on_lan.is_in_warmup("http://main:7997") is True

    # Second call: warmup active → secondary served immediately (no attempt on main)
    data2, url2 = client.post("/embeddings", json={"input": ["b"], "model": "x"})
    assert url2 == "http://secondary:7997"

    # The third mock entry (req 2 secondary) should have been used;
    # total calls == 3 (timeout on main + secondary for req1 + secondary for req2)
    assert post_mock.call_count == 3


def test_infinity_client_returns_to_main_after_warmup(monkeypatch):
    """Once the warmup window expires the client must try the main server again."""
    client = InfinityAPIClient(
        base_urls=["http://main:7997", "http://secondary:7997"],
        timeout=30,
        max_retries_per_url=1,
        wake_on_lan={
            "enabled": True,
            "warmup_seconds": 30,
            "initial_timeout_seconds": 2,
            "targets": [
                {
                    "mac_address": "FC:34:97:9E:C8:AF",
                    "wait_seconds": 0,
                }
            ],
        },
    )

    payload = {"data": [{"index": 0, "embedding": [0.9]}]}

    # Simulate a woken-but-not-yet-tried state: WoL was sent 31 s ago (warmup over)
    url_main = "http://main:7997"
    client.wake_on_lan._woken_hosts.add(url_main)
    client.wake_on_lan._waking_timestamps[url_main] = time.time() - 31

    post_mock = Mock(return_value=_Response(payload))
    monkeypatch.setattr("core_lib.api_utils.infinity_api.requests.post", post_mock)

    data, url = client.post("/embeddings", json={"input": ["hello"], "model": "x"})

    # Warmup expired — main server must be tried first
    assert url == "http://main:7997"
    first_call_url = post_mock.call_args_list[0].args[0]
    assert "main" in first_call_url


# ---------------------------------------------------------------------------
# enabled: defaults
# ---------------------------------------------------------------------------

def test_wol_enabled_by_default_when_key_absent():
    """A non-empty wake_on_lan block should be enabled implicitly."""
    strategy = WakeOnLanStrategy({"mac_address": "FC:34:97:9E:C8:AF"})
    assert strategy.enabled is True


def test_wol_disabled_when_enabled_false():
    """`enabled: false` must override the implicit default."""
    strategy = WakeOnLanStrategy(
        {"enabled": False, "mac_address": "FC:34:97:9E:C8:AF"}
    )
    assert strategy.enabled is False


def test_wol_disabled_for_empty_config():
    """An empty wake_on_lan config dict should leave WoL disabled."""
    strategy = WakeOnLanStrategy({})
    assert strategy.enabled is False


# ---------------------------------------------------------------------------
# Non-blocking warmup — ConnectionError path (the user-reported bug)
# ---------------------------------------------------------------------------

def test_wol_non_blocking_skips_post_wake_retry_on_connection_error(monkeypatch):
    """In warmup mode a ConnectionError must NOT trigger a blocking post-wake retry.

    The bug: the ConnectionError handler called the post-wake retry unconditionally,
    causing a 30-second stall before failing over to the secondary server.

    Production config uses a single base_url per InfinityAPIClient (failover is
    handled by FallbackEmbeddingClient at a higher level), so after the WoL fires
    the client must raise immediately so the caller can route to the secondary.
    """
    client = InfinityAPIClient(
        base_urls=["http://main:7997", "http://secondary:7997"],
        timeout=30,
        max_retries_per_url=1,
        wake_on_lan={
            "warmup_seconds": 30,
            "initial_timeout_seconds": 2,
            "targets": [
                {
                    "mac_address": "FC:34:97:9E:C8:AF",
                    "wait_seconds": 0,
                    "retry_timeout_seconds": 8,
                }
            ],
        },
    )

    payload = {"data": [{"index": 0, "embedding": [0.5]}]}

    # Simulate a ConnectTimeoutError (which is a ConnectionError in requests)
    connect_timeout = requests.exceptions.ConnectionError(
        "HTTPConnectionPool: Max retries exceeded (Caused by ConnectTimeoutError)"
    )

    post_mock = Mock(
        side_effect=[
            connect_timeout,            # main: connection timeout (WoL fires)
            _Response(payload),          # secondary: succeeds for req 1
            _Response(payload),          # secondary: succeeds for req 2
        ]
    )
    monkeypatch.setattr("core_lib.api_utils.infinity_api.requests.post", post_mock)
    monkeypatch.setattr(client.wake_on_lan, "_send_magic_packet", lambda _t: None)

    # First call: main fails with ConnectionError → WoL fires → no blocking retry →
    # falls through to secondary immediately
    data1, url1 = client.post("/embeddings", json={"input": ["a"], "model": "x"})
    assert url1 == "http://secondary:7997", (
        "First request should fall over to secondary without a blocking post-wake retry"
    )
    assert client.wake_on_lan.is_in_warmup("http://main:7997") is True

    # Second call: warmup active → main skipped immediately → secondary
    data2, url2 = client.post("/embeddings", json={"input": ["b"], "model": "x"})
    assert url2 == "http://secondary:7997"

    # Only 3 calls total — no extra post-wake retry on main
    assert post_mock.call_count == 3, (
        f"Expected 3 requests (1 failed main + 2 secondary), got {post_mock.call_count}"
    )


def test_infinity_client_is_in_warmup_delegates_to_strategy(monkeypatch):
    """InfinityAPIClient.is_in_warmup() should reflect the WoL strategy state."""
    client = InfinityAPIClient(
        base_urls=["http://main:7997"],
        wake_on_lan={
            "warmup_seconds": 30,
            "targets": [{"mac_address": "FC:34:97:9E:C8:AF"}],
        },
    )

    assert client.is_in_warmup() is False

    # Simulate a WoL wake
    client.wake_on_lan._waking_timestamps["http://main:7997"] = time.time()
    client.wake_on_lan._woken_hosts.add("http://main:7997")

    assert client.is_in_warmup() is True
