"""Tests for Infinity Wake-on-LAN recovery behavior."""

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
