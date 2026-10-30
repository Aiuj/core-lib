"""An overloaded inference server (HTTP 429/503) is retried briefly before failing over."""
from unittest.mock import Mock

import pytest
import requests

from core_lib.api_utils import infinity_api
from core_lib.api_utils.infinity_api import InfinityAPIClient, InfinityAPIError


class _Response:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"results": [{"index": 0, "score": 0.9}]}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._payload


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(infinity_api.time, "sleep", waited.append)
    return waited


def _client(post, urls=("http://gpu:8111",)):
    return InfinityAPIClient(base_urls=list(urls), timeout=5, max_retries_per_url=1, http_client=Mock(
        post=post, exceptions=requests.exceptions))


def test_momentary_overload_is_retried_on_the_same_server(sleeps):
    post = Mock(side_effect=[_Response(429), _Response(429), _Response(200)])
    data, used_url = _client(post).post("/rerank", json={"query": "q", "texts": ["t"]})

    assert used_url == "http://gpu:8111"
    assert data["results"][0]["score"] == 0.9
    assert post.call_count == 3
    assert sleeps == [0.1, 0.2]


def test_short_retry_after_is_honoured_and_long_ones_are_capped(sleeps):
    post = Mock(side_effect=[_Response(503, headers={"Retry-After": "0.3"}),
                             _Response(429, headers={"Retry-After": "120"}), _Response(200)])
    _client(post).post("/rerank", json={})

    assert sleeps == [0.3, infinity_api.OVERLOAD_MAX_DELAY_SECONDS]


def test_persistent_overload_fails_over_after_bounded_retries(sleeps):
    post = Mock(side_effect=[_Response(429)] * (infinity_api.OVERLOAD_RETRIES + 1) + [_Response(200)])
    data, used_url = _client(post, urls=("http://gpu:8111", "http://backup:8111")).post("/rerank", json={})

    assert used_url == "http://backup:8111"
    assert post.call_count == infinity_api.OVERLOAD_RETRIES + 2
    assert sum(sleeps) < 1.0


def test_other_client_errors_are_not_retried(sleeps):
    post = Mock(side_effect=[_Response(400)])
    with pytest.raises(InfinityAPIError):
        _client(post).post("/rerank", json={})

    assert post.call_count == 1
    assert sleeps == []
