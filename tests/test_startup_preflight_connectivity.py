"""Connectivity checks for startup preflight helpers."""

from urllib.error import HTTPError

from core_lib.llm import startup_preflight


class _FakeHTTPResponse:
    def __init__(self, status: int = 200, body: bytes = b""):
        self.status = status
        self._body = body

    def read(self):
        return self._body


def _http_error(url: str, code: int = 404, reason: str = "Not Found") -> HTTPError:
    return HTTPError(url=url, code=code, msg=reason, hdrs=None, fp=_FakeHTTPResponse(code, b""))


def test_check_openai_compatible_uses_models_list_success(monkeypatch):
    calls = []

    def _fake_http_get(url, headers, timeout=8):
        calls.append(url)
        assert headers["Authorization"].startswith("Bearer ")
        return 200, b'{"data":[{"id":"mistralai/Ministral-3-3B-Instruct-2512"}]}'

    monkeypatch.setattr(startup_preflight, "_http_get", _fake_http_get)

    status, details, error = startup_preflight._check_openai_compatible(
        model="mistralai/Ministral-3-3B-Instruct-2512",
        api_key="test-key",
        base_url="http://localhost:8101/v1",
    )

    assert status == "ok"
    assert error is None
    assert details == "1 model(s) available"
    assert calls == ["http://localhost:8101/v1/models"]


def test_check_openai_compatible_fallback_encodes_model_path(monkeypatch):
    calls = []

    def _fake_http_get(url, headers, timeout=8):
        calls.append(url)
        if url.endswith("/models"):
            raise _http_error(url, 404)
        if url.endswith("/models/mistralai%2FMinistral-3-3B-Instruct-2512"):
            return 200, b"{}"
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(startup_preflight, "_http_get", _fake_http_get)

    status, details, error = startup_preflight._check_openai_compatible(
        model="mistralai/Ministral-3-3B-Instruct-2512",
        api_key="test-key",
        base_url="http://localhost:8101/v1",
    )

    assert status == "ok"
    assert details is None
    assert error is None
    assert calls == [
        "http://localhost:8101/v1/models",
        "http://localhost:8101/v1/models/mistralai%2FMinistral-3-3B-Instruct-2512",
    ]
