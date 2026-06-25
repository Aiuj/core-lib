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


def test_check_vertex_success(monkeypatch):
    class FakeModel:
        def __init__(self, name):
            self.name = name

    class FakeModels:
        def list(self):
            return [FakeModel("publishers/google/models/gemini-3.5-flash")]

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    import sys
    from types import ModuleType
    import google
    
    # We dynamically mock google.genai and google.oauth2.service_account inside the google package namespace
    fake_genai = ModuleType("genai")
    fake_genai.Client = FakeClient
    monkeypatch.setattr(google, "genai", fake_genai, raising=False)
    
    fake_oauth = ModuleType("oauth2")
    fake_service_account = ModuleType("service_account")
    class FakeCredentials:
        @classmethod
        def from_service_account_file(cls, filename, scopes):
            return FakeCredentials()
    fake_service_account.Credentials = FakeCredentials
    fake_oauth.service_account = fake_service_account
    monkeypatch.setattr(google, "oauth2", fake_oauth, raising=False)

    # Mock os.path.isfile to return True for the credential file
    monkeypatch.setattr("os.path.isfile", lambda p: True)

    status, details, error = startup_preflight._check_vertex(
        model="gemini-3.5-flash",
        project="test-project",
        location="us-central1",
        service_account_file="/fake/path.json",
    )

    assert status == "ok"
    assert details == "Vertex AI model verified"
    assert error is None

