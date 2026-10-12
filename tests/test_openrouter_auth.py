import httpx
import pytest

from core_lib.llm import openrouter_auth as oa
from core_lib.llm.openrouter_auth import OpenRouterFederationError, OpenRouterWorkloadIdentity
from core_lib.llm.provider_registry import ProviderConfig


def _fake_post(calls, status=200, expires_in=900):
    def post(url, data=None, timeout=None):
        calls.append((url, data))
        body = {"access_token": f"tok{len(calls)}", "expires_in": expires_in}
        if status != 200:
            body = {"error": "invalid_grant"}
        return httpx.Response(status, json=body)

    return post


def test_exchange_and_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(oa.httpx, "post", _fake_post(calls))
    wi = OpenRouterWorkloadIdentity("pol_1", subject_token="jwt")
    assert wi() == "tok1"
    assert wi() == "tok1"
    assert len(calls) == 1
    data = calls[0][1]
    assert data["federation_policy_id"] == "pol_1"
    assert data["subject_token"] == "jwt"
    assert data["grant_type"] == oa.GRANT_TYPE


def test_refresh_near_expiry(monkeypatch):
    calls = []
    monkeypatch.setattr(oa.httpx, "post", _fake_post(calls, expires_in=30))
    wi = OpenRouterWorkloadIdentity("pol_1", subject_token=lambda: "jwt")
    assert wi() == "tok1"
    assert wi() == "tok2"


def test_token_file_reread(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(oa.httpx, "post", _fake_post(calls, expires_in=0))
    f = tmp_path / "t"
    f.write_text("a\n")
    wi = OpenRouterWorkloadIdentity("p", subject_token_file=str(f))
    wi()
    f.write_text("b")
    wi()
    assert [c[1]["subject_token"] for c in calls] == ["a", "b"]


def test_error(monkeypatch):
    monkeypatch.setattr(oa.httpx, "post", _fake_post([], status=400))
    with pytest.raises(OpenRouterFederationError, match="invalid_grant"):
        OpenRouterWorkloadIdentity("p", subject_token="j")()


def test_registry_uses_federation(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = ProviderConfig.from_dict({
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        "workload_identity": {"policy_id": "p", "subject_token": "j"},
    })
    assert cfg.is_configured()
    assert isinstance(cfg.to_llm_config().api_key, OpenRouterWorkloadIdentity)


def test_registry_static_key_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = ProviderConfig.from_dict({"provider": "openrouter", "api_key": "sk"})
    assert cfg.to_llm_config().api_key == "sk"
