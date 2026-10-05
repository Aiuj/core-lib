"""Native decision transport, safety and YAML routing contracts."""

import json

import httpx
import pytest

from core_lib.llm import (
    Choice,
    Noul,
    Score,
    DecisionClient,
    DecisionError,
    create_decision_client,
)
from core_lib.llm.provider_registry import ProviderConfig, ProviderRegistry

QUESTIONS = {
    "route": Choice(
        "Which action fits?", {"search": "Read evidence", "draft": "Write a draft"}
    ),
    "urgent": Noul("Is this urgent?"),
    "quality": Score("Rate evidence quality", ["Weak", "Strong"]),
}
RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "route": {
            "type": "choice",
            "choice": "search",
            "confidence": 0.8,
            "probabilities": {"search": 0.9, "draft": 0.1},
        },
        "urgent": {"type": "noul", "noul": 0.2},
        "quality": {
            "type": "score",
            "score": 0.8,
            "confidence": 0.6,
            "probabilities": {"0": 0.2, "1": 0.8},
            "legend": {"0": "Weak", "1": "Strong"},
        },
    },
    "usage": {"input_tokens": 100, "output_tokens": 3},
}


@pytest.fixture
def make_client(monkeypatch):
    clients = []
    logs = []
    monkeypatch.setattr(
        "core_lib.llm.decision_client.log_llm_usage", lambda **kw: logs.append(kw)
    )
    monkeypatch.setattr("core_lib.llm.decision_client.time.sleep", lambda delay: None)

    def make(handler, provider="typesafe", **options):
        http = httpx.Client(transport=httpx.MockTransport(handler))
        clients.append(http)
        client = DecisionClient(
            provider,
            api_key="test-secret" if provider == "typesafe" else None,
            http_client=http,
            **options,
        )
        return client, logs

    yield make
    for http in clients:
        http.close()


@pytest.mark.parametrize(
    "provider,base_url,model",
    [
        ("typesafe", "https://api.typesafe.ai", "jev-latest"),
        ("typesafe", "https://api.typesafe.ai/v1/", "jev-latest"),
        ("ollama-decision", "http://localhost:11434", "nimble"),
        ("ollama-decision", "http://localhost:11434/v1", "nimble"),
    ],
)
def test_native_batch_request_and_version_usage(make_client, provider, base_url, model):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                **RESPONSE,
                "model": (
                    "nimble" if provider == "ollama-decision" else RESPONSE["model"]
                ),
            },
        )

    client, logs = make_client(handler, provider, base_url=base_url)
    result = client.system_one(
        state={"text": "Check sources before drafting"}, questions=QUESTIONS
    )
    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url).endswith("/v1/systemone")
    assert "/v1/v1/" not in str(request.url)
    assert payload["model"] == model
    assert set(payload) == {"state", "model", "questions"}
    assert payload["questions"]["urgent"] == {
        "type": "noul",
        "instructions": "Is this urgent?",
    }
    assert (request.headers.get("Authorization") == "Bearer test-secret") == (
        provider == "typesafe"
    )
    assert result["requested_model"] == model
    assert result["answers"]["route"]["choice"] == "search"
    assert result["usage"] == RESPONSE["usage"]
    assert result["latency_ms"] >= 0
    assert logs[0]["model"] == result["model"]
    assert logs[0]["input_tokens"] == 100
    assert "state" not in logs[0]["metadata"]


def test_environment_defaults_and_auth(monkeypatch, make_client):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://typesafe.test/v1")
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.test:11434")
    with DecisionClient() as cloud, DecisionClient("ollama-decision") as local:
        assert cloud._headers()["Authorization"] == "Bearer env-key"
        assert cloud.api_url == "https://typesafe.test/v1"
        assert local.api_url == "http://ollama.test:11434/v1"
        assert "Authorization" not in local._headers()
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        DecisionClient()


@pytest.mark.parametrize(
    "state,questions",
    [
        (" ", QUESTIONS),
        (None, QUESTIONS),
        ("state", {}),
        ("state", {"route": Choice("Pick", {"only": "Single option"})}),
        ("state", {"q": {"type": "other", "instructions": "Ask"}}),
        ("state", {"q": {"type": "noul", "instructions": " "}}),
        ("state", {"q": {"type": "noul", "instructions": "Ask", "tools": []}}),
        ("state", {"q": Noul("Ask", {"maybe": "Perhaps"})}),
        ({"score": float("nan")}, QUESTIONS),
    ],
)
def test_invalid_input_never_reaches_network(make_client, state, questions):
    def fail(request):
        pytest.fail("Invalid input reached the API")

    client, _ = make_client(fail)
    with pytest.raises((ValueError, TypeError)):
        client.decide(state=state, questions=questions)


@pytest.mark.parametrize(
    "provider,question",
    [
        ("ollama-decision", Choice("Pick", {str(i): "option" for i in range(27)})),
        ("typesafe", Choice("Pick", {str(i): "option" for i in range(256)})),
        ("typesafe", Score("Rate", ["level"] * 11)),
        ("ollama-decision", Score("Rate", ["level"] * 27)),
    ],
)
def test_provider_specific_limits(make_client, provider, question):
    client, _ = make_client(
        lambda request: pytest.fail("Limit reached network"), provider
    )
    with pytest.raises(ValueError):
        client.decide(state="state", questions={"q": question})


def test_ollama_body_limit_counts_utf8(make_client):
    client, _ = make_client(
        lambda request: pytest.fail("Oversized body reached network"), "ollama-decision"
    )
    with pytest.raises(ValueError, match="64 KiB"):
        client.decide(state="é" * 34000, questions=QUESTIONS)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.update(model=None),
        lambda data: data["answers"].pop("urgent"),
        lambda data: data["answers"]["route"].update(choice="unknown"),
        lambda data: data["answers"]["route"].update(type="noul"),
        lambda data: data["answers"]["route"].update(confidence=2),
        lambda data: data["answers"]["route"].update(
            probabilities={"search": 0.1, "draft": 0.1}
        ),
        lambda data: data["answers"]["urgent"].update(noul=True),
        lambda data: data["answers"]["quality"].update(score=3),
        lambda data: data["usage"].update(input_tokens=-1),
    ],
)
def test_invalid_api_results_are_rejected(make_client, change):
    data = json.loads(json.dumps(RESPONSE))
    change(data)
    client, _ = make_client(lambda request: httpx.Response(200, json=data))
    with pytest.raises(DecisionError):
        client.decide(state="state", questions=QUESTIONS)


def test_retries_overload_honors_retry_after(make_client, monkeypatch):
    calls, delays = [], []
    monkeypatch.setattr("core_lib.llm.decision_client.time.sleep", delays.append)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                529, headers={"Retry-After": "2"}, json={"error": "busy"}
            )
        return httpx.Response(200, json=RESPONSE)

    client, _ = make_client(handler)
    client.decide(state="state", questions=QUESTIONS)
    assert len(calls) == 2
    assert delays == [2]


@pytest.mark.parametrize("status", [401, 422, 429, 529])
def test_http_error_is_safe_and_retry_is_bounded(make_client, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "test-secret private-state"})

    client, _ = make_client(handler, max_retries=1)
    with pytest.raises(DecisionError) as caught:
        client.decide(state="private-state", questions=QUESTIONS)
    assert caught.value.status_code == status
    assert "test-secret" not in str(caught.value)
    assert "private-state" not in str(caught.value)
    assert len(calls) == (2 if status in {429, 529} else 1)


def test_long_retry_after_does_not_retry_early(make_client):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "120"})

    client, _ = make_client(handler)
    with pytest.raises(DecisionError):
        client.decide(state="state", questions=QUESTIONS)
    assert len(calls) == 1


def test_timeout_does_not_replay_billable_request(make_client):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private-state test-secret", request=request)

    client, _ = make_client(handler)
    with pytest.raises(DecisionError, match="transport") as caught:
        client.decide(state="private-state", questions=QUESTIONS)
    assert "private-state" not in str(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "provider,path", [("typesafe", "/v1/models"), ("ollama-decision", "/api/tags")]
)
def test_no_token_model_listing(make_client, provider, path):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == path
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "nimble:latest", "description": "Local decision model"}
                ]
            },
        )

    client, logs = make_client(handler, provider)
    assert client.list_models()[0]["name"] == "nimble:latest"
    assert not logs
    client.close()
    assert not client._http.is_closed  # Injected clients remain caller-owned.


def test_yaml_registry_isolates_decisions_from_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    config_file = tmp_path / "llm_providers.yaml"
    config_file.write_text("""providers:
  - provider: typesafe
    model: jev-latest
    priority: 1
  - provider: ollama-decision
    model: nimble
    priority: 2
    host: http://localhost:11434
    timeout: 60
    keep_alive: 5m
  - provider: ollama
    model: chat-model
    priority: 20
""")
    registry = ProviderRegistry.from_file(str(config_file), substitute_env=False)
    assert [p.model for p in registry.providers] == ["chat-model"]
    assert [p.model for p in registry.get_providers_for_usage("chat")] == ["chat-model"]
    assert [p.model for p in registry.decision_providers] == ["jev-latest", "nimble"]
    assert len(registry.all_providers) == 3
    assert all(not p.supports_tools for p in registry.decision_providers)
    with create_decision_client("ollama-decision", registry=registry) as client:
        assert client.model == "nimble"
        assert client.keep_alive == "5m"
        assert client.timeout == 60
    with registry.decision_providers[0].to_client() as client:
        assert isinstance(client, DecisionClient)
    with pytest.raises(ValueError, match="not chat"):
        registry.decision_providers[0].to_llm_config()


def test_disabled_and_unconfigured_decisions_are_not_selected(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    registry = ProviderRegistry(
        [
            ProviderConfig(provider="typesafe"),
            ProviderConfig(provider="ollama-decision", enabled=False),
        ]
    )
    assert not registry.decision_providers
    with pytest.raises(ValueError, match="No configured"):
        create_decision_client(registry=registry)


def test_environment_only_decision_registration(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    monkeypatch.setenv("OLLAMA_DECISION_MODEL", "nimble")
    registry = ProviderRegistry()
    registry._load_legacy_env_vars()
    assert [p.model for p in registry.decision_providers] == ["jev-latest", "nimble"]
    assert not any(p.is_decision_provider for p in registry.providers)


def test_environment_file_with_only_decisions_is_honored(tmp_path, monkeypatch):
    config_file = tmp_path / "decisions.yaml"
    config_file.write_text(
        "providers:\n  - provider: ollama-decision\n    model: nimble\n    host: http://custom-host:11434\n"
    )
    monkeypatch.setenv("LLM_PROVIDERS_FILE", str(config_file))
    registry = ProviderRegistry.from_env()
    assert len(registry.all_providers) == 1
    assert registry.decision_providers[0].host == "http://custom-host:11434"


def test_invalid_json_is_safe(make_client):
    client, _ = make_client(
        lambda request: httpx.Response(200, text="private-state not-json")
    )
    with pytest.raises(DecisionError, match="invalid JSON"):
        client.decide(state="private-state", questions=QUESTIONS)


def test_ollama_keep_alive_and_raw_questions(make_client):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["keep_alive"] == "5m"
        assert payload["questions"] == {"q": {"type": "noul", "instructions": "Ready?"}}
        return httpx.Response(
            200,
            json={
                "model": "nimble",
                "answers": {"q": {"type": "noul", "noul": 0.7}},
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        )

    client, _ = make_client(handler, "ollama-decision", keep_alive="5m")
    assert (
        client.decide(
            state="Ready", questions={"q": {"type": "noul", "instructions": "Ready?"}}
        )["model"]
        == "nimble"
    )


def test_native_connectivity_probes_do_not_use_chat(monkeypatch):
    from core_lib.llm.startup_preflight import _probe_connectivity, _probe_provider
    from unittest.mock import MagicMock

    client = MagicMock()
    client.__enter__.return_value = client
    client.list_models.return_value = [{"name": "nimble:latest"}]
    monkeypatch.setattr(ProviderConfig, "to_decision_client", lambda self, **kw: client)
    local = ProviderConfig(provider="ollama-decision")
    assert _probe_connectivity(local)[0] == "ok"
    client.chat.assert_not_called()
    client.decide.assert_not_called()
    client.list_models.return_value = [{"name": "chat-model:latest"}]
    assert _probe_connectivity(local)[0] == "down"
    assert _probe_provider(local) is None
    client.decide.assert_called_once()
    client.chat.assert_not_called()


def test_jev_and_nimble_usage_pricing():
    from core_lib.tracing.service_usage import calculate_llm_cost

    assert calculate_llm_cost(
        "typesafe", "jev-1.13.0", 1_000_000, 100
    ) == pytest.approx(0.042)
    assert calculate_llm_cost("ollama-decision", "nimble", 1_000_000, 100) == 0
