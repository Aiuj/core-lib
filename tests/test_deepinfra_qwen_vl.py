"""Pricing and provider-interface coverage for DeepInfra Qwen3-VL OCR models."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core_lib.llm import FallbackLLMClient, ProviderRegistry
from core_lib.llm.provider_health import ProviderHealthTracker
from core_lib.llm.provider_registry import ProviderConfig
from core_lib.tracing.service_pricing import get_llm_pricing


@pytest.mark.parametrize(
    ("model", "input_price", "output_price"),
    [
        ("Qwen/Qwen3-VL-30B-A3B-Instruct", 0.00015, 0.00060),
        ("Qwen/Qwen3-VL-235B-A22B-Instruct", 0.00020, 0.00088),
    ],
)
def test_deepinfra_qwen_vl_pricing(model, input_price, output_price):
    pricing = get_llm_pricing(model)

    assert pricing is not None
    assert pricing["input"] == pytest.approx(input_price)
    assert pricing["output"] == pytest.approx(output_price)


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
    ],
)
def test_deepinfra_qwen_vl_uses_openai_compatible_interface(model):
    config = ProviderConfig.from_dict(
        {
            "provider": "deepinfra",
            "api_key": "test-key",
            "model": model,
            "usage": ["vision", "ocr"],
        }
    )

    assert config.provider == "openai"
    assert config.host == "https://api.deepinfra.com/v1/openai"
    assert config.model == model
    assert config.usage == ["vision", "ocr"]


@pytest.mark.parametrize(
    "fallback_model",
    [
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
    ],
)
def test_glm_server_failure_routes_multimodal_payload_to_deepinfra(
    fallback_model, monkeypatch
):
    glm = ProviderConfig(
        provider="openai",
        api_key="local",
        host="http://ocr.invalid/v1",
        model="zai-org/GLM-OCR",
        priority=1,
        usage=["vision", "ocr"],
    )
    deepinfra = ProviderConfig.from_dict(
        {
            "provider": "deepinfra",
            "api_key": "test-key",
            "model": fallback_model,
            "priority": 2,
            "usage": ["vision", "ocr"],
        }
    )
    client = FallbackLLMClient(
        ProviderRegistry([glm, deepinfra]),
        health_tracker=ProviderHealthTracker(cache_client=False),
        max_retries_per_provider=1,
        usage="ocr",
    )
    glm_client = MagicMock()
    glm_client.chat.side_effect = ConnectionError("GLM server is down")
    deepinfra_client = MagicMock()
    deepinfra_client.chat.return_value = {
        "content": "Invoice total: $42.00",
        "usage": {"prompt_tokens": 100, "completion_tokens": 8},
        "error": None,
    }
    clients = {
        "zai-org/GLM-OCR": glm_client,
        fallback_model: deepinfra_client,
    }
    monkeypatch.setattr(client, "_get_client", lambda config: clients[config.model])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0"}},
                {"type": "text", "text": "Transcribe this document."},
            ],
        }
    ]

    response = client.chat(messages=messages)

    assert response["content"] == "Invoice total: $42.00"
    assert response["_fallback_metadata"] == {
        "provider": "openai",
        "model": fallback_model,
        "was_fallback": True,
        "attempts": 2,
    }
    deepinfra_client.chat.assert_called_once()
    assert deepinfra_client.chat.call_args.kwargs["messages"] == messages
