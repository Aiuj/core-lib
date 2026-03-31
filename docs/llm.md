# LLM Client Usage Guide

The LLM module provides a unified interface for working with different Large Language Model providers.  
For **multi-provider fallback with health tracking and per-task routing**, see [FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md).

---

## Quick Start

### Single Provider (LLMClient)

```python
from core_lib.llm import create_llm_client

# Auto-detect provider from environment variables
client = create_llm_client()
response = client.chat("What is the capital of France?")
print(response["content"])
```

### Multi-Provider with Fallback (Recommended for Production)

```python
from core_lib.llm import FallbackLLMClient

# Load providers from llm_providers.yaml (set LLM_PROVIDERS_FILE env var)
client = FallbackLLMClient.from_env()
response = client.chat("What is the capital of France?")
print(response["content"])
```

See [FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md) for configuration, `usage=` routing, intelligence levels, and health tracking.

---

## Single-Provider Usage (LLMClient)

### Creating a Client

**Factory function — auto-detect from environment:**
```python
from core_lib.llm import create_llm_client

client = create_llm_client()                              # auto-detect
client = create_llm_client(provider="openai", model="gpt-4o")
client = create_llm_client(provider="gemini", model="gemini-2.5-flash")
client = create_llm_client(provider="ollama", model="qwen3.5:4b")
```

**Provider-specific factory functions:**
```python
from core_lib.llm import (
    create_gemini_client,
    create_ollama_client,
    create_openai_responses_client,
    create_alibaba_client,
    create_azure_openai_client,
)

client = create_ollama_client(model="qwen3.5:4b", temperature=0.1)
client = create_gemini_client(api_key="your-key", model="gemini-2.5-flash")
client = create_openai_responses_client(api_key="sk-...", model="gpt-4.1")
client = create_alibaba_client(model="qwen3-max")      # reads DASHSCOPE_API_KEY
client = create_azure_openai_client()                   # reads AZURE_OPENAI_* env vars
```

**Using the factory class:**
```python
from core_lib.llm import LLMFactory

client = LLMFactory.gemini(model="gemini-2.5-pro")
client = LLMFactory.ollama(model="qwen3.5:4b", base_url="http://localhost:11434")
client = LLMFactory.openai(model="gpt-4o", temperature=0.3)
client = LLMFactory.openai_responses(model="gpt-4.1", reasoning_effort="low")
client = LLMFactory.alibaba(model="qwen3-max", thinking_enabled=True)
client = LLMFactory.azure_openai(deployment="gpt-4o")
client = LLMFactory.openrouter(model="anthropic/claude-3.5-sonnet")
```

---

## Chat Interface

All clients share the same `chat()` interface:

```python
response = client.chat(messages)
print(response["content"])
```

### Response Format

```python
{
    "content": str | dict,    # text, or dict when structured_output is used
    "structured": bool,       # True if structured_output was requested
    "tool_calls": list,       # function calls requested by the model (if any)
    "usage": dict,            # token usage statistics
    "error": str | None,      # present on failure
    # OpenAI Responses API only:
    "response_id": str,       # pass back as previous_response_id for stateful multi-turn
}
```

Always check for errors:
```python
response = client.chat("Hello")
if response.get("error"):
    print(f"Error: {response['error']}")
else:
    print(response["content"])
```

---

## Common Patterns

### Simple Chat

```python
response = client.chat("Explain quantum computing in one paragraph")
print(response["content"])
```

### System Message + Multi-Turn

```python
messages = [
    {"role": "user", "content": "Hello, I'm planning a trip to Japan."},
    {"role": "assistant", "content": "Great! What would you like to know?"},
    {"role": "user", "content": "What are the best times to visit Kyoto?"}
]

response = client.chat(messages, system_message="You are a helpful travel expert.")
print(response["content"])
```

### Structured Output (Pydantic)

Pass a Pydantic model class to get validated structured output:

```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    location: str
    temperature: float
    condition: str

response = client.chat(
    "What's the weather like in Paris today?",
    structured_output=WeatherReport,
)

if response["structured"] and not response.get("error"):
    report = response["content"]           # already a dict (model_dump())
    print(f"{report['location']}: {report['temperature']}°C, {report['condition']}")
```

### Tool / Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat("What's the weather in Tokyo?", tools=tools)
if response["tool_calls"]:
    print(response["tool_calls"])
```

### Thinking / Reasoning Mode

Supported by Gemini 2.5+ and Qwen3/3.5 models:

```python
client = create_ollama_client(model="qwen3.5:4b", thinking_enabled=True)
response = client.chat("Explain why P≠NP is hard to prove")
print(response["content"])
```

### Vision and Image Input

Multimodal models (Gemini, qwen3.5, qwen2.5vl) accept images via the standard message format:

```python
import base64

with open("chart.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe what you see in this image"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ]
}]

# qwen3.5 is fully multimodal (all sizes: 0.8b–122b)
client = create_ollama_client(model="qwen3.5:4b")
response = client.chat(messages)
print(response["content"])
```

### Google Search Grounding (Gemini)

```python
from core_lib.llm import create_gemini_client

client = create_gemini_client(model="gemini-2.5-flash")
response = client.chat(
    "What are the latest updates on the Mars mission?",
    use_search_grounding=True,
)
print(response["content"])
```

---

## Provider-Specific Features

### OpenAI Responses API (Stateful Multi-Turn)

```python
from core_lib.llm import create_openai_responses_client

client = create_openai_responses_client(api_key="sk-...")

resp1 = client.chat("My name is Alice.")
prev_id = resp1["response_id"]

client.config.previous_response_id = prev_id
resp2 = client.chat("What is my name?")
print(resp2["content"])   # "Alice" — context preserved server-side
```

### Alibaba Cloud / Qwen (DashScope)

```python
from core_lib.llm import create_alibaba_client

# Reads DASHSCOPE_API_KEY automatically
client = create_alibaba_client(model="qwen3-max", thinking_enabled=True)
response = client.chat("Solve: if 3x + 7 = 22, what is x?")
print(response["content"])

# China Beijing region
client = create_alibaba_client(model="qwen3-max", region="china")
```

### Azure OpenAI

```python
from core_lib.llm import create_azure_openai_client

# From env vars (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT)
client = create_azure_openai_client()

# Explicit parameters
client = create_azure_openai_client(
    api_key="your-key",
    azure_endpoint="https://my-resource.openai.azure.com",
    deployment="gpt-4o",
)
response = client.chat("Summarize in one sentence.")
print(response["content"])
```

### OpenRouter (300+ Models)

```python
from core_lib.llm import create_llm_client

# OPENROUTER_API_KEY from env
client = create_llm_client(provider="openrouter", model="anthropic/claude-3.5-sonnet")
response = client.chat("Explain the Renaissance briefly.")
print(response["content"])
```

### Ollama (Local)

```python
from core_lib.llm import create_ollama_client

client = create_ollama_client(
    model="qwen3.5:4b",
    base_url="http://localhost:11434",
    temperature=0.1,
    thinking_enabled=False,   # Disable for faster, deterministic output
)
response = client.chat("Classify this text: ...")
```

---

## Environment Variables

### Auto-Detection Priority

1. `LLM_PROVIDER` — explicit: `gemini`, `openai`, `openai-responses`, `openrouter`, `azure`, `ollama`, `alibaba`
2. `GEMINI_API_KEY` or `GOOGLE_GENAI_API_KEY` → Gemini
3. `OPENAI_API_KEY` → OpenAI Chat Completions
4. `AZURE_OPENAI_API_KEY` → Azure OpenAI
5. Default fallback → Ollama

### Gemini (Developer API)
| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | API key (also `GOOGLE_GENAI_API_KEY`) |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Model name |
| `GEMINI_TEMPERATURE` | `0.1` | Sampling temperature |
| `GEMINI_MAX_TOKENS` | — | Max output tokens |
| `GEMINI_THINKING_ENABLED` | — | Enable thinking mode (`true`/`false`) |

### Gemini (Vertex AI via ADC)
| Variable | Description |
|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | Region, e.g. `us-central1` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON |

### OpenAI Responses API (`openai-responses` / `alibaba`)
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` | — | API key |
| `OPENAI_RESPONSES_MODEL` | `gpt-4.1` | Model name |
| `OPENAI_BASE_URL` | — | Custom endpoint (set for Alibaba) |
| `OPENAI_TEMPERATURE` | `0.7` | Sampling temperature |
| `OPENAI_THINKING_ENABLED` | — | Enable chain-of-thought |
| `OPENAI_REASONING_EFFORT` | `medium` | `low`/`medium`/`high` for o-series |

### OpenAI Chat Completions
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_TEMPERATURE` | `0.7` | Sampling temperature |
| `OPENAI_MAX_TOKENS` | — | Max output tokens |
| `OPENAI_ORG` / `OPENAI_PROJECT` | — | Organization/project IDs |

### Azure OpenAI
| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_API_KEY` | — | Azure API key |
| `AZURE_OPENAI_ENDPOINT` | — | Resource endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o-mini` | Deployment/model name |
| `AZURE_OPENAI_API_VERSION` | `2024-08-01-preview` | REST API version |

### Ollama
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen3:1.7b` | Model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Server URL |
| `OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature |
| `OLLAMA_THINKING_ENABLED` | — | Enable thinking mode |
| `OLLAMA_TIMEOUT` | `60` | Request timeout (seconds) |
| `OLLAMA_NUM_CTX` | — | Context window size |

---

## Provider Comparison

| Feature | Ollama | Gemini | OpenAI (Chat) | OpenAI Responses | OpenRouter | Alibaba (Qwen) |
|---------|--------|--------|---------------|-----------------|------------|----------------|
| Cost | Free (local) | Paid | Paid | Paid | Paid | Paid |
| Privacy | Local | Cloud | Cloud | Cloud | Cloud | Cloud |
| Structured Output | JSON + Pydantic | `response_schema` | `response_format` | `text.format` | `response_format` | `text.format` |
| Tool Calling | Model-dependent | Native | Native | Native | Model-dependent | Native |
| Vision / Multimodal | qwen3.5, qwen2.5vl | All | GPT-4o+ | GPT-4o+ | Model-dependent | Qwen-VL models |
| Thinking / Reasoning | qwen3, qwen3.5 | 2.5-series | Not forwarded | o-series | Model-dependent | Qwen3+ |
| Web search grounding | ❌ | ✅ Google Search | ❌ | ✅ `web_search_preview` | Model-dependent | ✅ |
| Stateful multi-turn | ❌ | ❌ | ❌ | ✅ `previous_response_id` | ❌ | ✅ |
| Rate limiting | None (local) | Auto (model-specific) | None | None | None | None |
| Auto-retry | None | Exponential backoff | None | None | None | None |

---

## Resilience Features (Gemini)

Gemini clients include built-in rate limiting and retry logic:

- **Per-model RPM limits**: Gemini 2.5 Pro: 5 RPM, Flash: 10 RPM, Flash-Lite: 15 RPM
- **Automatic retry**: Up to 3 retries with exponential backoff (1s → 2s → 4s + jitter) on rate limits, server errors, and network failures
- **Circuit breaker**: Partial failures don't block the entire system

No extra configuration needed — these operate transparently.

For **cross-provider failover**, use `FallbackLLMClient` (see [FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md)).

---

## Advanced: Vertex AI Context Caching

Reduce costs and latency for large, repeated contexts using Vertex AI context caching.

### Implicit Caching (Automatic)

Vertex AI automatically caches context when:
- Prompt is **> 2048 tokens**
- Large static content is at the **beginning** of the prompt
- Subsequent requests share the same prefix

**~90% cost reduction** on cached tokens; no code changes required.

### Explicit Caching

```python
from google import genai
from google.genai.types import CreateCachedContentConfig

# Create a cache once
gclient = genai.Client(vertexai=True, project="my-project", location="us-central1")
cache = gclient.caches.create(
    model="gemini-2.0-flash",
    config=CreateCachedContentConfig(
        contents=[...],        # Your large static context
        ttl="3600s",
        display_name="my-rfx-cache"
    )
)
cache_name = cache.name

# Use the cache via core-lib LLMClient
from core_lib.llm import create_gemini_client
client = create_gemini_client(model="gemini-2.0-flash", project="my-project", location="us-central1")
response = client.chat(
    messages=[{"role": "user", "content": "Analyze section 4 of the cached RFx."}],
    cached_content=cache_name,
)
```

---

## References

- [FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md) — Multi-provider configuration, usage-based routing, health tracking
- [PROVIDER_REGISTRY_QUICK_REFERENCE.md](PROVIDER_REGISTRY_QUICK_REFERENCE.md) — ProviderRegistry API reference
- [LLM_QUICK_REFERENCE.md](LLM_QUICK_REFERENCE.md) — Cheat sheet
- Google GenAI Python SDK: https://googleapis.github.io/python-genai/
- Gemini Structured Output: https://ai.google.dev/gemini-api/docs/structured-output
