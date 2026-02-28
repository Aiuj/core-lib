# Alibaba Cloud (Qwen) Setup Guide

This guide explains how to configure and use Alibaba Cloud's Qwen models via the `OpenAIProvider`
in core-lib. Alibaba Cloud Model Studio exposes a **Chat Completions** endpoint that is fully
compatible with the OpenAI SDK, so no special provider is needed.

---

## Compatibility Matrix

| core-lib API | Alibaba Endpoint | Status |
|---|---|---|
| `OpenAIProvider` (Chat Completions) | `/compatible-mode/v1/chat/completions` | ✅ Fully supported |
| `OpenAIResponsesProvider` (Responses API) | `/api/v2/apps/protocols/compatible-mode/v1/responses` | ✅ Fully supported |

> **Which API should I use?**  Both work with Qwen models on Alibaba Cloud.  The Chat Completions
> provider (`OpenAIProvider`) is the **recommended and default choice** — it is the path used by
> `provider: alibaba` and `create_alibaba_client()`, and it keeps the stable, widely-adopted
> OpenAI-compatible interface.  Use `OpenAIResponsesProvider` only if you specifically need
> stateful multi-turn via `previous_response_id`, built-in tools such as web search / code
> interpreter, or `reasoning_effort` for OpenAI reasoning models.

---

## Prerequisites

1. Create an [Alibaba Cloud Model Studio account](https://www.alibabacloud.com/product/modelstudio).
2. Obtain an API key from the [API key page](https://www.alibabacloud.com/help/en/model-studio/get-api-key).
3. Install core-lib with OpenAI extras:
   ```bash
   uv add core-lib[openai]
   # or
   pip install openai
   ```

---

## Regions and Base URLs

### Chat Completions (`OpenAIProvider`)

| Region | `base_url` |
|---|---|
| Singapore / Virginia (International) | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| China (Beijing) | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

### Responses API (`OpenAIResponsesProvider`)

| Region | `base_url` |
|---|---|
| Singapore / Virginia (International) | `https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1` |
| China (Beijing) | `https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1` |

Use the **international** endpoint (`dashscope-intl`) if your services run outside mainland China.

---

## Environment Variables

### For the Responses API (`OpenAIResponsesProvider`)

```env
# Required
OPENAI_API_KEY=sk-your-dashscope-api-key

# Required: Responses API endpoint
OPENAI_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1

# Required: set the model
OPENAI_RESPONSES_MODEL=qwen-plus

# Optional
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2048
OPENAI_THINKING_ENABLED=false    # set to 'true' for Qwen3 thinking mode
OPENAI_REASONING_EFFORT=medium   # low / medium / high (OpenAI reasoning models only)
```

### For the Chat Completions API (`OpenAIProvider`)

```env
# Required
OPENAI_API_KEY=sk-your-dashscope-api-key

# Required: Chat Completions endpoint
OPENAI_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# Required: set the model
OPENAI_MODEL=qwen-plus

# Optional
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2048
```

> You can also use `DASHSCOPE_API_KEY` — the factory helpers for Alibaba
> (`create_alibaba_client`, `LLMFactory.alibaba`) will read it automatically.

---

## Quick Start

### Chat Completions — recommended (via `create_alibaba_client`)

```python
from core_lib.llm import create_alibaba_client

# Reads DASHSCOPE_API_KEY (or OPENAI_API_KEY) from environment
client = create_alibaba_client(model="qwen-plus")

response = client.chat("What is the capital of France?")
print(response["content"])      # "Paris..."
```

### Responses API (via explicit config)

```python
import os
from core_lib.llm import LLMClient
from core_lib.llm.providers.openai_responses_provider import OpenAIResponsesConfig

config = OpenAIResponsesConfig.for_alibaba(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    model="qwen-plus",
    temperature=0.7,
)
client = LLMClient(config)
response = client.chat("Hello, who are you?")
print(response["content"])
print(response["response_id"])  # save for stateful follow-up
```

### Chat Completions API (via `OpenAIProvider`) — explicit config

```python
import os
from core_lib.llm import LLMClient
from core_lib.llm.providers.openai_provider import OpenAIConfig

config = OpenAIConfig(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
)
client = LLMClient(config)
response = client.chat("Hello, who are you?")
print(response["content"])
```

### From environment variables

```python
from core_lib.llm import create_client_from_env

# Set LLM_PROVIDER=alibaba (or openai), DASHSCOPE_API_KEY, OPENAI_MODEL
client = create_client_from_env("alibaba")
response = client.chat("Hello")
print(response["content"])
```

---

## Using llm_providers.yaml (Multi-Provider / Fallback Setup)

For production deployments with fallback, configure providers via a YAML file instead of individual environment variables. This is the recommended approach when mixing Alibaba models with other providers.

### Installation

Point the registry at your file:

```bash
export LLM_PROVIDERS_FILE=llm_providers.yaml
export DASHSCOPE_API_KEY=sk-your-key
```

### Alibaba only

```yaml
# llm_providers.yaml
providers:
  - provider: alibaba           # alias for openai (Chat Completions) + DashScope endpoint
    api_key: ${DASHSCOPE_API_KEY}
    model: qwen-plus
    priority: 1
    tier: standard
```

```python
from core_lib.llm import ProviderRegistry

registry = ProviderRegistry.from_env()
client = registry.get_client()
response = client.chat("Hello!")
print(response["content"])
```

### Alibaba with Gemini fallback

```yaml
# llm_providers.yaml
providers:
  - provider: alibaba
    api_key: ${DASHSCOPE_API_KEY}
    model: ${ALIBABA_MODEL:-qwen-plus}
    priority: 1
    tier: standard

  - provider: gemini
    api_key: ${GEMINI_API_KEY}
    model: ${GEMINI_MODEL:-gemini-2.0-flash}
    priority: 2
    tier: standard
```

### OpenAI Responses API (generic endpoint)

```yaml
# llm_providers.yaml
providers:
  - provider: openai-responses
    api_key: ${OPENAI_API_KEY}
    model: gpt-4.1
    priority: 1
    tier: high
    reasoning_effort: medium      # "low" | "medium" | "high"
```

### Thinking / chain-of-thought

```yaml
providers:
  - provider: alibaba
    api_key: ${DASHSCOPE_API_KEY}
    model: qwen3-max
    thinking: true                # enables Qwen extended thinking
    thinking_budget: 4000         # max tokens for the thinking step (optional)
    priority: 1
    tier: high
```

You can also pass the budget as the value of `thinking` directly (integer → `thinking_budget`, enables thinking automatically):

```yaml
  - provider: alibaba
    api_key: ${DASHSCOPE_API_KEY}
    model: qwen3-max
    thinking: 4000                # shorthand: budget=4000, thinking=true
```

### China region (Beijing DashScope endpoint)

```yaml
providers:
  - provider: openai
    api_key: ${DASHSCOPE_API_KEY}
    model: qwen-plus
    host: https://dashscope.aliyuncs.com/compatible-mode/v1
    priority: 1
```

> **Behind the scenes**: `provider: alibaba` is a convenience alias that automatically sets
> `provider: openai` and the international DashScope Chat Completions endpoint
> (`dashscope-intl.aliyuncs.com/compatible-mode/v1`). For the China region, use
> `provider: openai` and specify `host` explicitly, as shown above.

### Provider Config Fields for Alibaba / OpenAI Responses

| Field | Description |
|---|---|
| `provider` | `alibaba` (auto-sets DashScope international Chat Completions URL) or `openai` or `openai-responses` |
| `api_key` | DashScope API key — or leave unset and export `DASHSCOPE_API_KEY` |
| `model` | Qwen model name (e.g. `qwen-plus`, `qwen3-max`) or OpenAI model |
| `host` | Override endpoint — required for China region |
| `reasoning_effort` | `"low"` / `"medium"` / `"high"` — for OpenAI reasoning models |
| `thinking` | `true` / `false` — enables Qwen chain-of-thought mode |
| `thinking_budget` | Integer token budget for the thinking step (e.g. `4000`); implies `thinking: true` when passed as the `thinking` value |
| `temperature` | Sampling temperature (default: `0.7`) |
| `max_tokens` | Maximum output tokens |
| `priority` | Lower = higher priority in fallback chain |
| `tier` | `"low"` / `"standard"` / `"high"` — for intelligence-level routing |

---

## Supported Qwen Models

The following models can be used with the Chat Completions endpoint. Pass any of these as the
`model` parameter or `OPENAI_MODEL` env var:

### Qwen3 / Qwen3.5 (Latest)

| Model | Description |
|---|---|
| `qwen3-max` | Most capable Qwen3 model |
| `qwen3-max-2026-01-23` | Pinned snapshot |
| `qwen3.5-plus` | Balanced Qwen3.5 model |
| `qwen3.5-plus-2026-02-15` | Pinned snapshot |
| `qwen3.5-397b-a17b` | Large MoE model |
| `qwen3.5-flash` | Fast, cost-effective |
| `qwen3.5-flash-2026-02-23` | Pinned snapshot |
| `qwen3.5-122b-a10b` | Dense 122B model |
| `qwen3.5-27b` | 27B parameter model |
| `qwen3.5-35b-a3b` | 35B MoE model |

### Older Stable Models

| Model | Description |
|---|---|
| `qwen-max` | Most capable commercial Qwen |
| `qwen-plus` | Balanced capability and cost |
| `qwen-turbo` | Fastest, lowest cost |
| `qwen-long` | Extended context window |

For a full list see the
[Alibaba Cloud model catalog](https://www.alibabacloud.com/help/en/model-studio/getting-started/models).

---

## Feature Support

### `OpenAIResponsesProvider` (Responses API)

| Feature | Alibaba Support | Notes |
|---|---|---|
| Text generation | ✅ | Fully compatible |
| System message | ✅ | Injected as first `input` item with `role: "system"` |
| Multi-turn conversation (messages array) | ✅ | Pass full messages array |
| Stateful multi-turn (`previous_response_id`) | ✅ | Server manages context; pass `response["response_id"]` back |
| Structured output (`json_schema`) | ✅ | Via `text.format` parameter |
| Tool / function calling | ✅ | Standard OpenAI function schema |
| Web search grounding | ✅ | Adds `web_search_preview` built-in tool |
| Thinking / chain-of-thought | ✅ | Alibaba: `enable_thinking=True` via `extra_body` |
| `instructions` parameter | ❌ | Not supported by Alibaba; provider converts to `system` message automatically |

### `OpenAIProvider` (Chat Completions API)

| Feature | Alibaba Support | Notes |
|---|---|---|
| Text generation | ✅ | Fully compatible |
| System message | ✅ | Standard `system` role |
| Multi-turn conversation | ✅ | Pass full `messages` array |
| Structured output (`response_format`) | ✅ | `json_schema` supported |
| Tool / function calling | ✅ | Standard OpenAI function schema |
| `use_search_grounding` | ⚠️ | Maps to `enable_search=true` on Alibaba; current provider injects `web_search` tool which may not align — test before using |
| Thinking / chain-of-thought | ⚠️ | Alibaba uses `enable_thinking` extra parameter; not automatically forwarded — use the Responses provider instead |

---

## Thinking / Reasoning Mode

Qwen3 and Qwen3.5 models support a "thinking" mode that produces visible chain-of-thought
before the final answer.

### Via `OpenAIResponsesProvider` (recommended)

Pass `thinking_enabled=True` to `create_alibaba_client` or set it in the config.  The Responses
provider automatically sends `enable_thinking=True` via `extra_body` for Alibaba endpoints.

```python
from core_lib.llm import create_alibaba_client

client = create_alibaba_client(
    api_key="sk-your-key",
    model="qwen3-max",
    thinking_enabled=True,
)
response = client.chat("Explain the Monty Hall problem step by step.")
print(response["content"])
```

### Via `OpenAIProvider` (Chat Completions) — manual workaround

The Chat Completions provider does not forward `enable_thinking` automatically.  Call the raw
SDK directly when you need this feature with the Chat Completions endpoint:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-key",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-max",
    messages=[{"role": "user", "content": "Explain the Monty Hall problem."}],
    extra_body={"enable_thinking": True},
)
print(completion.choices[0].message.content)
```

---

## Multi-Turn Conversation

### Stateful multi-turn via `previous_response_id` (Responses API)

The server automatically maintains context.  You only send the new message each turn.

```python
from core_lib.llm import create_alibaba_client

client = create_alibaba_client(api_key="sk-your-key", model="qwen-plus")

# First turn
resp = client.chat("My name is Alice.")
prev_id = resp["response_id"]

# Second turn — server recalls the context automatically
from core_lib.llm.providers.openai_responses_provider import OpenAIResponsesConfig
client.config.previous_response_id = prev_id
resp2 = client.chat("What is my name?")
print(resp2["content"])  # Should say "Alice"
```

### Stateless multi-turn via messages array (both APIs)

```python
from core_lib.llm import create_alibaba_client

client = create_alibaba_client(api_key="sk-your-key", model="qwen-plus")

messages = [{"role": "user", "content": "My name is Alice."}]
resp = client.chat(messages=messages)

messages += [
    {"role": "assistant", "content": resp["content"]},
    {"role": "user", "content": "What is my name?"},
]
resp2 = client.chat(messages=messages)
print(resp2["content"])  # Should recall "Alice"
```

---

## Structured Output Example

```python
from pydantic import BaseModel
from core_lib.llm import create_alibaba_client

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

client = create_alibaba_client(api_key="sk-your-key", model="qwen-plus")

resp = client.chat(
    "Summarize: Python is a versatile language used in web, data science, and automation.",
    structured_output=Summary,
)
print(resp["content"])    # dict with title, key_points, sentiment
print(resp["structured"]) # True
```

The Responses API uses `text.format` with a JSON schema — the provider handles the conversion
from your Pydantic model automatically.

---

## Tool Calling Example

```python
from core_lib.llm import create_alibaba_client

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]

client = create_alibaba_client(api_key="sk-your-key", model="qwen-plus")
resp = client.chat(
    messages=[{"role": "user", "content": "What is the weather in Singapore?"}],
    tools=tools,
)
print(resp["tool_calls"])
# [{"id": "call_...", "type": "function",
#   "function": {"name": "get_weather", "arguments": '{"city": "Singapore"}'}}]
```

---

## China Region Setup

If your infrastructure is in mainland China, pass `region="china"` to the factory:

```python
from core_lib.llm import create_alibaba_client

client = create_alibaba_client(
    api_key="sk-your-china-key",
    model="qwen-plus",
    region="china",  # Uses dashscope.aliyuncs.com (no "intl")
)
```

Or set the environment variable for Chat Completions:
```env
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

The API key for mainland China accounts is different from international accounts.
Obtain it at [https://bailian.console.aliyun.com/](https://bailian.console.aliyun.com/).

---

## Responses API vs Chat Completions — Key Differences

| Aspect | `OpenAIResponsesProvider` | `OpenAIProvider` |
|---|---|---|
| SDK method | `client.responses.create()` | `client.chat.completions.create()` |
| System message | Injected as first `input` item | `{"role": "system", ...}` in `messages` |
| Context management | Stateful via `previous_response_id` | Stateless — send full history each turn |
| Structured output param | `text.format` (json_schema) | `response_format` (json_schema) |
| Web search | `web_search_preview` built-in tool | `enable_search` extra param (not forwarded) |
| Thinking mode | Auto-detected (Alibaba → `extra_body`, OpenAI → `reasoning`) | Not forwarded — manual workaround required |
| Response field | `response["response_id"]` for next turn | N/A |
| Usage field keys | `input_tokens` / `output_tokens` (normalised to `prompt_tokens` / `completion_tokens`) | `prompt_tokens` / `completion_tokens` |

Both providers return the same unified response shape:
```python
{
    "content": str | dict,   # text or parsed structured dict
    "structured": bool,
    "tool_calls": list,
    "usage": dict,           # always uses prompt_tokens / completion_tokens keys
    "response_id": str,      # only present in OpenAIResponsesProvider responses
    "error": str,            # only present on failure
}
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `AuthenticationError` | Wrong API key or wrong region | Verify `DASHSCOPE_API_KEY`; international key for `dashscope-intl`, China key for `dashscope` |
| `NotFoundError` on model | Model name typo or model not available in your region | Check the [model list](https://www.alibabacloud.com/help/en/model-studio/getting-started/models) |
| `404 Not Found` on `/responses` | Using `OpenAIProvider` (Chat Completions) with a Responses API URL | Use `create_alibaba_client` or `OpenAIResponsesConfig.for_alibaba(...)` which sets the correct endpoint |
| `404 Not Found` on `/chat/completions` | Using `OpenAIResponsesProvider` with a Chat Completions URL | Ensure `base_url` ends in `.../compatible-mode/v1` for Chat Completions, or use the Responses endpoint |
| Empty `tool_calls` returned | Tool calling not triggered | Rephrase the prompt to make tool use more explicit |
| `structured` is `False` even with `structured_output` | Model returned non-JSON | Add `"Respond only with valid JSON."` to the system message |
| `ConnectionError` / timeout | Wrong `base_url` | Ensure no trailing slash and correct regional domain |
| Thinking tokens appear in output text | `thinking_enabled=True` with a non-thinking model | Use `qwen3-max`, `qwen3.5-plus`, or another Qwen3/3.5 thinking-capable model |
