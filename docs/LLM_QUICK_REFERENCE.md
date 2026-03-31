# LLM Quick Reference

Cheat sheet for core-lib LLM usage.  
Full docs: [llm.md](llm.md) (client usage) · [FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md) (provider config & fallback)

---

## Single-Provider Quick Start

```python
from core_lib.llm import create_llm_client

client = create_llm_client()                                          # auto-detect
client = create_llm_client(provider="gemini", model="gemini-2.5-flash")
client = create_llm_client(provider="ollama", model="qwen3.5:4b")
client = create_llm_client(provider="openai", model="gpt-4o")

response = client.chat("Hello!")
print(response["content"])
```

## Multi-Provider Fallback (Production)

```python
from core_lib.llm import FallbackLLMClient

# Load from llm_providers.yaml (set LLM_PROVIDERS_FILE env var)
client = FallbackLLMClient.from_env()

# With usage-based routing
client = FallbackLLMClient.from_env(usage="rag")
client = FallbackLLMClient.from_env(usage="ocr", intelligence_level=7)

response = client.chat("Summarize these documents")
print(f"Provider: {client.last_used_provider}, fallback: {client.last_was_fallback}")
```

## chat() — Full Signature

```python
# LLMClient
response = client.chat(
    messages,                     # str or list of {"role": ..., "content": ...}
    tools=None,                   # OpenAI-format function schemas
    structured_output=MyModel,    # Pydantic class for structured JSON output
    system_message="You are...",  # System prompt
    use_search_grounding=False,   # Gemini: Google Search grounding
    thinking_enabled=None,        # Override thinking mode per call
)

# FallbackLLMClient adds:
response = client.chat(
    messages,
    intelligence_level=5,         # Override provider filter for this call
    usage="vision",               # Override usage routing for this call
    return_fallback_result=False, # Return FallbackResult instead of dict
)
```

## Response Format

```python
{
    "content": str | dict,    # text, or dict when structured_output used
    "structured": bool,
    "tool_calls": list,
    "usage": dict,
    "error": str | None,
    "response_id": str,       # OpenAI Responses API only (stateful multi-turn)
}
```

## Usage-Based Routing

Tag providers in `llm_providers.yaml` to route workloads:

```yaml
providers:
  - provider: ollama
    model: qwen3.5:4b             # Fully multimodal (text + image)
    priority: 1
    usage: [rag, chat, vision, ocr, translate, classify, extract]

  - provider: gemini
    model: gemini-2.5-flash-lite
    api_key: ${GEMINI_API_KEY}
    priority: 2
    usage: [rag, chat, vision]    # vision but NOT ocr (less document-accurate)

  - provider: gemini
    model: gemini-2.5-flash
    api_key: ${GEMINI_API_KEY}
    priority: 50
    usage: [vision, ocr]          # Document-grade OCR accuracy

  - provider: gemini
    model: gemini-2.5-pro         # No usage tag = general-purpose fallback
    api_key: ${GEMINI_API_KEY_HIGH}
    priority: 100
```

Route at client creation or per call:
```python
# Default routing at creation
rag_client = FallbackLLMClient.from_env(usage="rag")

# Override per call
rag_client.chat("Describe this image", usage="vision")
rag_client.chat("Extract text from document", usage="ocr")
```

**Known usage tags:**

| Tag | Workload |
|-----|----------|
| `rag` | Q&A retrieval-augmented generation |
| `chat` | Conversational / chat endpoints |
| `translation` | Cross-language translation |
| `quality_analysis` | Search quality / answer grounding |
| `query_expansion` | Query rewriting for retrieval |
| `classify` | Document classification |
| `extract` | Information extraction |
| `agent` | LangGraph / agentic reasoning |
| `vision` | Image understanding |
| `ocr` | Optical character recognition |

Providers **without** a `usage` tag match any requested usage (general-purpose fallback).

## Common Patterns

### Structured Output
```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]

resp = client.chat("Summarize this text: ...", structured_output=Summary)
data = resp["content"]   # dict (model_dump())
print(data["title"])
```

### Multi-Turn Conversation
```python
messages = [
    {"role": "user", "content": "I'm researching climate topics."},
    {"role": "assistant", "content": "Great! What would you like to know?"},
    {"role": "user", "content": "Explain carbon capture briefly."},
]
response = client.chat(messages, system_message="You are a science expert.")
```

### Vision / Image Input
```python
import base64

with open("scan.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

messages = [{"role": "user", "content": [
    {"type": "text", "text": "Extract all text from this document"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
]}]

# qwen3.5 is fully multimodal (0.8b to 122b, all sizes support vision+OCR)
client = create_llm_client(provider="ollama", model="qwen3.5:4b")
response = client.chat(messages)
```

### Thinking / Reasoning Mode
```python
client = create_llm_client(provider="ollama", model="qwen3.5:4b", thinking_enabled=True)
response = client.chat("Solve this complex problem step by step: ...")
```

### Stateful Multi-Turn (OpenAI Responses API)
```python
from core_lib.llm import create_openai_responses_client

client = create_openai_responses_client(model="gpt-4.1")
resp1 = client.chat("My name is Alice.")
client.config.previous_response_id = resp1["response_id"]
resp2 = client.chat("What is my name?")   # "Alice"
```

### Alibaba Cloud / Qwen
```python
from core_lib.llm import create_alibaba_client

client = create_alibaba_client(model="qwen3-max", thinking_enabled=True)
response = client.chat("Solve: 3x + 7 = 22")
```

### Fallback with Metadata
```python
result = client.chat("Hello", return_fallback_result=True)
if result.error:
    print(f"All providers failed: {result.error}")
else:
    print(f"{result.content} | via {result.provider} (fallback={result.was_fallback})")
```

---

## llm_providers.yaml — Full Field Reference

```yaml
providers:
  - provider: gemini              # gemini | openai | openai-responses | alibaba |
                                  # azure-openai | openrouter | ollama
    model: gemini-2.5-flash
    api_key: ${GEMINI_API_KEY}    # env var substitution supported
    host: ""                      # custom endpoint URL (Ollama, Azure, etc.)
    temperature: 0.1
    max_tokens: 4096
    priority: 1                   # lower = higher priority
    enabled: true
    tier: standard                # low | standard | high (label only)
    min_intelligence_level: 0     # 0-10 scale
    max_intelligence_level: 6
    usage: [rag, chat]            # list of tags, or omit for general-purpose
    thinking_enabled: false       # qwen3/3.5, gemini 2.5
    reasoning_effort: medium      # OpenAI o-series: low | medium | high
```

---

## Environment Variables

### Provider Detection
| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | Explicit provider: `gemini`, `openai`, `openai-responses`, `openrouter`, `azure`, `ollama`, `alibaba` |
| `LLM_PROVIDERS_FILE` | Path to `llm_providers.yaml` |
| `LLM_PROVIDERS` | JSON array of provider configs (fallback) |

### Gemini
| Variable | Default |
|----------|---------|
| `GEMINI_API_KEY` | — |
| `GEMINI_MODEL` | `gemini-1.5-flash` |
| `GEMINI_TEMPERATURE` | `0.1` |

### OpenAI Responses / Alibaba
| Variable | Default |
|----------|---------|
| `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` | — |
| `OPENAI_RESPONSES_MODEL` | `gpt-4.1` |
| `OPENAI_THINKING_ENABLED` | — |
| `OPENAI_REASONING_EFFORT` | `medium` |

### Ollama
| Variable | Default |
|----------|---------|
| `OLLAMA_MODEL` | `qwen3:1.7b` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `OLLAMA_TEMPERATURE` | `0.1` |
| `OLLAMA_THINKING_ENABLED` | — |

---

## Provider Comparison

| Feature | Ollama | Gemini | OpenAI Responses | Azure OpenAI | OpenRouter | Alibaba |
|---------|--------|--------|-----------------|--------------|------------|---------|
| Cost | Free (local) | Paid | Paid | Paid | Paid | Paid |
| Structured output | JSON + Pydantic | `response_schema` | `text.format` | `response_format` | `response_format` | `text.format` |
| Tool calling | Model-dependent | Native | Native | Native | Model-dependent | Native |
| Vision / multimodal | qwen3.5, qwen2.5vl | All models | GPT-4o+ | GPT-4o+ | Model-dependent | Qwen-VL |
| Thinking / reasoning | qwen3, qwen3.5 | 2.5-series | o-series | Not forwarded | Model-dependent | Qwen3+ |
| Web search | ❌ | ✅ Google Search | ✅ `web_search_preview` | ❌ | Model-dependent | ✅ |
| Stateful multi-turn | ❌ | ❌ | ✅ `response_id` | ❌ | ❌ | ✅ |
| Rate limiting | None | Auto (model RPM) | None | None | None | None |
| Auto-retry | None | Exponential backoff | None | None | None | None |

---

## See Also

- [llm.md](llm.md) — Full usage guide (factories, patterns, env vars, provider details)
- [FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md) — Multi-provider config, usage routing, health tracking
- [PROVIDER_REGISTRY_QUICK_REFERENCE.md](PROVIDER_REGISTRY_QUICK_REFERENCE.md) — ProviderRegistry API
