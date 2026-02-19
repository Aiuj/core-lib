# LLM Quick Reference

## Installation

The LLM module is included in `core-lib` and requires Python 3.12+.

Runtime dependencies (installed by the package):
- `google-genai` (official Google GenAI Python SDK for Gemini)
- `ollama` (official Ollama Python client)
- `pydantic` (for structured outputs)

No LangChain dependency is required for LLM calls.

## Quick Start

```python
from core_lib.llm import create_ollama_client, create_gemini_client

# Local Ollama (single turn)
client = create_ollama_client(model="llama3.2", temperature=0.7)
resp = client.chat("Hello, how are you?")
print(resp["content"])  # plain text

# Google Gemini (single turn)
client = create_gemini_client(api_key="your-key", model="gemini-1.5-flash")
resp = client.chat("Explain quantum computing")
print(resp["content"])  # plain text

# Google Vertex AI (single turn via ADC)
client = create_gemini_client(
    model="gemini-1.5-pro",
    project="my-gcp-project",
    location="europe-west9",
)
resp = client.chat("Summarize the last earnings call")
print(resp["content"])
```

## Key Features

✅ Unified interface across providers  
✅ Environment configuration (sensible defaults, .from_env helpers)  
✅ Native tool/function-calling structures  
✅ Structured outputs with Pydantic schemas (response_schema)  
✅ Single-turn and multi-turn support  
✅ Thinking mode for Gemini 2.5 models  
✅ Optional Google Search grounding (Gemini)  
✅ **Model-specific rate limiting** (5-100 RPM per model)  
✅ **Automatic retry with exponential backoff** (network errors, rate limits)  
✅ **Circuit breaker pattern** with graceful degradation  
✅ Graceful error handling

## Response Format

```python
{
    "content": str | dict,    # text for normal calls, dict for structured outputs
    "structured": bool,       # True if structured_output was requested
    "tool_calls": list,       # provider-reported function calls (if any)
    "usage": dict,            # usage metadata (if available)
    "error": str | None       # present on failure
}
```

## Configuration Methods

1) Direct instantiation
```python
client = create_ollama_client(model="llama3.2", temperature=0.8)
```

2) From environment
```python
from core_lib.llm import create_client_from_env
client = create_client_from_env("ollama")
```

3) Using config objects
```python
from core_lib.llm import LLMClient, OllamaConfig
config = OllamaConfig(model="llama3.2", temperature=0.8)
client = LLMClient(config)
```

### Environment variables
- Common:
    - `LLM_PROVIDER` = `gemini` | `ollama`
- Gemini (Developer API):
    - `GEMINI_API_KEY` (or `GOOGLE_GENAI_API_KEY`)
    - `GEMINI_MODEL` (default: `gemini-1.5-flash`)
    - `GEMINI_TEMPERATURE` (default: `0.1`)
    - `GEMINI_MAX_TOKENS`
    - `GEMINI_THINKING_ENABLED` = `true|false`
    - `GEMINI_BASE_URL` (default: Google endpoint)
- Gemini (Vertex AI via ADC):
    - `GOOGLE_CLOUD_PROJECT` (or `GOOGLE_PROJECT_ID`)
    - `GOOGLE_CLOUD_LOCATION` (or `GOOGLE_CLOUD_REGION`)
    - `GOOGLE_APPLICATION_CREDENTIALS` (optional, path to service account JSON)
    - `GEMINI_SERVICE_ACCOUNT_FILE` (optional, alias for service account file)
- Ollama:
    - `OLLAMA_MODEL` (default: `qwen3:1.7b`)
    - `OLLAMA_TEMPERATURE` (default: `0.1`)
    - `OLLAMA_MAX_TOKENS`
    - `OLLAMA_THINKING_ENABLED` = `true|false`
    - `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
    - `OLLAMA_TIMEOUT`, `OLLAMA_NUM_CTX`, `OLLAMA_NUM_PREDICT`, `OLLAMA_REPEAT_PENALTY`, `OLLAMA_TOP_K`, `OLLAMA_TOP_P`

## Common Patterns

### Structured Output (Pydantic)
```python
from pydantic import BaseModel

class Weather(BaseModel):
        location: str
        temperature: float

# Pass the Pydantic class; providers enforce schema natively when supported
resp = client.chat("Weather in Paris?", structured_output=Weather)
# resp["content"] is a dict (Pydantic exported via model_dump)
print(resp["content"]["location"], resp["content"]["temperature"])
```

### Tool Calling (OpenAI-style functions)
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

resp = client.chat("What's the weather in Tokyo?", tools=tools)
```

### Google Search Grounding (Gemini only)
```python
from core_lib.llm import create_gemini_client
client = create_gemini_client(api_key="your-key", model="gemini-1.5-flash")
resp = client.chat("What are the latest Mars mission updates?", use_search_grounding=True)
print(resp["content"])
```
Notes: This flag is forwarded to all providers. Providers that don't support grounding will ignore it.

### System message and multi-turn
```python
messages = [
    {"role": "system", "content": "You are concise and helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Tell me about Python"}
]
resp = client.chat(messages)
```

## Provider behavior

- Gemini (google-genai):
    - Single turn: uses `models.generate_content()`
    - Multi-turn: uses `chats.create(...).send_message()`
    - Structured: passes your Pydantic class via `response_schema`; returns `dict` (from BaseModel.model_dump)
    - Thinking (2.5-series): if `thinking_enabled=True`, sets `include_thoughts`; if disabled, sets `thinking_budget=0` for non‑pro models
    - Tools: passes the provided functions list through
    - Search grounding: when `use_search_grounding=True`, enables Google Search tool and tool_config per official docs
    - **Rate limiting**: Model-specific RPM limits (Gemini 2.5 Pro: 5 RPM, Flash: 10 RPM, Flash-Lite: 15 RPM, Gemma 3: 30 RPM, Embedding: 100 RPM)
    - **Retry logic**: Automatic retry on rate limits (429), server errors (500/503 including "model is overloaded"), network failures with exponential backoff (3 retries, 1-30s delays with jitter)

- Ollama (ollama):
    - Uses `ollama.chat()` with OpenAI-style messages
    - Structured: uses `format='json'`; validates with Pydantic when provided; returns `dict`
    - Tools: passes functions through (model support varies)
    - Search grounding: ignored (not supported)
    - **Rate limiting**: None (local server)
    - **Retry logic**: None (local server)

## Provider Comparison

| Feature | Ollama | Gemini |
|---------|--------|--------|
| Cost | Free (local) | Paid API |
| Privacy | Local | Cloud |
| Setup | Run Ollama server | API key |
| Models | Open-source | Google models |
| Tools | Model-dependent | Native support |
| Structured Output | JSON + Pydantic validate | response_schema + parsed |
| Thinking | N/A | 2.5-series (configurable) |
| **Rate Limiting** | **None** | **Model-specific (5-100 RPM)** |
| **Retry Logic** | **None** | **Auto-retry with backoff** |

## Best Practices

1. Prefer environment-based configuration for deployments
2. Check `response["error"]` and handle gracefully
3. Verify model support for tools/structured output
4. Monitor usage with cloud providers
5. Keep prompts concise to stay within context
6. **Rate limiting is handled automatically** - no need to implement client-side throttling
7. **Retries are transparent** - failed requests automatically retry with exponential backoff
8. **For high-volume applications**, consider model selection based on RPM limits (Flash > Pro for rate)

## Resilience Features

### Automatic Rate Limiting (Gemini)
- **Per-model RPM limits**: Automatically enforced based on model type
- **Transparent throttling**: Requests are delayed to respect API limits
- **Fail-soft design**: Rate limiter failures don't block requests

### Automatic Retry Logic (Gemini)
- **Exponential backoff**: 1s → 2s → 4s delays with 50% jitter
- **Retryable errors**: Network failures, rate limits (429), server errors (500/503 including "model is overloaded"), timeouts
- **Max 3 retries**: Prevents infinite loops while providing resilience
- **Clean error logging**: Retryable errors log warnings during retry, only non-retryable errors show full tracebacks
- **Error preservation**: Final error is returned if all retries fail

### Circuit Breaker Pattern
- **Graceful degradation**: Partial failures don't break the entire system
- **Comprehensive logging**: All retry attempts and rate limit events are logged
- **Transparent operation**: No code changes required to benefit from resilience

## Advanced Features

### Vertex AI Context Caching

Reduce costs and latency for large, repetitive contexts (e.g., long documents, rule sets) using Context Caching.

#### Implicit Caching (Automatic)
Vertex AI automatically caches context for **all users** at no extra cost if:
1. The prompt is **> 2048 tokens**.
2. You place the large, static content (documents, instructions) at the **beginning** of the prompt.
3. Subsequent requests share the exact same prefix.

**Savings:** ~90% cost reduction on cached input tokens.
**Action:** Structure your prompts so static info (documents, system prompts) comes first.

#### Explicit Caching (Manual)
For guaranteed caching of specific large contexts (e.g., a referenced RFx document):

1. **Create the cache** (using `google-genai` SDK directly):
```python
from google import genai
from google.genai.types import CreateCachedContentConfig

# Initialize standard client
client = genai.Client(vertexai=True, project="my-project", location="us-central1")

# Create cache
cache = client.caches.create(
    model="gemini-1.5-pro-001",
    config=CreateCachedContentConfig(
        contents=[...], # Your large content content list
        ttl="3600s",
        display_name="my-rfx-cache" 
    )
)
cache_name = cache.name # e.g. "projects/.../cachedContents/123..."
print(f"Created cache: {cache_name}")
```

2. **Use the cache** with `core-lib`:
```python
client = create_gemini_client(
    model="gemini-1.5-pro-001", 
    project="my-project", 
    location="us-central1"
)

# Pass the cache name to chat() via cached_content
response = client.chat(
    messages=[{"role": "user", "content": "Analyze section 4 based on the cached RFx."}],
    cached_content=cache_name
)
```

**Note:** When using `cached_content`, system instructions and tools configured in the cache are used. Providing a new `system_message` in `chat()` when using a cache may be ignored or cause warnings, as instructions should typically be baked into the cache.

## References
- Google GenAI Python SDK: https://googleapis.github.io/python-genai/
- Gemini Structured Output: https://ai.google.dev/gemini-api/docs/structured-output
- Gemini Thinking Mode: https://ai.google.dev/gemini-api/docs/thinking
- Gemini Text Generation: https://ai.google.dev/gemini-api/docs/text-generation
