# Fallback LLM Client Guide

The `FallbackLLMClient` provides transparent automatic failover between multiple LLM providers. When a provider fails (rate limit, timeout, server error), it automatically switches to backup providers and tracks health status for intelligent routing.

## Quick Start

```python
from core_lib.llm import FallbackLLMClient, create_fallback_llm_client

# Simplest: load from environment/config file
client = create_fallback_llm_client()

# Use just like a normal LLMClient
response = client.chat("What is the capital of France?")
print(response["content"])

# Check which provider was used
print(f"Provider: {client.last_used_provider}")
print(f"Was fallback: {client.last_was_fallback}")
```

## Configuration

### From Environment

The client loads providers from (in order of priority):

1. **Config file** via `LLM_PROVIDERS_FILE` environment variable
2. **JSON array** in `LLM_PROVIDERS` environment variable  
3. **Individual env vars** (`GEMINI_API_KEY`, `OPENAI_API_KEY`, etc.)

```yaml
# llm_providers.yaml (set LLM_PROVIDERS_FILE=llm_providers.yaml)
providers:
  - provider: gemini
    api_key: ${GEMINI_API_KEY}  # Environment variable substitution
    model: gemini-2.0-flash
    priority: 1  # Lower = higher priority

  - provider: openai
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o-mini
    priority: 2

  - provider: ollama
    host: ${OLLAMA_HOST:-http://localhost:11434}  # With default
    model: llama3.2
    priority: 3
```

### From Code

```python
# With explicit provider configurations
client = FallbackLLMClient.from_config([
    {"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash", "priority": 1},
    {"provider": "openai", "api_key": "...", "model": "gpt-4o-mini", "priority": 2},
    {"provider": "ollama", "host": "http://localhost:11434", "model": "llama3.2", "priority": 3},
])

# Using an existing ProviderRegistry
from core_lib.llm import ProviderRegistry
registry = ProviderRegistry.from_env()
client = FallbackLLMClient.from_registry(registry)
```

## Features

### Automatic Failover

When a provider fails, the client automatically tries the next provider:

```python
client = FallbackLLMClient.from_config([
    {"provider": "gemini", "api_key": "...", "priority": 1},
    {"provider": "openai", "api_key": "...", "priority": 2},
])

# If Gemini is down, automatically uses OpenAI
response = client.chat("Hello!")

# Check what happened
if client.last_was_fallback:
    print(f"Primary failed, used {client.last_used_provider}")
```

### Health Tracking

Providers that fail are marked "unhealthy" and skipped for subsequent requests until they recover:

```python
# First request - Gemini fails, OpenAI succeeds
response1 = client.chat("Question 1")  # Tries Gemini -> fails -> tries OpenAI

# Second request - Gemini is still marked unhealthy
response2 = client.chat("Question 2")  # Goes directly to OpenAI (Gemini skipped)

# After TTL expires (default: 5 minutes), Gemini is tried again
```

Error-specific recovery times:
- **Rate limits**: 5 minutes
- **Quota exceeded**: 1 hour
- **Timeouts**: 1 minute
- **Server errors**: 2 minutes
- **Auth errors**: 1 hour

### Retries Before Fallback

Configure retries per provider before moving to the next:

```python
client = FallbackLLMClient.from_config(
    providers=[...],
    max_retries=3,  # Try each provider 3 times before moving on
)
```

### Intelligence Level Filtering

Filter providers by intelligence level for cost optimization:

```python
# Configure providers with intelligence level ranges
client = FallbackLLMClient.from_config([
    {
        "provider": "gemini",
        "model": "gemini-2.0-flash",
        "min_intelligence_level": 0,
        "max_intelligence_level": 6,
        "tier": "low",
    },
    {
        "provider": "gemini",
        "model": "gemini-2.0-flash-thinking",
        "min_intelligence_level": 7,
        "max_intelligence_level": 10,
        "tier": "high",
    },
])

# Use lower-tier model for simple queries
response = client.chat("What is 2+2?", intelligence_level=3)

# Use higher-tier model for complex queries
response = client.chat("Explain quantum entanglement", intelligence_level=8)
```

### Rich Metadata

Get detailed information about the request:

```python
# Standard response includes metadata
response = client.chat("Hello")
metadata = response.get("_fallback_metadata", {})
print(f"Provider: {metadata['provider']}")
print(f"Model: {metadata['model']}")
print(f"Fallback used: {metadata['was_fallback']}")
print(f"Attempts: {metadata['attempts']}")

# Or use FallbackResult for typed access
from core_lib.llm import FallbackResult

result = client.chat("Hello", return_fallback_result=True)
print(f"Provider: {result.provider}")
print(f"Model: {result.model}")
print(f"Was fallback: {result.was_fallback}")
print(f"Attempts: {result.attempts}")
```

### Provider Status Monitoring

Check the health status of all providers:

```python
status = client.get_provider_status()
for provider in status:
    print(f"{provider['provider']}:{provider['model']}")
    print(f"  Healthy: {provider['is_healthy']}")
    print(f"  Priority: {provider['priority']}")
    if provider['failure_reason']:
        print(f"  Failure reason: {provider['failure_reason']}")
        print(f"  Recovery at: {provider['recovery_at']}")
```

### Manual Health Control

Override health status when needed:

```python
# Mark a provider as unhealthy (e.g., known maintenance)
client.mark_provider_unhealthy("gemini", "gemini-2.0-flash", reason="maintenance")

# Mark as healthy after recovery
client.mark_provider_healthy("gemini", "gemini-2.0-flash")
```

## Error Handling

### With Exceptions (Default)

```python
try:
    response = client.chat("Hello")
except RuntimeError as e:
    print(f"All providers failed: {e}")
```

### With FallbackResult (No Exceptions)

```python
result = client.chat("Hello", return_fallback_result=True)
if result.error:
    print(f"Error: {result.error}")
else:
    print(f"Response: {result.content}")
```

## Full Example

```python
from core_lib.llm import FallbackLLMClient

# Create client with multiple providers
client = FallbackLLMClient.from_config([
    {
        "provider": "gemini",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model": "gemini-2.0-flash",
        "priority": 1,
        "min_intelligence_level": 0,
        "max_intelligence_level": 10,
    },
    {
        "provider": "openai",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-mini",
        "priority": 2,
    },
    {
        "provider": "ollama",
        "host": "http://localhost:11434",
        "model": "llama3.2",
        "priority": 3,
    },
], max_retries=2)

# Use the client
with client:
    # Simple chat
    response = client.chat("What is Python?")
    print(response["content"])
    
    # With structured output
    from pydantic import BaseModel
    
    class Answer(BaseModel):
        answer: str
        confidence: float
    
    result = client.chat(
        "What is the capital of France?",
        structured_output=Answer,
        return_fallback_result=True,
    )
    
    if not result.error:
        answer = result.content  # {"answer": "Paris", "confidence": 0.99}
        print(f"Answer: {answer['answer']}")
        print(f"Used: {result.provider} (fallback={result.was_fallback})")

# Client is automatically closed when exiting the context
```

## Migration from Manual Fallback

**Before** (manual fallback loop):
```python
from config.model_selection import iter_healthy_llm_clients, mark_provider_healthy, mark_provider_unhealthy

for client, is_fallback, config in iter_healthy_llm_clients(intelligence_level=5):
    try:
        response = await client.generate_response(prompt)
        mark_provider_healthy(config)
        break
    except Exception as e:
        mark_provider_unhealthy(config, error=e)
        continue
```

**After** (transparent fallback):
```python
from core_lib.llm import FallbackLLMClient

client = FallbackLLMClient.from_env(intelligence_level=5)
response = client.chat(prompt)  # Fallback is handled automatically

# Optional: check what happened
print(f"Used: {client.last_used_provider}, fallback: {client.last_was_fallback}")
```

## Configuration Reference

### Provider Configuration Options

| Option | Type | Description |
|--------|------|-------------|
| `provider` | str | Provider name: `gemini`, `openai`, `azure-openai`, `ollama` |
| `model` | str | Model name/identifier |
| `api_key` | str | API key (for cloud providers) |
| `host` | str | Base URL (for Ollama or custom endpoints) |
| `temperature` | float | Sampling temperature (default: 0.7) |
| `max_tokens` | int | Maximum tokens to generate |
| `priority` | int | Lower = higher priority (default: 100) |
| `enabled` | bool | Whether provider is enabled (default: true) |
| `min_intelligence_level` | int | Minimum intelligence level (0-10) |
| `max_intelligence_level` | int | Maximum intelligence level (0-10) |
| `tier` | str | Model tier: `low`, `standard`, `high` |

### FallbackLLMClient Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_retries` | int | 1 | Retries per provider before fallback |
| `intelligence_level` | int | None | Default intelligence level filter |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDERS_FILE` | Path to YAML/JSON config file |
| `LLM_PROVIDERS` | JSON array of provider configs |
| `LLM_UNHEALTHY_TTL` | Default unhealthy TTL in seconds |
| `GEMINI_API_KEY` | Gemini API key (legacy) |
| `OPENAI_API_KEY` | OpenAI API key (legacy) |
| `OLLAMA_HOST` | Ollama host URL (legacy) |
