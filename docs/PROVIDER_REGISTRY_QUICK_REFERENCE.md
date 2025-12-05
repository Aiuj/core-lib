# LLM Provider Registry Quick Reference

Configure multiple LLM providers with automatic fallback support.

## Installation

The `ProviderRegistry` is included in `core_lib.llm`:

```python
from core_lib.llm import ProviderRegistry, ProviderConfig
```

## Configuration Methods

### 1. Environment Variable (JSON)

```bash
export LLM_PROVIDERS='[
  {"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash", "priority": 1, "tier": "standard"},
  {"provider": "openai", "api_key": "sk-...", "model": "gpt-4o", "priority": 2, "min_level": 7, "max_level": 10, "tier": "high"}
]'
```

```python
registry = ProviderRegistry.from_env()
```

### 2. Legacy Environment Variables

When `LLM_PROVIDERS` is not set, falls back to individual vars:

```bash
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash
OPENAI_API_KEY=sk-...
OLLAMA_HOST=http://localhost:11434
```

### 3. Configuration File

```python
registry = ProviderRegistry.from_file("llm_providers.json")
```

### 4. Programmatic

```python
registry = ProviderRegistry.from_list([
    {"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash"},
    {"provider": "openai", "api_key": "...", "model": "gpt-4o-mini"},
])

# Or using ProviderConfig directly
registry = ProviderRegistry()
registry.add(ProviderConfig(provider="gemini", api_key="...", model="..."))
```

## Provider Config Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | str | Yes | `gemini`, `openai`, `azure-openai`, `ollama` |
| `model` | str | No | Model name (has sensible defaults) |
| `api_key` | str | Cloud | API key for authentication |
| `host` | str | Ollama | Base URL for the service |
| `temperature` | float | No | Sampling temperature (default: 0.7) |
| `max_tokens` | int | No | Max generation tokens |
| `priority` | int | No | Lower = higher priority (default: 100) |
| `enabled` | bool | No | Enable/disable provider (default: true) |
| `min_intelligence_level` | int | No | Minimum question complexity (0-10, default: 0) |
| `max_intelligence_level` | int | No | Maximum question complexity (0-10, default: 10) |
| `tier` | str | No | Model tier label: `low`, `standard`, `high` |

### Provider-Specific

**Azure OpenAI:**
- `azure_endpoint`: Azure resource endpoint
- `azure_api_version`: API version (default: 2024-08-01-preview)

**OpenAI:**
- `organization`: OpenAI organization ID
- `project`: OpenAI project ID

**Ollama:**
- Extra options in `extra` dict: `timeout`, `num_ctx`, `num_predict`, etc.

## Usage Examples

### Get Primary Client

```python
registry = ProviderRegistry.from_env()
client = registry.get_client()  # Primary provider
response = client.chat([{"role": "user", "content": "Hello"}])
```

### Iterate with Fallback

```python
for client, is_fallback, provider_info in registry.iter_clients():
    try:
        response = client.chat(messages)
        print(f"Success with {provider_info['model']}! Fallback: {is_fallback}")
        break
    except Exception as e:
        print(f"Provider {provider_info['provider']} failed: {e}")
        continue
```

### Filter by Intelligence Level

```python
# Get providers suitable for a specific complexity level
providers = registry.get_providers_for_intelligence_level(level=5)

# Get the best (highest priority) provider for a level
best = registry.get_best_provider_for_level(level=8)
if best:
    client = best.to_client()
```

### Get Specific Provider

```python
primary = registry.get_client(0)    # Primary
fallback1 = registry.get_client(1)  # First fallback
fallback2 = registry.get_client(2)  # Second fallback
```

### Convert to LLMConfig

```python
provider = registry.get_primary()
llm_config = provider.to_llm_config()  # GeminiConfig, OpenAIConfig, etc.
client = provider.to_client()          # LLMClient instance
```

## Example Configurations

### Development
```json
[
  {"provider": "ollama", "model": "qwen3:8b", "priority": 1, "tier": "standard"},
  {"provider": "gemini", "api_key": "...", "model": "gemma-3-4b-it", "priority": 2, "tier": "low"}
]
```

### Production with Intelligence Levels
```json
[
  {"provider": "ollama", "model": "qwen3:1.7b", "priority": 1, "min_level": 0, "max_level": 3, "tier": "low"},
  {"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash", "priority": 1, "min_level": 4, "max_level": 7, "tier": "standard"},
  {"provider": "openai", "api_key": "...", "model": "gpt-4o", "priority": 1, "min_level": 8, "max_level": 10, "tier": "high"}
]
```

### Cost-Optimized with Fallback
```json
[
  {"provider": "ollama", "model": "llama3.2", "priority": 1, "tier": "low"},
  {"provider": "gemini", "api_key": "...", "model": "gemma-3-4b-it", "priority": 2, "tier": "low"},
  {"provider": "openai", "api_key": "...", "model": "gpt-4o-mini", "priority": 3, "tier": "standard"}
]
```

## Intelligence Level Selection

Intelligence levels (0-10) allow routing questions to appropriate models:

| Level Range | Use Case | Typical Model |
|-------------|----------|---------------|
| 0-3 | Simple lookups, yes/no questions | `qwen3:1.7b`, `gemma-3-4b` |
| 4-7 | Standard questions, moderate reasoning | `gemini-2.0-flash`, `gpt-4o-mini` |
| 8-10 | Complex analysis, detailed synthesis | `gpt-4o`, `claude-sonnet` |

```python
# Configure providers for different complexity levels
registry = ProviderRegistry.from_list([
    {"provider": "ollama", "model": "qwen3:1.7b", "min_level": 0, "max_level": 3, "tier": "low"},
    {"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash", "min_level": 4, "max_level": 10},
])

# Select model based on question complexity
for question in questions:
    level = assess_complexity(question)  # Returns 0-10
    provider = registry.get_best_provider_for_level(level)
    if provider:
        response = provider.to_client().chat([{"role": "user", "content": question}])
```

## Resource Management

```python
with ProviderRegistry.from_env() as registry:
    client = registry.get_client()
    response = client.chat(messages)
# Clients automatically closed

# Or manually:
registry.clear_cache()  # Close all cached clients
```
