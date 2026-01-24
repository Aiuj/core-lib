# AI Agent Instructions for core-lib

Shared Python library for MCP agent tools providing LLM clients, embeddings, reranking, caching, and unified settings/logging.

**Always use `uv run` to execute scripts and tests.**

## Architecture Overview

```
core_lib/
├── llm/           # Provider-agnostic LLM: OpenAI, Gemini, Ollama, Azure
├── embeddings/    # Multi-provider embeddings with failover
├── reranker/      # Semantic reranking: Infinity, Cohere, Local
├── cache/         # Redis-backed caching with TTL
├── jobs/          # Redis-based job queue with workers
├── config/        # Unified settings system (StandardSettings)
├── tracing/       # Langfuse/OTLP abstraction + structured logging
├── api_utils/     # APIClient base, time-based auth, FastAPI helpers
└── utils/         # Language detection, health checks
```

## Core Patterns

### LLM Providers
```python
from core_lib.llm import create_client_from_env, create_gemini_client

# Auto-detect from env vars (GEMINI_API_KEY, OPENAI_API_KEY, etc.)
client = create_client_from_env()

# Explicit provider
client = create_gemini_client(model="gemini-2.0-flash")

# Unified response: {content, structured, tool_calls, usage, text?, content_json?}
response = client.chat(messages=[{"role": "user", "content": "Hello"}])
```

### Embeddings with Failover
```python
from core_lib.embeddings import create_embedding_client, FallbackEmbeddingClient

# Simple
client = create_embedding_client(provider="openai", model="text-embedding-3-small")

# With failover chain
client = FallbackEmbeddingClient([openai_client, google_client, local_client])
```

### Settings Singleton
```python
from core_lib.config import StandardSettings, initialize_settings, get_settings

# In entrypoints (main.py, server.py) - force reload from .env
settings = initialize_settings(settings_class=MySettings, force=True, setup_logging=True)

# Anywhere else
settings = get_settings()
```

### Logging
```python
from core_lib import get_module_logger, setup_logging
from core_lib.tracing import LoggingContext, parse_from

# Setup once in entrypoint (or via initialize_settings(setup_logging=True))
setup_logging(app_name="my-app", level="INFO")

# Use in modules
logger = get_module_logger()

# Add request context (user_id, session_id, company_id)
with LoggingContext(parse_from(from_param)):
    logger.info("Processing request")  # Includes context fields
```

## Developer Workflows

```powershell
# Setup
uv sync -U --all-extras
& .\.venv\Scripts\Activate.ps1

# Tests
uv run pytest -q                    # Unit tests (mocked, no network)
uv run pytest -q --runnetwork       # Include network tests (needs API keys)

# Lint
flake8 core_lib tests && black core_lib tests
```

## Adding a New Provider

1. Create `llm/providers/<name>_provider.py` implementing `BaseProvider.chat()`
2. Add config class in `llm/llm_config.py` with `from_env()` classmethod
3. Wire in `llm_client._initialize_provider()` and add factory helper in `llm/utils.py`
4. Add tests mirroring `tests/test_llm.py` patterns (mock provider, check unified response schema)

## Key Conventions

- **Message format**: OpenAI-style `[{role: "user"|"assistant"|"system", content: str}]`
- **Structured output**: Pass Pydantic `BaseModel` type → response includes both `structured` (dict) and `text`/`content_json` (JSON string)
- **Tool calling**: Accept `tools` in OpenAI function schema, return `tool_calls` list
- **Grounding**: `use_search_grounding=True` enables provider-specific web search
- **Tracing**: Use `add_trace_metadata({...})` with minimal, non-PII data (provider, model, usage tokens)

## Key Files

| Component | Key Files |
|-----------|-----------|
| LLM | `llm/llm_client.py`, `llm/providers/*.py`, `llm/llm_config.py` |
| Embeddings | `embeddings/factory.py`, `embeddings/fallback_client.py` |
| Reranker | `reranker/factory.py`, `reranker/infinity_provider.py` |
| Settings | `config/standard_settings.py`, `config/settings_singleton.py` |
| Logging | `tracing/logger.py`, `tracing/logging_context.py` |
| Cache | `cache/redis_cache.py`, `cache/cache_manager.py` |

## Gotchas

- Providers must never raise due to tracing—wrap `add_trace_metadata` in try/except
- Do not leak prompt/response content into traces—only record lengths and usage
- Network tests are skipped by default; use `--runnetwork` flag with API keys set
- For OpenAI structured output, prefer `response_format` over manual JSON parsing
- Settings `from_env()` uses `load_dotenv=False` by default inside nested calls to avoid double-loading
