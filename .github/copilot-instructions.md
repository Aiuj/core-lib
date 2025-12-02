# AI coding agents: core-lib quickstart

This doc orients AI agents to be productive in this repo in minutes. Keep responses concise, act directly in code, and run tests after changes.

Always use "uv run" to execute scripts and tests, ensuring the virtual environment and dependencies are correctly applied.

## Architecture at a glance
- Library, not an app. Python package `core_lib/` with focused modules:
  - `utils/` and `mcp_utils.py`: MCP transport argument parsing.
  - `cache/`: Redis-backed caching via `RedisCache` plus functional facade `set_cache`, `cache_get`, `cache_set`.
  - `llm/`: Provider-agnostic LLM client with native SDK adapters.
    - `llm/providers/base.py`: provider contract (chat with OpenAI-style messages, tools, structured output, optional grounding flag). Return a unified dict: `content`, `structured`, `tool_calls`, `usage`, optional `text` and `content_json` when structured.
    - `llm/providers/google_genai_provider.py`: Google Gemini via `google.genai` with structured outputs, tools, optional Search grounding.
    - `llm/providers/ollama_provider.py`: Local Ollama via `ollama`. Supports function tools and JSON format.
    - `llm/providers/openai_provider.py`: OpenAI/Azure/OpenAI-compatible endpoints via official `openai` SDK. Structured outputs via `response_format`, tool calling; can add web/file search tools when `use_search_grounding=True`.
    - `llm/llm_client.py`: entry point selecting provider by config, normalizes messages, forwards flags, and attaches trace metadata before/after calls.
    - `llm/llm_config.py`: config classes (`GeminiConfig`, `OllamaConfig`, `OpenAIConfig`) with `from_env()` to populate from env vars.
  - `reranker/`: Provider-agnostic reranking for improved search quality.
    - `reranker/base.py`: base contract with `rerank()` returning `RerankResult` dataclass (index, score, document).
    - `reranker/infinity_provider.py`: Infinity server for local high-throughput reranking.
    - `reranker/cohere_provider.py`: Cohere cloud reranking API.
    - `reranker/local_provider.py`: Local cross-encoder models via sentence-transformers.
    - `reranker/factory.py`: `create_reranker_client()`, provider-specific helpers, and `RerankerFactory` class.
    - `reranker/reranker_config.py`: `RerankerSettings` with `from_env()` for env-based configuration.
  - `tracing/`: thin wrapper around Langfuse/OpenTelemetry, plus logging context with `parse_from()`. Use `setup_tracing()` and `add_trace_metadata()`; providers should not import Langfuse directly.
- Tests live in `tests/` and mock provider classes where possible to avoid network calls.

## Key patterns and conventions
- OpenAI-style message dicts: `{role: "user"|"assistant"|"system", content: str}` are the lingua franca across providers.
- Structured outputs: pass a Pydantic `BaseModel` type; providers must return both Python `dict` and serialized JSON under `text`/`content_json` when structured.
- Tool calling: accept `tools` in OpenAI function schema and return `tool_calls` from the provider.
- Grounding flag: `use_search_grounding` is provider-specific. Gemini enables Google Search; OpenAI adds `web_search`/`file_search` tool specs if the account supports them.
- Tracing: call `add_trace_metadata({...})` with minimal, non-PII context (provider, model, usage). Do not depend on provider-specific tracing libs in adapters.


## Developer workflows
- **Always activate the uv-managed `.venv` before running any Python script or command.**
  - Create/sync: `uv sync -U --all-extras`
  - Activate (PowerShell): `& .\.venv\Scripts\Activate.ps1`
- Install editable: `uv pip install -e ".[dev]"` (or `pip install -e .[dev]`)
- Run tests: `uv run pytest -q` (Task "Run core-lib tests" available in VS Code)
- Lint/format: `flake8 core_lib tests` and `black core_lib tests`

## External integrations
- Redis: via `redis` lib; see env vars in README (REDIS_HOST, PORT, DB, PREFIX, TTL, PASSWORD, TIMEOUT).
- Google GenAI: `google-genai` SDK. Config via `GeminiConfig.from_env()` reads `GEMINI_*` and `GOOGLE_GENAI_*`.
- OpenAI/Azure/OpenAI-compatible: official `openai` SDK using `OpenAIConfig`. Supports:
  - OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL`, `OPENAI_ORG`, `OPENAI_PROJECT`.
  - Azure: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT`.
  - Compatible gateways (e.g., Ollama proxy/LiteLLM): set `OPENAI_BASE_URL` and an API key string.
- Tracing: `core_lib.tracing.tracing` abstracts Langfuse/OpenTelemetry. Use `setup_tracing()` once and `add_trace_metadata()` during calls.

## How to add a new provider
1. Create `llm/providers/<name>_provider.py` implementing `BaseProvider.chat()`.
2. Add a config class in `llm/llm_config.py` with `from_env()`.
3. Wire it in `llm_client._initialize_provider()` and add factory helpers in `llm/utils.py`.
4. Tests: mirror `tests/test_llm.py` patterns. Mock the provider class and return a dict matching the unified schema.

## Examples from code
- Normalization and tracing entry: `LLMClient.chat()` in `llm/llm_client.py` shows how messages and flags are handled and where metadata is attached.
- Structured outputs round-trip expectation: see `tests/test_llm.py::test_structured_output_includes_python_and_json_fields`.
- Gemini thoughts config: `google_genai_provider._build_config()` shows how optional features are enabled without breaking.

## Gotchas
- Providers must never raise due to tracing; wrap `add_trace_metadata` in try/except.
- Do not leak full prompt/response content into traces; record lengths and usage only.
- For OpenAI structured output, prefer `response_format`; if helper is unavailable, build a `json_schema` param from Pydantic.
- Keep networked tests skipped by default; use env keys to run manually.

## Quick API reference
- Create from env: `from core_lib.llm import create_client_from_env`
- Explicit providers: `create_gemini_client`, `create_ollama_client`, `create_openai_client`, `create_azure_openai_client`, `create_openai_compatible_client`.
- Cache: `set_cache`, `cache_get`, `cache_set`; class: `RedisCache`.
- Reranker: `create_reranker_client`, `create_infinity_reranker`, `create_cohere_reranker`, `create_local_reranker`; class: `RerankerFactory`.
- Tracing/logging: `parse_from` (from `core_lib.tracing`), `LoggingContext`, `setup_logging`, `setup_tracing`.
- MCP utils: `get_transport_from_args`.
