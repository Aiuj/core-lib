# Environment Variables Reference

This file lists all environment variables for configuring LLM clients, embeddings, rerankers, and logging.

## Embeddings Configuration

### Unified Configuration (Recommended for Single Provider)

Simplest approach when using one embedding provider:

```bash
export EMBEDDING_PROVIDER=infinity              # Provider: openai|google_genai|ollama|infinity|local
export EMBEDDING_MODEL=BAAI/bge-small-en-v1.5  # Model name
export EMBEDDING_BASE_URL=http://localhost:7997 # Server URL (works for all providers)
export EMBEDDING_TIMEOUT=30                     # Request timeout in seconds
export EMBEDDING_DIMENSION=384                  # Optional: embedding dimension
export EMBEDDING_TASK_TYPE=SEMANTIC_SIMILARITY  # Optional: task type
export EMBEDDING_CACHE_DURATION_SECONDS=7200    # Cache duration in seconds (default: 7200 = 2 hours)
                                                # Set to 0 to disable caching entirely
```

### Provider-Specific Configuration (For Multi-Provider Setups)

When running multiple embedding providers simultaneously:

```bash
# Common defaults
export EMBEDDING_BASE_URL=http://default-server:7997
export EMBEDDING_TIMEOUT=30

# Provider-specific overrides (take precedence)
export INFINITY_BASE_URL=http://infinity-server:7997
export INFINITY_TIMEOUT=30
export OLLAMA_URL=http://ollama-server:11434
export OLLAMA_TIMEOUT=60
export OPENAI_BASE_URL=https://api.openai.com/v1

# API keys
export OPENAI_API_KEY=your-key
export GOOGLE_GENAI_API_KEY=your-key
```

**Priority Chain:**
- Ollama: `OLLAMA_URL` > `EMBEDDING_BASE_URL` > default
- Infinity: `EMBEDDING_BASE_URL` > default
- OpenAI: `OPENAI_BASE_URL` > `EMBEDDING_BASE_URL` > default

**See:** [EMBEDDINGS_GUIDE.md](../docs/EMBEDDINGS_GUIDE.md) for detailed configuration guide

## Reranker Configuration

### Basic Reranker Settings

```bash
export RERANKER_PROVIDER=infinity              # Provider: infinity|cohere|local
export RERANKER_MODEL=BAAI/bge-reranker-v2-m3  # Model name (multilingual default)
export RERANKER_TIMEOUT=30                     # Request timeout in seconds
export RERANKER_CACHE_DURATION_SECONDS=3600    # Cache duration (0 to disable)
export RERANKER_DEFAULT_TOP_K=10               # Default number of results
export RERANKER_SCORE_THRESHOLD=               # Optional minimum score threshold
```

### Infinity Reranker (Local Server)

```bash
# Server URL (shared with embeddings Infinity server)
export INFINITY_BASE_URL=http://localhost:7997

# Or use dedicated reranker URL (takes precedence)
export INFINITY_RERANK_URL=http://reranker:7997

# Authentication (optional)
export INFINITY_TOKEN=your-auth-token
# Or use reranker-specific token
export RERANKER_TOKEN=your-auth-token

# Timeout
export INFINITY_TIMEOUT=30
```

### Cohere Reranker (Cloud API)

```bash
export RERANKER_PROVIDER=cohere
export COHERE_API_KEY=your-cohere-api-key
export RERANKER_MODEL=rerank-multilingual-v3.0  # or rerank-english-v3.0
```

### Local Reranker (sentence-transformers)

```bash
export RERANKER_PROVIDER=local
export RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
export RERANKER_DEVICE=auto                     # cpu, cuda, or auto
export RERANKER_CACHE_DIR=/path/to/models       # Optional model cache directory
export RERANKER_TRUST_REMOTE_CODE=false         # Trust remote code from HuggingFace
```

**See:** [RERANKER_GUIDE.md](./RERANKER_GUIDE.md) for detailed configuration guide

## Ollama Configuration

```bash
# Model and basic settings
export OLLAMA_MODEL=llama3.2                    # Model name to use

# For LLM (chat/generation)
export OLLAMA_BASE_URL=http://localhost:11434   # Ollama server URL for LLM
export OLLAMA_TEMPERATURE=0.7                   # Sampling temperature (0.0-2.0)
export OLLAMA_MAX_TOKENS=                       # Maximum tokens to generate (optional)
export OLLAMA_THINKING_ENABLED=false            # Enable step-by-step thinking mode

# For Embeddings (use unified config or provider-specific)
export EMBEDDING_BASE_URL=http://localhost:11434  # Recommended: unified config
# OR
# export OLLAMA_URL=http://localhost:11434       # Alternative: provider-specific

# Timeout settings
export EMBEDDING_TIMEOUT=30                     # Unified timeout for embeddings
# OR  
# export OLLAMA_TIMEOUT=60                      # Provider-specific timeout
export OLLAMA_TIMEOUT=60                        # Timeout for LLM operations

# Advanced Ollama settings
export OLLAMA_NUM_CTX=                          # Context window size (optional)
export OLLAMA_NUM_PREDICT=                      # Max tokens to predict (optional)
export OLLAMA_REPEAT_PENALTY=                   # Repetition penalty (optional)
export OLLAMA_TOP_K=                            # Top-K sampling (optional)
export OLLAMA_TOP_P=                            # Top-P sampling (optional)
```

## Example .env file

```bash
# Required settings
export GEMINI_API_KEY=your-google-api-key       # Google API key (REQUIRED)

# Model and basic settings  
export GEMINI_MODEL=gemini-1.5-flash            # Model name to use
export GEMINI_TEMPERATURE=0.7                   # Sampling temperature (0.0-2.0)
export GEMINI_MAX_TOKENS=                       # Maximum tokens to generate (optional)
export GEMINI_THINKING_ENABLED=false            # Enable step-by-step thinking mode

# Advanced Gemini settings
export GEMINI_BASE_URL=https://generativelanguage.googleapis.com  # API base URL
```

## Example .env file

```bash
# Choose your provider
# For Ollama (local)
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEMPERATURE=0.7
OLLAMA_THINKING_ENABLED=true

# For Gemini (cloud)
# GEMINI_API_KEY=your-actual-api-key-here
# GEMINI_MODEL=gemini-1.5-flash
# GEMINI_TEMPERATURE=0.3
```

## Logging Configuration

### Basic Logging

```bash
export LOG_LEVEL=INFO                        # Root log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
export LOG_FILE_ENABLED=true                 # Enable file logging
export LOG_FILE_PATH=logs/app.log            # Log file path
export LOG_FILE_MAX_BYTES=1048576            # Max file size before rotation (1MB)
export LOG_FILE_BACKUP_COUNT=3               # Number of backup files
```

### OTLP (OpenTelemetry) Logging

**Quick Setup (Auto-Enable):**
```bash
# Minimal config - OTLP auto-enables with these settings
export ENABLE_LOGGER=true                                   # Enable logging features
export OTLP_ENDPOINT=http://localhost:4318/v1/logs          # OTLP collector endpoint

# Optional: customize service identification (auto-detected)
export APP_NAME=my-app                                      # Auto-used for OTLP_SERVICE_NAME
# OTLP_SERVICE_VERSION auto-detected from pyproject.toml
```

**Full Configuration:**
```bash
export OTLP_ENABLED=true                                    # Explicit enable (optional if auto-enabled)
export OTLP_ENDPOINT=http://localhost:4318/v1/logs          # OTLP collector endpoint
export OTLP_LOG_LEVEL=INFO                                  # Independent OTLP log level (optional)
export OTLP_HEADERS='{"Authorization": "Bearer token"}'     # Auth headers (JSON)
export OTLP_TIMEOUT=10                                      # Request timeout (seconds)
export OTLP_INSECURE=false                                  # Skip SSL verification
export OTLP_SERVICE_NAME=my-app                             # Service name (default: APP_NAME or "core-lib")
export OTLP_SERVICE_VERSION=1.0.0                           # Service version (default: from pyproject.toml)
```

**Auto-Enable Logic:**
OTLP automatically enables when:
1. `ENABLE_LOGGER=true` (or `LOG_FILE_ENABLED=true`) AND
2. `OTLP_ENDPOINT` is defined AND
3. `OTLP_ENABLED` is not explicitly set to `false`

**Independent Log Levels:**
```bash
# Example: DEBUG on console, only INFO+ to OTLP
export LOG_LEVEL=DEBUG
export OTLP_LOG_LEVEL=INFO

# Example: Reduce OTLP costs - only errors
export LOG_LEVEL=INFO
export OTLP_LOG_LEVEL=ERROR
```

### OVH Logs Data Platform

```bash
export OVH_LDP_ENABLED=true                  # Enable OVH LDP
export OVH_LDP_TOKEN=your-token              # Authentication token
export OVH_LDP_ENDPOINT=gra1.logs.ovh.com    # OVH endpoint
export OVH_LDP_PORT=12202                    # GELF TCP port
export OVH_LDP_PROTOCOL=gelf_tcp             # Protocol: gelf_tcp, gelf_udp, syslog_tcp, syslog_udp
export OVH_LDP_USE_TLS=true                  # Use TLS encryption
```

## Usage in Code

```python
from core_lib import create_client_from_env
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing.logger import setup_logging

# LLM clients use environment variables automatically
ollama_client = create_client_from_env("ollama")
gemini_client = create_client_from_env("gemini")

# Logging uses environment variables automatically
logger_settings = LoggerSettings.from_env()
logger = setup_logging(logger_settings=logger_settings)
```

## Running Scripts

Use `uv run` to execute scripts with the proper environment:

```bash
# Run Python scripts
uv run python your_script.py

# Run examples
uv run python examples/example_embeddings_usage.py
uv run python examples/example_llm_usage.py

# Run tests
uv run pytest -q
```
