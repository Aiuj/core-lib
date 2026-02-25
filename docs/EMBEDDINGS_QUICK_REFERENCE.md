# Embeddings Quick Reference

Get started with embeddings in `core-lib` in minutes. **For comprehensive documentation, see [EMBEDDINGS_GUIDE.md](./EMBEDDINGS_GUIDE.md).**

## Installation

```bash
# Recommended: Install with all embedding providers
uv pip install "core-lib[all]"

# Or install specific providers
uv pip install "core-lib[embeddings]"  # Core only
uv pip install "core-lib[infinity]"    # + Infinity
uv pip install "core-lib[openai]"      # + OpenAI
```

## Quick Start

### Development (Single Host)

```bash
# .env
EMBEDDING_PROVIDER=infinity
EMBEDDING_BASE_URL=http://localhost:7997
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

```python
from core_lib.embeddings import create_embedding_client

client = create_embedding_client()
embedding = client.generate_embedding("Hello, world!")
```

### Production (High Availability)

```bash
# .env - Comma-separated URLs = automatic failover
EMBEDDING_PROVIDER=infinity
INFINITY_BASE_URL=http://h1:7997,http://h2:7997,http://h3:7997
INFINITY_TOKEN=token1,token2,token3
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

```python
from core_lib.embeddings import create_embedding_client

# Automatically creates FallbackEmbeddingClient with 3 providers
client = create_embedding_client()
embedding = client.generate_embedding("Production text")
```

**That's it!** Automatic failover with zero configuration changes.

## Key Features

✅ **Zero-Config HA**: Comma-separated URLs trigger automatic fallback  
✅ **Token Authentication**: Secure your embedding servers  
✅ **Multiple Providers**: Infinity, OpenAI, Google GenAI, Ollama, Local  
✅ **Task-Specific**: Optimize for similarity, clustering, classification  
✅ **Caching**: Automatic deduplication and result caching  
✅ **Health Checks**: Monitor provider availability  

## Recommended Setup

### Why Infinity?

- ✅ Local deployment (privacy, no API costs)
- ✅ High throughput (GPU acceleration)
- ✅ Any HuggingFace model
- ✅ OpenAI-compatible API

### Start Infinity Server

```bash
# Docker (recommended)
docker run -p 7997:7997 michaelf34/infinity:latest --model-name-or-path BAAI/bge-small-en-v1.5

# Or install locally
uv pip install infinity-emb[all]
infinity_emb v2 --model-name-or-path BAAI/bge-small-en-v1.5
```

## Common Patterns

### Single Embedding

```python
from core_lib.embeddings import create_embedding_client

client = create_embedding_client()
embedding = client.generate_embedding("Your text here")
print(f"Dimension: {len(embedding)}")
```

### Batch Embeddings

```python
# More efficient than individual calls
embeddings = client.generate_embeddings([
    "First document",
    "Second document",
    "Third document"
])

for i, emb in enumerate(embeddings):
    print(f"Document {i+1}: {len(emb)} dimensions")
```

### With Task Type

```python
from core_lib.embeddings import create_embedding_client, TaskType

client = create_embedding_client()
embedding = client.generate_embedding(
    "Search query",
    task_type=TaskType.RETRIEVAL_QUERY
)
```

### Custom Dimensions

```python
from core_lib.embeddings import create_openai_client

# Reduce token usage for OpenAI
client = create_openai_client(
    model="text-embedding-3-small",
    embedding_dim=512  # Default is 1536
)
```

## Essential Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `EMBEDDING_PROVIDER` | Provider type | `infinity`, `openai` |
| `EMBEDDING_MODEL` | Model name | `BAAI/bge-small-en-v1.5` |
| `INFINITY_BASE_URL` | Infinity server URL(s) | `http://localhost:7997` or<br>`http://h1:7997,http://h2:7997` |
| `INFINITY_TOKEN` | Auth token(s) | `token123` or<br>`token1,token2,token3` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |

**For complete configuration reference, see [EMBEDDINGS_GUIDE.md](./EMBEDDINGS_GUIDE.md).**

## Quick Examples

### Development

```bash
# .env
EMBEDDING_PROVIDER=infinity
EMBEDDING_BASE_URL=http://localhost:7997
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Staging (HA without Auth)

```bash
# .env
EMBEDDING_PROVIDER=infinity
INFINITY_BASE_URL=http://staging-h1:7997,http://staging-h2:7997
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Production (HA with Auth)

```bash
# .env
EMBEDDING_PROVIDER=infinity
INFINITY_BASE_URL=http://prod-h1:7997,http://prod-h2:7997,http://prod-h3:7997
INFINITY_TOKEN=token1,token2,token3
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024
EMBEDDING_CACHE_DURATION_SECONDS=7200
```

## Providers at a Glance

| Provider | Best For | Configuration |
|----------|----------|---------------|
| **Infinity** | Production (local, fast) | `INFINITY_BASE_URL`, `INFINITY_TOKEN` |
| **OpenAI** | Cloud, latest models | `OPENAI_API_KEY` |
| **Google GenAI** | Task-specific embeddings | `GOOGLE_GENAI_API_KEY` |
| **Ollama** | Local experimentation | `OLLAMA_URL` |
| **Local** | Offline, privacy | `EMBEDDING_MODEL` (HuggingFace) |

## Health Checks

```python
# Check provider availability
if client.health_check():
    print("✓ Provider is healthy")
```

## Common Task Types

```python
from core_lib.embeddings import TaskType

# Most common
TaskType.SEMANTIC_SIMILARITY  # Default - similarity search
TaskType.RETRIEVAL_QUERY      # Search queries
TaskType.RETRIEVAL_DOCUMENT   # Document indexing
TaskType.CLASSIFICATION       # Text classification
TaskType.CLUSTERING           # Grouping similar texts
```

## Popular Models

### Infinity (HuggingFace Models)

```bash
# Fast, good quality (384 dims)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Balanced (768 dims)
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Best quality (1024 dims)
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Multilingual (768 dims)
EMBEDDING_MODEL=intfloat/e5-base-v2
```

### OpenAI

```bash
# Cost-effective (1536 dims, can reduce)
EMBEDDING_MODEL=text-embedding-3-small

# Best performance (3072 dims, can reduce)
EMBEDDING_MODEL=text-embedding-3-large

# Legacy (1536 dims)
EMBEDDING_MODEL=text-embedding-ada-002
```

## Performance Tips

1. **Use batching** for multiple texts (more efficient)
2. **Enable caching** to avoid redundant API calls
3. **Use Infinity** for high throughput (local GPU)
4. **Reduce dimensions** for OpenAI to save costs
5. **Set appropriate timeouts** (30s default)

## Next Steps

- 📖 **Comprehensive Guide**: [EMBEDDINGS_GUIDE.md](./EMBEDDINGS_GUIDE.md)
- 🚀 **Infinity Setup**: [INFINITY_QUICKSTART.md](./INFINITY_QUICKSTART.md)
- 🔧 **Infinity Provider**: [INFINITY_PROVIDER.md](./INFINITY_PROVIDER.md)
- 🌙 **Wake-on-LAN (Embeddings)**: [WAKE_ON_LAN.md](./WAKE_ON_LAN.md)

## Support

For issues or questions:
- Check [EMBEDDINGS_GUIDE.md](./EMBEDDINGS_GUIDE.md) troubleshooting section
- Review examples in `examples/example_embeddings_usage.py`
- See test cases in `tests/test_embeddings.py`
