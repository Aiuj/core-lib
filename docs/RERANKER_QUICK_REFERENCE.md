````markdown
# Reranker Quick Reference

Get started with reranking in `core-lib` in minutes. **For comprehensive documentation, see [RERANKER_GUIDE.md](./RERANKER_GUIDE.md).**

## Installation

```bash
# Recommended: Install with all providers
uv pip install "core-lib[all]"

# Or install specific providers
uv pip install "core-lib[reranker]"     # Core + Infinity (requests)
uv pip install "core-lib[cohere]"       # + Cohere
uv pip install "core-lib[local]"        # + sentence-transformers for local models
```

## Quick Start

### Development (Single Host)

```bash
# .env
RERANKER_PROVIDER=infinity
INFINITY_BASE_URL=http://localhost:7997
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

```python
from core_lib.reranker import create_reranker_client

client = create_reranker_client()
results = client.rerank(
    query="What is machine learning?",
    documents=[
        "Machine learning is a subset of AI.",
        "Python is a programming language.",
        "Deep learning uses neural networks.",
    ],
    top_k=2
)

for result in results:
    print(f"Score: {result.score:.4f} | {result.document[:50]}...")
```

### Production (Cohere Cloud)

```bash
# .env
RERANKER_PROVIDER=cohere
COHERE_API_KEY=your-api-key
RERANKER_MODEL=rerank-multilingual-v3.0
```

```python
from core_lib.reranker import create_reranker_client

client = create_reranker_client()
results = client.rerank(query="query", documents=["doc1", "doc2"])
```

**That's it!** Automatic provider selection based on environment.

## Key Features

✅ **Multiple Providers**: Infinity (local), Cohere (cloud), Local (sentence-transformers)  
✅ **Caching**: Automatic result caching with configurable TTL  
✅ **Health Checks**: Monitor provider availability  
✅ **Cross-Encoder Models**: High-quality semantic reranking  
✅ **Multilingual Support**: Works with multiple languages  

## Recommended Setup

### Why Reranking?

Rerankers improve search quality by re-scoring query-document pairs using cross-encoder models that consider the full context of both query and document. Unlike embeddings (bi-encoders), rerankers are more accurate but slower.

**Typical RAG Pipeline:**
1. **Retrieve** (fast): Get top-100 candidates using embeddings
2. **Rerank** (accurate): Re-score top-100 → select top-10

### Why Infinity?

- ✅ Local deployment (privacy, no API costs)
- ✅ High throughput (GPU acceleration)
- ✅ Any HuggingFace cross-encoder model
- ✅ Cohere-compatible API

### Start Infinity Server (with Reranking)

```bash
# Docker (recommended)
docker run -p 7997:7997 michaelf34/infinity:latest \
  --model-name-or-path BAAI/bge-reranker-v2-m3

# Or install locally
uv pip install infinity-emb[all]
infinity_emb v2 --model-name-or-path BAAI/bge-reranker-v2-m3
```

## Common Patterns

### Basic Reranking

```python
from core_lib.reranker import create_reranker_client

client = create_reranker_client()

results = client.rerank(
    query="What is the capital of France?",
    documents=[
        "Paris is the capital of France.",
        "London is the capital of England.",
        "Berlin is the capital of Germany.",
    ],
    top_k=2
)

for result in results:
    print(f"[{result.index}] Score: {result.score:.4f} - {result.document}")
```

### Rerank with Tuples

```python
# Get (index, score, document) tuples
results = client.rerank_with_scores(
    query="machine learning",
    documents=["ML is AI", "Python code", "Neural networks"],
    top_k=2
)

for index, score, doc in results:
    print(f"Document {index}: {score:.4f}")
```

### Provider-Specific Creation

```python
from core_lib.reranker import (
    create_infinity_reranker,
    create_cohere_reranker,
    create_local_reranker,
)

# Infinity (local server)
infinity_client = create_infinity_reranker(
    model="BAAI/bge-reranker-v2-m3",
    base_url="http://localhost:7997"
)

# Cohere (cloud API)
cohere_client = create_cohere_reranker(
    model="rerank-multilingual-v3.0",
    api_key="your-api-key"
)

# Local (sentence-transformers)
local_client = create_local_reranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda"  # or "cpu", "auto"
)
```

### With Caching

```python
from core_lib.reranker import create_reranker_client

# Cache results for 2 hours
client = create_reranker_client(cache_duration_seconds=7200)

# First call - computes reranking
results1 = client.rerank(query="test", documents=["doc1", "doc2"])

# Second call - returns cached results
results2 = client.rerank(query="test", documents=["doc1", "doc2"])

# Disable caching
client_no_cache = create_reranker_client(cache_duration_seconds=0)
```

## Essential Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RERANKER_PROVIDER` | Provider type | `infinity`, `cohere`, `local` |
| `RERANKER_MODEL` | Model name | `BAAI/bge-reranker-v2-m3` |
| `INFINITY_BASE_URL` | Infinity server URL | `http://localhost:7997` |
| `INFINITY_RERANK_URL` | Dedicated reranker URL (overrides base) | `http://reranker:7997` |
| `INFINITY_TOKEN` | Auth token | `token123` |
| `COHERE_API_KEY` | Cohere API key | `co-...` |
| `RERANKER_TIMEOUT` | Request timeout (seconds) | `30` |
| `RERANKER_CACHE_DURATION_SECONDS` | Cache TTL (0 to disable) | `3600` |

**For complete configuration reference, see [RERANKER_GUIDE.md](./RERANKER_GUIDE.md).**

## Providers at a Glance

| Provider | Best For | Configuration |
|----------|----------|---------------|
| **Infinity** | Production (local, fast, GPU) | `INFINITY_BASE_URL`, `INFINITY_TOKEN` |
| **Cohere** | Cloud, high quality | `COHERE_API_KEY` |
| **Local** | Offline, privacy, experimentation | `RERANKER_MODEL` (HuggingFace) |

## Popular Models

### Infinity / Local (HuggingFace Models)

```bash
# Multilingual, high quality (recommended)
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# English, high quality
RERANKER_MODEL=BAAI/bge-reranker-large

# English, balanced
RERANKER_MODEL=BAAI/bge-reranker-base

# English, fast
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# English, very fast
RERANKER_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2
```

### Cohere

```bash
# Multilingual (recommended)
RERANKER_MODEL=rerank-multilingual-v3.0

# English only
RERANKER_MODEL=rerank-english-v3.0

# Previous generation
RERANKER_MODEL=rerank-english-v2.0
RERANKER_MODEL=rerank-multilingual-v2.0
```

## Health Checks

```python
# Check provider availability
if client.health_check():
    print("✓ Reranker is healthy")
else:
    print("✗ Reranker unavailable")

# Get rerank timing
results = client.rerank(query, documents)
print(f"Reranking took {client.get_rerank_time_ms():.2f}ms")
```

## Performance Tips

1. **Limit documents** - Rerank top-100, not thousands
2. **Use caching** - Avoid redundant API calls for same queries
3. **Use Infinity** - For high throughput with GPU acceleration
4. **Choose appropriate model** - Faster models for real-time, larger for quality
5. **Set appropriate timeouts** - 30s default, increase for large batches

## RAG Integration Example

```python
from core_lib.embeddings import create_embedding_client
from core_lib.reranker import create_reranker_client

# Initialize clients
embedding_client = create_embedding_client()
reranker_client = create_reranker_client()

# Step 1: Retrieve candidates using embeddings (fast)
query = "What is machine learning?"
query_embedding = embedding_client.generate_embedding(query)
# ... search your vector database for top-100 candidates ...
candidates = ["doc1", "doc2", "doc3", ...]  # top-100 from vector search

# Step 2: Rerank candidates (accurate)
results = reranker_client.rerank(
    query=query,
    documents=candidates,
    top_k=10  # Final top-10
)

# Use reranked results
for result in results:
    print(f"Score: {result.score:.4f} | {result.document[:100]}")
```

## Next Steps

- 📖 **Comprehensive Guide**: [RERANKER_GUIDE.md](./RERANKER_GUIDE.md)
- 🚀 **Infinity Setup**: [INFINITY_QUICKSTART.md](./INFINITY_QUICKSTART.md)
- 🔧 **Embeddings Guide**: [EMBEDDINGS_GUIDE.md](./EMBEDDINGS_GUIDE.md)

## Support

For issues or questions:
- Check [RERANKER_GUIDE.md](./RERANKER_GUIDE.md) troubleshooting section
- Review test cases in `tests/test_reranker.py`
- See the reranker module source in `core_lib/reranker/`

````
