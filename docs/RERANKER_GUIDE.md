````markdown
# Reranker Guide

This guide covers the complete reranker functionality in `core-lib`, including all providers, configuration options, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Providers](#providers)
   - [Infinity](#infinity-provider)
   - [Cohere](#cohere-provider)
   - [Local](#local-provider)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Caching](#caching)
8. [RAG Integration](#rag-integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

### What is Reranking?

Reranking is a technique to improve search result quality by re-scoring query-document pairs using cross-encoder models. Unlike bi-encoder embeddings that encode query and documents separately, cross-encoders process both together, capturing nuanced semantic relationships.

**Embeddings (Bi-encoders):**
- Fast: Encode once, compare vectors
- Scalable: Search millions of documents
- Less accurate: No direct query-document interaction

**Rerankers (Cross-encoders):**
- Slower: Process each query-document pair
- More accurate: Full attention between query and document
- Use after retrieval: Rerank top-N candidates

### Typical RAG Pipeline

```
Query → Embedding Search (top-100) → Reranking (top-10) → LLM Response
         (fast, approximate)         (accurate)
```

### Supported Providers

| Provider | Type | Best For |
|----------|------|----------|
| **Infinity** | Local server | Production, privacy, GPU acceleration |
| **Cohere** | Cloud API | High quality, managed service |
| **Local** | In-process | Offline, experimentation, privacy |

## Installation

### Core Installation

```bash
# Install core-lib with reranker support
uv pip install "core-lib[reranker]"
```

### Provider-Specific Dependencies

```bash
# Infinity (uses requests - included in core)
uv pip install requests

# Cohere
uv pip install cohere

# Local (sentence-transformers)
uv pip install sentence-transformers
```

### All Providers

```bash
uv pip install "core-lib[all]"
```

## Quick Start

### Basic Usage

```python
from core_lib.reranker import create_reranker_client

# Auto-detect provider from environment
client = create_reranker_client()

# Rerank documents
results = client.rerank(
    query="What is machine learning?",
    documents=[
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Deep learning uses neural networks for complex tasks.",
    ],
    top_k=2
)

# Print results (sorted by score, highest first)
for result in results:
    print(f"[{result.index}] Score: {result.score:.4f}")
    print(f"    {result.document}")
```

### Output

```
[0] Score: 0.9234
    Machine learning is a subset of artificial intelligence.
[2] Score: 0.8567
    Deep learning uses neural networks for complex tasks.
```

## Providers

### Infinity Provider

Infinity is a high-throughput, low-latency REST API for serving reranking models locally.

#### Setup

```bash
# Start Infinity with a reranking model
docker run -p 7997:7997 michaelf34/infinity:latest \
  --model-name-or-path BAAI/bge-reranker-v2-m3

# Or install and run locally
pip install infinity-emb[all]
infinity_emb v2 --model-name-or-path BAAI/bge-reranker-v2-m3
```

#### Configuration

```bash
# .env
RERANKER_PROVIDER=infinity
INFINITY_BASE_URL=http://localhost:7997
# Or use dedicated reranker URL (takes precedence)
INFINITY_RERANK_URL=http://reranker:7997
INFINITY_TOKEN=optional-auth-token
INFINITY_TIMEOUT=30
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

#### Usage

```python
from core_lib.reranker import create_infinity_reranker

client = create_infinity_reranker(
    model="BAAI/bge-reranker-v2-m3",
    base_url="http://localhost:7997",
    timeout=30,
    token="optional-auth-token"
)

results = client.rerank(query="query", documents=["doc1", "doc2"])

# Check available models
models = client.get_available_models()
print(f"Available reranking models: {models}")
```

#### Recommended Infinity Models

| Model | Language | Quality | Speed |
|-------|----------|---------|-------|
| `BAAI/bge-reranker-v2-m3` | Multilingual | High | Medium |
| `BAAI/bge-reranker-large` | English | High | Medium |
| `BAAI/bge-reranker-base` | English | Medium | Fast |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | English | Medium | Fast |

### Cohere Provider

Cohere provides high-quality reranking through their cloud API.

#### Configuration

```bash
# .env
RERANKER_PROVIDER=cohere
COHERE_API_KEY=your-api-key
RERANKER_MODEL=rerank-multilingual-v3.0
```

#### Usage

```python
from core_lib.reranker import create_cohere_reranker

client = create_cohere_reranker(
    model="rerank-multilingual-v3.0",
    api_key="your-api-key"
)

results = client.rerank(query="query", documents=["doc1", "doc2"])
```

#### Cohere Models

| Model | Language | Description |
|-------|----------|-------------|
| `rerank-multilingual-v3.0` | 100+ languages | Latest multilingual |
| `rerank-english-v3.0` | English | Latest English-only |
| `rerank-multilingual-v2.0` | Multilingual | Previous generation |
| `rerank-english-v2.0` | English | Previous generation |

### Local Provider

Run cross-encoder models locally using sentence-transformers.

#### Configuration

```bash
# .env
RERANKER_PROVIDER=local
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_DEVICE=auto  # cpu, cuda, auto
RERANKER_CACHE_DIR=/path/to/cache  # Optional model cache
RERANKER_TRUST_REMOTE_CODE=false
```

#### Usage

```python
from core_lib.reranker import create_local_reranker

client = create_local_reranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda",  # or "cpu", "auto"
)

results = client.rerank(query="query", documents=["doc1", "doc2"])
```

#### Recommended Local Models

| Model | Language | Speed | Quality |
|-------|----------|-------|---------|
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | English | Very Fast | Lower |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | English | Fast | Medium |
| `BAAI/bge-reranker-base` | English | Medium | Good |
| `BAAI/bge-reranker-large` | English | Slower | High |
| `BAAI/bge-reranker-v2-m3` | Multilingual | Medium | High |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RERANKER_PROVIDER` | Provider: `infinity`, `cohere`, `local` | `infinity` |
| `RERANKER_MODEL` | Model name | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_TIMEOUT` | Request timeout (seconds) | `30` |
| `RERANKER_CACHE_DURATION_SECONDS` | Cache TTL (0 to disable) | `3600` |
| `RERANKER_DEFAULT_TOP_K` | Default top-k results | `10` |
| `RERANKER_SCORE_THRESHOLD` | Minimum score threshold | None |

#### Infinity-Specific

| Variable | Description | Default |
|----------|-------------|---------|
| `INFINITY_BASE_URL` | Server URL (shared with embeddings) | `http://localhost:7997` |
| `INFINITY_RERANK_URL` | Dedicated reranker URL (overrides base) | None |
| `INFINITY_TOKEN` | Authentication token | None |
| `INFINITY_TIMEOUT` | Request timeout | `30` |

#### Cohere-Specific

| Variable | Description |
|----------|-------------|
| `COHERE_API_KEY` | Cohere API key (required) |

#### Local-Specific

| Variable | Description | Default |
|----------|-------------|---------|
| `RERANKER_DEVICE` | Device: `cpu`, `cuda`, `auto` | `auto` |
| `RERANKER_CACHE_DIR` | Model cache directory | None |
| `RERANKER_TRUST_REMOTE_CODE` | Trust remote code | `false` |

### Programmatic Configuration

```python
from core_lib.reranker import RerankerSettings, RerankerFactory

# Create custom settings
settings = RerankerSettings(
    provider="infinity",
    model="BAAI/bge-reranker-v2-m3",
    infinity_url="http://localhost:7997",
    timeout=60,
    cache_duration_seconds=7200,
)

# Create client from settings
client = RerankerFactory.from_config(settings)
```

## API Reference

### RerankResult

```python
@dataclass
class RerankResult:
    index: int           # Original document index
    score: float         # Relevance score (higher = more relevant)
    document: str | None # Document text (optional)
```

### BaseRerankerClient

```python
class BaseRerankerClient:
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank documents by relevance to query."""
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float, str]]:
        """Rerank and return (index, score, document) tuples."""
    
    def health_check(self) -> bool:
        """Check if provider is available."""
    
    def get_rerank_time_ms(self) -> float:
        """Get last rerank operation time in milliseconds."""
```

### Factory Functions

```python
from core_lib.reranker import (
    create_reranker_client,      # Auto-detect from env
    create_client_from_env,      # Explicit from env
    create_infinity_reranker,    # Infinity provider
    create_cohere_reranker,      # Cohere provider
    create_local_reranker,       # Local provider
    get_reranker_client,         # Legacy alias
    RerankerFactory,             # Factory class
)
```

## Caching

Reranker results are cached automatically using Redis.

### Enable/Disable Caching

```python
# Enable caching (default: 1 hour)
client = create_reranker_client(cache_duration_seconds=3600)

# Disable caching
client = create_reranker_client(cache_duration_seconds=0)

# Custom cache duration (2 hours)
client = create_reranker_client(cache_duration_seconds=7200)
```

### Cache Key Components

Cache keys are generated from:
- Query text
- Document list
- Top-k value
- Model name

Same inputs with same model = cache hit.

### Environment Configuration

```bash
# Cache duration in seconds (default: 3600)
RERANKER_CACHE_DURATION_SECONDS=7200

# Disable caching
RERANKER_CACHE_DURATION_SECONDS=0
```

## RAG Integration

### Complete RAG Pipeline

```python
from core_lib.embeddings import create_embedding_client
from core_lib.reranker import create_reranker_client
from core_lib.llm import create_llm_client

# Initialize clients
embedding_client = create_embedding_client()
reranker_client = create_reranker_client()
llm_client = create_llm_client()

def rag_query(query: str, vector_db, top_k_retrieve: int = 100, top_k_rerank: int = 5):
    """Complete RAG pipeline with reranking."""
    
    # Step 1: Generate query embedding
    query_embedding = embedding_client.generate_embedding(query)
    
    # Step 2: Retrieve candidates from vector database (fast, approximate)
    candidates = vector_db.search(query_embedding, top_k=top_k_retrieve)
    documents = [doc["text"] for doc in candidates]
    
    # Step 3: Rerank candidates (accurate)
    reranked = reranker_client.rerank(
        query=query,
        documents=documents,
        top_k=top_k_rerank
    )
    
    # Step 4: Build context from top reranked documents
    context = "\n\n".join([r.document for r in reranked])
    
    # Step 5: Generate response with LLM
    messages = [
        {"role": "system", "content": f"Answer based on this context:\n\n{context}"},
        {"role": "user", "content": query}
    ]
    
    response = llm_client.chat(messages)
    return response["content"]
```

### Reranking Search Results

```python
def search_with_reranking(query: str, search_results: list) -> list:
    """Rerank search results for better relevance."""
    
    client = create_reranker_client()
    
    # Extract text from search results
    documents = [result["text"] for result in search_results]
    
    # Rerank
    reranked = client.rerank(query=query, documents=documents, top_k=10)
    
    # Map back to original results with scores
    reranked_results = []
    for r in reranked:
        original = search_results[r.index].copy()
        original["rerank_score"] = r.score
        reranked_results.append(original)
    
    return reranked_results
```

## Best Practices

### 1. Limit Document Count

Reranking is O(n) per query-document pair. Limit to top-100 or top-200 candidates.

```python
# Good: Rerank top-100 from embedding search
results = client.rerank(query, documents[:100], top_k=10)

# Bad: Reranking thousands of documents
results = client.rerank(query, all_documents, top_k=10)  # Slow!
```

### 2. Use Appropriate Models

| Use Case | Recommended Model |
|----------|-------------------|
| Production (multilingual) | `BAAI/bge-reranker-v2-m3` |
| Production (English) | `BAAI/bge-reranker-large` |
| Real-time / Low latency | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Highest quality | `rerank-multilingual-v3.0` (Cohere) |

### 3. Enable Caching for Repeated Queries

```python
# Cache popular queries
client = create_reranker_client(cache_duration_seconds=3600)
```

### 4. Use Health Checks

```python
if not client.health_check():
    # Fallback to embedding-only search
    return embedding_search_results
```

### 5. Set Appropriate Timeouts

```python
# Increase timeout for large batches
client = create_infinity_reranker(timeout=60)
```

### 6. Monitor Performance

```python
results = client.rerank(query, documents)
print(f"Reranking took {client.get_rerank_time_ms():.2f}ms")
```

## Troubleshooting

### Connection Errors

```
RerankerError: Failed to connect to Infinity server at http://localhost:7997
```

**Solution:** Ensure Infinity server is running and accessible.

```bash
# Check if server is running
curl http://localhost:7997/health

# Start server if needed
docker run -p 7997:7997 michaelf34/infinity:latest \
  --model-name-or-path BAAI/bge-reranker-v2-m3
```

### Timeout Errors

```
RerankerError: Infinity rerank request timed out after 30s
```

**Solution:** Increase timeout or reduce document count.

```python
client = create_infinity_reranker(timeout=60)
# Or reduce documents
results = client.rerank(query, documents[:50])
```

### Missing Dependencies

```
ImportError: cohere is required for CohereRerankerClient
```

**Solution:** Install the required package.

```bash
pip install cohere
```

### Model Not Found

```
RerankerError: Failed to load cross-encoder model
```

**Solution:** Check model name and internet connectivity.

```python
# Verify model exists on HuggingFace
# https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
```

### API Key Errors

```
ValueError: Cohere API key is required
```

**Solution:** Set the API key.

```bash
export COHERE_API_KEY=your-api-key
```

### Cache Issues

If you suspect stale cache results:

```python
# Disable caching temporarily
client = create_reranker_client(cache_duration_seconds=0)

# Or clear Redis cache manually
from core_lib.cache import get_cache
cache = get_cache()
# Clear rerank keys (pattern: rerank:*)
```

## See Also

- [RERANKER_QUICK_REFERENCE.md](./RERANKER_QUICK_REFERENCE.md) - Quick start guide
- [EMBEDDINGS_GUIDE.md](./EMBEDDINGS_GUIDE.md) - Embeddings documentation
- [INFINITY_QUICKSTART.md](./INFINITY_QUICKSTART.md) - Infinity server setup
- [cache.md](./cache.md) - Caching documentation

````
