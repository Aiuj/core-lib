# Infinity Multi-Server Failover Implementation

## Overview

Implemented a unified failover mechanism for both Infinity embeddings and reranking services, eliminating code duplication and enabling high-availability configurations.

## Architecture Changes

### 1. Shared Infinity API Client

Created `core_lib/api_utils/infinity_api.py` - a reusable HTTP client with multi-URL failover:

```python
from core_lib.api_utils import InfinityAPIClient

client = InfinityAPIClient(
    base_urls="http://server1:7997,http://server2:7997,http://server3:7997",
    timeout=30
)

# Automatically retries on backup servers if primary fails
response, used_url = client.post('/embeddings', json={...})
```

**Features:**
- Comma-separated URL support
- Automatic failover on connection/timeout errors
- Tracks current working URL to prefer it on next request
- Configurable retries per URL
- Unified error handling

### 2. Refactored Infinity Providers

Both `InfinityEmbeddingClient` and `InfinityRerankerClient` now use the shared API client:

**Before:**
```python
# Duplicated requests code in both providers
response = requests.post(f"{self.base_url}/embeddings", ...)
```

**After:**
```python
# Shared API client with built-in failover
data, used_url = self._api_client.post('/embeddings', json=request_body)
```

### 3. FallbackRerankerClient

Created reranker equivalent of `FallbackEmbeddingClient` for provider-level failover:

```python
from core_lib.reranker import FallbackRerankerClient

# Multiple providers with different backends
client = FallbackRerankerClient.from_config([
    {"provider": "infinity", "base_url": "http://local:7997"},
    {"provider": "cohere", "api_key": "..."},  # Cloud fallback
])
```

**Features:**
- Health tracking with Redis caching
- Automatic provider switching on failure
- Smart overload vs. permanent failure detection
- Configurable retry strategies

### 4. Auto-Detection Factory Function

New convenience function automatically creates fallback clients when multiple URLs detected:

```python
from core_lib.reranker import create_reranker_from_env_with_fallback

# Automatically detects comma-separated URLs and creates FallbackRerankerClient
client = create_reranker_from_env_with_fallback()
```

## Configuration

### Environment Variables

```bash
# Single URL - creates standard InfinityRerankerClient/InfinityEmbeddingClient
INFINITY_BASE_URL=http://localhost:7997

# Multiple URLs - creates automatic failover clients
INFINITY_BASE_URL=http://server1:7997,http://server2:7997,http://server3:7997

# Both embeddings and reranking use the same configuration
EMBEDDING_PROVIDER=infinity
RERANKER_PROVIDER=infinity
```

### mcp-doc-qa Integration

Updated `src/retrieval/reranker_service.py` to use auto-failover:

```python
# Before
base_url = os.getenv("INFINITY_BASE_URL")
if base_url:
    self._client = create_infinity_reranker(base_url=base_url)

# After
self._client = create_reranker_from_env_with_fallback()
```

## Benefits

1. **Code Reuse**: Single API client implementation for both embeddings and reranking
2. **High Availability**: Automatic failover between multiple servers
3. **Transparent**: Existing code works without changes
4. **Flexible**: Supports mixing providers (local + cloud fallback)
5. **Resilient**: Detects temporary overload vs. permanent failures

## Usage Examples

### Simple Multi-Host Setup

```bash
# .env
INFINITY_BASE_URL=http://192.168.1.100:7997,http://192.168.1.101:7997
```

```python
from core_lib.embeddings import create_embedding_client
from core_lib.reranker import create_reranker_from_env_with_fallback

# Both automatically use all listed servers with failover
embeddings = create_embedding_client()
reranker = create_reranker_from_env_with_fallback()
```

### Advanced Multi-Provider Setup

```python
from core_lib.reranker import FallbackRerankerClient

client = FallbackRerankerClient.from_config([
    # Primary: Local Infinity server
    {"provider": "infinity", "base_url": "http://localhost:7997"},
    
    # Backup: Second Infinity server
    {"provider": "infinity", "base_url": "http://backup:7997"},
    
    # Fallback: Cloud Cohere API (always available)
    {"provider": "cohere", "api_key": "your-api-key"},
], max_retries_per_provider=2)

# Automatically tries each provider in order until one succeeds
results = client.rerank(query="...", documents=[...])
```

### Manual Infinity API Client

```python
from core_lib.api_utils import InfinityAPIClient, InfinityAPIError

client = InfinityAPIClient(
    base_urls=["http://server1:7997", "http://server2:7997"],
    timeout=60,
    token="optional-auth-token"
)

try:
    data, used_url = client.post('/rerank', json={
        'model': 'BAAI/bge-reranker-v2-m3',
        'query': 'search query',
        'documents': ['doc1', 'doc2'],
        'top_n': 5
    })
    print(f"Succeeded using {used_url}")
except InfinityAPIError as e:
    print(f"All servers failed: {e}")

# Check status of all URLs
for status in client.get_url_status():
    print(f"URL {status['url']}: {status['failures']} failures")
```

## Error Resolution

### Original Error

```
Failed to parse: http://powerspec:7997,http://127.0.0.1:7997/rerank
```

**Cause**: Reranker tried to use comma-separated string as single URL

**Fix**: `InfinityAPIClient` properly parses and handles multiple URLs

### Verification

```python
from core_lib.reranker import create_reranker_from_env_with_fallback

client = create_reranker_from_env_with_fallback()

# Check if failover is active
if hasattr(client, 'get_provider_status'):
    for provider in client.get_provider_status():
        print(f"Provider {provider['index']}: {provider['urls']}")
```

## Files Modified

### core-lib
- `core_lib/api_utils/infinity_api.py` (NEW) - Shared API client
- `core_lib/api_utils/__init__.py` - Export API client
- `core_lib/embeddings/infinity_provider.py` - Use shared client
- `core_lib/reranker/infinity_provider.py` - Use shared client
- `core_lib/reranker/fallback_client.py` (NEW) - Fallback reranker
- `core_lib/reranker/factory.py` - Added failover factory functions
- `core_lib/reranker/__init__.py` - Export new classes/functions

### mcp-doc-qa
- `src/retrieval/reranker_service.py` - Use auto-failover initialization
- `.env` - Updated comments, restored multi-URL configuration

## Testing

```bash
# Test multi-URL configuration
cd core-lib
export INFINITY_BASE_URL=http://server1:7997,http://server2:7997
uv run python -c "
from core_lib.reranker import create_reranker_from_env_with_fallback
client = create_reranker_from_env_with_fallback()
print(f'Client type: {type(client).__name__}')
if hasattr(client, 'providers'):
    print(f'Providers: {len(client.providers)}')
"
```

## Migration Guide

### For Library Developers

**Old approach (manual URL handling):**
```python
import os
base_url = os.getenv("INFINITY_BASE_URL")
client = create_infinity_reranker(base_url=base_url)
```

**New approach (automatic failover):**
```python
from core_lib.reranker import create_reranker_from_env_with_fallback
client = create_reranker_from_env_with_fallback()
```

### For End Users

No code changes required - just update `.env`:

```bash
# Old (single URL)
INFINITY_BASE_URL=http://localhost:7997

# New (multi-URL for HA)
INFINITY_BASE_URL=http://server1:7997,http://server2:7997
```
