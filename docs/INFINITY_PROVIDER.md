# Infinity Embedding Provider

Use Infinity as a local embedding backend for `core-lib`.
This guide focuses on practical setup, validation, and troubleshooting.

## What it does

Infinity is an OpenAI-compatible embedding server (default port `7997`) that works with `core-lib` embedding and reranking flows.

## Minimal setup

### 1) Run Infinity

```bash
docker run -d --name infinity -p 7997:7997 \
  michaelf34/infinity:latest \
  --model-name-or-path BAAI/bge-small-en-v1.5
```

### 2) Configure environment

```dotenv
EMBEDDING_PROVIDER=infinity
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384

# Prefer provider-specific setting for Infinity
INFINITY_BASE_URL=http://localhost:7997
INFINITY_TIMEOUT=30
```

## Basic usage

```python
from core_lib.embeddings import create_embedding_client

client = create_embedding_client()
vec = client.generate_embedding("Hello world")
```

Batching is preferred for throughput:

```python
vectors = client.generate_embedding(["text 1", "text 2", "text 3"])
```

## High availability (multi-URL failover)

Use comma-separated Infinity URLs:

```dotenv
EMBEDDING_PROVIDER=infinity
INFINITY_BASE_URL=http://infinity1:7997,http://infinity2:7997
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384
INFINITY_TIMEOUT=30
```

With this config, `create_embedding_client()` automatically uses failover behavior.

Optional per-host tokens (matched by URL position):

```dotenv
INFINITY_TOKEN=token-host1,token-host2
```

Or a shared token:

```dotenv
INFINITY_TOKEN=shared-token
```

## With Wake-on-LAN

If your primary Infinity host can sleep, pair multi-URL failover with WoL:
- request hits sleeping primary
- WoL packet is sent
- secondary handles traffic until primary is available again

See WoL docs for exact `wake_on_lan` settings.

## Quick validation

Server:

```bash
curl http://localhost:7997/health
```

Client:

```python
from core_lib.embeddings import create_infinity_client

client = create_infinity_client()
print(client.health_check())
print(client.get_available_models())
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Connection refused | Infinity not running / wrong host-port | Start container, verify URL and network |
| HTTP 404 on embeddings | Model not loaded on server | Start Infinity with `--model-name-or-path` |
| Timeout errors | Model cold start / CPU-only load | Increase timeout, enable GPU, reduce batch size |
| No failover | Only one URL configured | Add multiple URLs to `INFINITY_BASE_URL` |
| Wrong vector size | `EMBEDDING_DIMENSION` mismatch | Align dimension with selected model |

## Related docs

- [docs/INFINITY_FAILOVER.md](INFINITY_FAILOVER.md)
- [docs/WAKE_ON_LAN.md](WAKE_ON_LAN.md)
- [docs/EMBEDDINGS_QUICK_REFERENCE.md](EMBEDDINGS_QUICK_REFERENCE.md)