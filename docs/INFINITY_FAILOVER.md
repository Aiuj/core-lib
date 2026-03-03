# Infinity Failover

Use this guide to run Infinity embeddings/reranking with high availability.
This guide focuses on practical setup, validation, and troubleshooting.

## What it does

When Infinity is configured with multiple URLs, clients automatically:
- try the highest-priority/first URL
- fail over to the next URL on timeout/connection errors
- continue using a healthy URL until another failure occurs

This applies to both embeddings and reranking flows.

## Minimal configuration

Single URL (no failover):

```dotenv
INFINITY_BASE_URL=http://localhost:7997
```

Multi-URL failover:

```dotenv
INFINITY_BASE_URL=http://server1:7997,http://server2:7997,http://server3:7997
EMBEDDING_PROVIDER=infinity
RERANKER_PROVIDER=infinity
```

## Typical usage

Embeddings:

```python
from core_lib.embeddings import create_embedding_client

client = create_embedding_client()
```

Reranker (auto-fallback aware factory):

```python
from core_lib.reranker import create_reranker_from_env_with_fallback

client = create_reranker_from_env_with_fallback()
```

No special call pattern is needed after configuration.

## With Wake-on-LAN

Failover works well with sleeping primary hosts:
- primary times out
- WoL can wake it
- traffic is served by secondary during wake/warmup

See the WoL setup guide for full examples and timing controls.

## Quick validation

Use a temporary env override and verify client type/behavior:

```bash
uv run python -c "from core_lib.reranker import create_reranker_from_env_with_fallback; c=create_reranker_from_env_with_fallback(); print(type(c).__name__)"
```

Recommended tests:

```bash
uv run pytest -q tests/test_vector_provider_yaml_config.py
uv run pytest -q tests/test_infinity_wake_on_lan.py
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Failed to parse` with comma-separated URL string | URL list treated as one raw URL by custom code path | Use `INFINITY_BASE_URL` and standard factory methods |
| No failover observed | Only one URL configured | Add at least two URLs in `INFINITY_BASE_URL` |
| Frequent switching/instability | Intermittent network or low timeout | Increase timeout and stabilize network path |
| Secondary works but primary never returns | Primary remains unavailable | Check primary health, DNS, firewall, and service logs |

## Related docs

- [docs/INFINITY_PROVIDER.md](INFINITY_PROVIDER.md)
- [docs/WAKE_ON_LAN.md](WAKE_ON_LAN.md)
- [docs/EMBEDDINGS_QUICK_REFERENCE.md](EMBEDDINGS_QUICK_REFERENCE.md)
- [docs/FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md)