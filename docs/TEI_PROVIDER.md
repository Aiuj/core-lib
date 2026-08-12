# Hugging Face TEI providers

`core-lib` supports Hugging Face Text Embeddings Inference (TEI) as a
first-class provider for both embeddings and reranking.

Configure TEI services in the shared provider file:

```yaml
embedding_providers:
  - provider: tei
    enabled: true
    model: Qwen/Qwen3-Embedding-0.6B
    base_url: http://192.168.1.204:8110
    priority: 10

reranker_providers:
  - provider: tei
    enabled: true
    model: Alibaba-NLP/gte-multilingual-reranker-base
    base_url: http://192.168.1.204:8111
    priority: 10
```

The embedding client calls TEI's OpenAI-compatible `POST /v1/embeddings`
endpoint. The reranker client calls TEI's native `POST /rerank` endpoint and
handles its `texts` request field and array response format.

Single-provider environment variables are also supported:

```env
EMBEDDING_PROVIDER=tei
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
TEI_BASE_URL=http://192.168.1.204:8110

RERANKER_PROVIDER=tei
RERANKER_MODEL=Alibaba-NLP/gte-multilingual-reranker-base
TEI_RERANK_URL=http://192.168.1.204:8111
```

Use `TEI_TOKEN` when the TEI endpoints require bearer authentication.

## Non-blocking Wake-on-LAN fallback

Set `warmup_seconds` on the primary TEI embedding provider and put DeepInfra
next in the priority chain. When the short initial TEI connection attempt
fails, core-lib sends the WoL packet and immediately executes the same request
against DeepInfra. It does not sleep or retry TEI while the warm-up window is
active.

```yaml
embedding_providers:
  - provider: tei
    enabled: true
    model: Qwen/Qwen3-Embedding-0.6B
    base_url: http://192.168.1.204:8110
    priority: 10
    wake_on_lan:
      enabled: true
      initial_timeout_seconds: 0.5
      warmup_seconds: 90
      broadcast_ip: 192.168.1.255
      mac_address: FC:34:97:9E:C8:AF
      port: 9

  - provider: deepinfra
    enabled: true
    model: Qwen/Qwen3-Embedding-0.6B
    priority: 20
    api_key: ${DEEPINFRA_API_KEY}
```

`initial_timeout_seconds` is the only latency paid before failover when the
machine is asleep. Keep it low but high enough for normal LAN connection
variance; `0.5` seconds is a practical starting point. Subsequent requests
during the 90-second warm-up window skip TEI immediately.

The same non-blocking behavior applies to `reranker_providers`: configure
`wake_on_lan` with `warmup_seconds` on the TEI primary and place the cloud
reranker second in priority order. In warm-up mode, `wait_seconds` and
`retry_timeout_seconds` are intentionally not used.

## Shared behavior across service types

LLM, embedding, and reranker fallback clients all use the same
`WakeOnLanStrategy` and `WarmupFallbackRouter`. This gives each service type
the same contract:

1. Attempt the primary for `initial_timeout_seconds`.
2. Send WoL when the primary is unreachable.
3. With `warmup_seconds`, do not sleep or perform the blocking retry.
4. Route the current and subsequent requests to the next provider.
5. Do not mark the waking primary unhealthy.
6. Retry the priority primary as soon as warm-up expires.

Without `warmup_seconds`, all three retain blocking mode: `wait_seconds` is
used before retrying the primary with `retry_timeout_seconds`.
