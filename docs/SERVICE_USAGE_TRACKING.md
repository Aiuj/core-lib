# Service Usage Tracking with OpenTelemetry

This document explains how core-lib automatically tracks AI service usage (LLM, embeddings, OCR) and sends detailed metrics to OpenSearch via OpenTelemetry, eliminating the need for complex Langfuse span management.

## Overview

The library automatically logs structured service usage events to OpenTelemetry/OpenSearch whenever you use:
- **LLM services** (OpenAI, Google Gemini, Ollama)
- **Embedding services** (OpenAI, Infinity, Ollama, Google GenAI)
- **OCR services** (Azure Document Intelligence, Tesseract, etc.)

These events include (deduped field set):
- Service type, provider, and model
- Token usage (`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `tokens.total`)
- Performance metrics (`gen_ai.response.latency_ms`, `tokens_per_second`)
- **Automatic cost calculation** (`gen_ai.usage.cost`)
- Request context (user_id, session_id, company_id from `LoggingContext`)
- Feature flags (structured output, tools, search grounding)
- Host and system identifiers (`gen_ai.host`, `gen_ai.system`) for multi-provider dashboards

## Field Reference (LLM logs)

Each LLM event emitted by `log_llm_usage`/providers includes these standard attributes for dashboards:

- `service.type`: always `llm`
- `service.provider`: provider slug (`openai`, `google-gemini`, `ollama`)
- `service.model` / `gen_ai.request.model`: model/deployment name
- `gen_ai.system`: provider identifier (aligns with OTel semantic conventions)
- `gen_ai.host`: base URL/endpoint (when available)
- Tokens: `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `tokens.total` (computed when possible)
- Performance: `gen_ai.response.latency_ms`, `tokens_per_second` (when total tokens and latency are known)
- Cost: `gen_ai.usage.cost`
- Feature flags: `features.structured_output`, `features.tools`, `features.search_grounding`
- Status/error: `status` (`success`/`error`), `error` message when present
- Context (if `LoggingContext` is set): `user.id`, `session.id`, `organization.id`, etc.

Latency measurement: `gen_ai.response.latency_ms` is the wall-clock time spent waiting for the model API call (request → response) as measured by the client; `tokens_per_second` is derived from `tokens.total` and `gen_ai.response.latency_ms` when both are available. Duplicate cost/token aliases have been removed; use the `gen_ai.*` fields above.

## No Span Management Required

Unlike Langfuse, you don't need to manage spans or worry about context errors like:
```
Context error: No active span in current context...
```

The service usage tracking uses standard Python logging infrastructure, which means:
- ✅ No span creation or context management
- ✅ Automatic integration with existing OTLP logging
- ✅ Works seamlessly with async and multi-threaded code
- ✅ All metadata from `LoggingContext` is automatically included

## Setup

### 1. Enable OTLP Logging

Configure your logger settings to enable OTLP:

```python
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing import setup_logging

# Configure OTLP
logger_settings = LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",  # Your OpenSearch collector
    otlp_service_name="my-app",
    otlp_service_version="1.0.0",
    log_level="INFO",
)

# Initialize logging
setup_logging(logger_settings=logger_settings)
```

Or use environment variables:
```bash
OTLP_ENABLED=true
OTLP_ENDPOINT=http://localhost:4318/v1/logs
OTLP_SERVICE_NAME=my-app
LOG_LEVEL=INFO
```

### 2. Optional: Add Request Context

Use `LoggingContext` to automatically add user/session/company metadata:

```python
from core_lib.tracing import LoggingContext, parse_from

@app.post("/api/chat")
async def chat_endpoint(from_: Optional[str] = Query(None, alias="from")):
    from_dict = parse_from(from_)
    
    # All logs within this context will include user_id, session_id, etc.
    with LoggingContext(from_dict):
        response = await llm_client.chat(messages=[...])
        # Service usage automatically logged with context!
    
    return response
```

## What Gets Logged

### LLM Usage Events

Every LLM request automatically logs:

```json
{
  "message": "LLM usage: openai/gpt-4 - 150 tokens, $0.004500",
  "severity": "INFO",
  "attributes": {
    "service.type": "llm",
    "service.provider": "openai",
    "service.model": "gpt-4",
    "gen_ai.request.model": "gpt-4",
    "gen_ai.system": "openai",
    "gen_ai.usage.input_tokens": 100,
    "gen_ai.usage.output_tokens": 50,
    "tokens.total": 150,
    "tokens_per_second": 100.0,
    "gen_ai.response.latency_ms": 1500,
    "gen_ai.usage.cost": 0.0045,
    "features.structured_output": false,
    "features.tools": true,
    "features.search_grounding": false,
    "status": "success",
    
    // From LoggingContext (if set)
    "session.id": "sess-123",
    "user.id": "user-456",
    "organization.id": "org-789"
  }
}
```

### Embedding Usage Events

Every embedding request automatically logs:

```json
{
  "message": "Embedding usage: openai/text-embedding-3-small - 10 texts, 500 tokens, $0.000010",
  "severity": "INFO",
  "attributes": {
    "service.type": "embedding",
    "service.provider": "openai",
    "service.model": "text-embedding-3-small",
    "gen_ai.request.model": "text-embedding-3-small",
    "gen_ai.system": "openai",
    "tokens.input": 500,
    "gen_ai.usage.input_tokens": 500,
    "embedding.num_texts": 10,
    "embedding.dimension": 1536,
    "latency_ms": 250,
    "texts_per_second": 40.0,
    "cost_usd": 0.00001,
    "status": "success"
  }
}
```

### OCR Usage Events

For OCR services:

```json
{
  "message": "OCR usage: azure-di/prebuilt-read - 5 pages, $0.050000",
  "severity": "INFO",
  "attributes": {
    "service.type": "ocr",
    "service.provider": "azure-di",
    "service.model": "prebuilt-read",
    "ocr.num_pages": 5,
    "latency_ms": 3000,
    "pages_per_second": 1.67,
    "cost_usd": 0.05,
    "status": "success"
  }
}
```

## Querying in OpenSearch

### Total Cost by Service

```json
GET /logs-*/_search
{
  "size": 0,
  "query": {
    "bool": {
      "must": [
        {"term": {"attributes.service.type": "llm"}},
        {"range": {"@timestamp": {"gte": "now-7d"}}}
      ]
    }
  },
  "aggs": {
    "total_cost": {"sum": {"field": "attributes.cost_usd"}},
    "by_model": {
      "terms": {"field": "attributes.service.model"},
      "aggs": {
        "cost": {"sum": {"field": "attributes.cost_usd"}},
        "tokens": {"sum": {"field": "attributes.tokens.total"}}
      }
    }
  }
}
```

### Usage by User

```json
GET /logs-*/_search
{
  "size": 0,
  "query": {
    "bool": {
      "must": [
        {"exists": {"field": "attributes.user.id"}},
        {"range": {"@timestamp": {"gte": "now-24h"}}}
      ]
    }
  },
  "aggs": {
    "by_user": {
      "terms": {"field": "attributes.user.id"},
      "aggs": {
        "total_cost": {"sum": {"field": "attributes.cost_usd"}},
        "total_tokens": {"sum": {"field": "attributes.tokens.total"}},
        "llm_requests": {
          "filter": {"term": {"attributes.service.type": "llm"}},
          "aggs": {"count": {"value_count": {"field": "_id"}}}
        }
      }
    }
  }
}
```

### Performance Metrics

```json
GET /logs-*/_search
{
  "size": 0,
  "query": {
    "term": {"attributes.service.type": "llm"}
  },
  "aggs": {
    "latency_stats": {
      "stats": {"field": "attributes.latency_ms"}
    },
    "tokens_per_second_stats": {
      "stats": {"field": "attributes.tokens_per_second"}
    },
    "by_provider": {
      "terms": {"field": "attributes.service.provider"},
      "aggs": {
        "avg_latency": {"avg": {"field": "attributes.latency_ms"}},
        "p95_latency": {
          "percentiles": {
            "field": "attributes.latency_ms",
            "percents": [95]
          }
        }
      }
    }
  }
}
```

### Error Analysis

```json
GET /logs-*/_search
{
  "query": {
    "bool": {
      "must": [
        {"term": {"attributes.status": "error"}},
        {"exists": {"field": "attributes.service.type"}}
      ]
    }
  },
  "aggs": {
    "error_by_service": {
      "terms": {"field": "attributes.service.type"}
    },
    "error_by_provider": {
      "terms": {"field": "attributes.service.provider"}
    }
  }
}
```

## OpenSearch Dashboard Visualizations

### 1. Cost Dashboard

Create a dashboard with:
- **Total Cost (24h)**: Sum of `attributes.cost_usd` filtered by last 24 hours
- **Cost by Service Type**: Pie chart grouping by `attributes.service.type`
- **Cost Trend**: Line chart of `attributes.cost_usd` over time
- **Top 5 Models by Cost**: Bar chart of `attributes.service.model` by sum of cost

### 2. Usage Dashboard

Create visualizations for:
- **Requests per Minute**: Count of documents over time
- **Tokens per Minute**: Sum of `attributes.tokens.total` over time
- **Requests by User**: Table of `attributes.user.id` with count and total cost
- **Performance Heatmap**: `attributes.latency_ms` by provider and model

### 3. Alerts

Set up alerts for:
- **High Cost**: When hourly cost exceeds threshold
- **High Error Rate**: When error rate > 5% in 5 minutes
- **Slow Requests**: When p95 latency > threshold
- **Unusual Usage**: Anomaly detection on token usage

## Cost Calculation

The library includes built-in pricing for common models in `core_lib/tracing/service_pricing.py`.

### LLM Pricing (per 1K tokens)

| Model | Input | Output |
|-------|--------|--------|
| gpt-4 | $0.03 | $0.06 |
| gpt-4-turbo | $0.01 | $0.03 |
| gpt-4o | $0.005 | $0.015 |
| gpt-4o-mini | $0.00015 | $0.0006 |
| gemini-1.5-pro | $0.00125 | $0.005 |
| gemini-1.5-flash | $0.000075 | $0.0003 |
| claude-3-opus | $0.015 | $0.075 |
| claude-3-sonnet | $0.003 | $0.015 |

### Embedding Pricing (per 1K tokens)

| Model | Price |
|-------|-------|
| text-embedding-3-small | $0.00002 |
| text-embedding-3-large | $0.00013 |
| text-embedding-ada-002 | $0.0001 |
| text-embedding-004 (Gemini) | $0.00001 |

### Updating Pricing

To update or add pricing, edit `core_lib/tracing/service_pricing.py`:

```python
# Add new LLM model pricing
LLM_PRICING["new-model"] = {"input": 0.001, "output": 0.002}

# Add new embedding model pricing
EMBEDDING_PRICING["new-embedding-model"] = 0.00005

# Add OCR pricing
OCR_PRICING["provider/model"] = {"per_page": 0.01}
```

The pricing data is centralized in this single file for easy maintenance.

## Advanced: Manual Logging

If you build custom integrations, you can manually log usage:

```python
from core_lib.tracing.service_usage import log_llm_usage, log_embedding_usage

# Log custom LLM usage
log_llm_usage(
    provider="custom-provider",
    model="custom-model",
    input_tokens=100,
    output_tokens=50,
    latency_ms=1500,
    metadata={"endpoint": "/api/custom"}
)

# Log custom embedding usage
log_embedding_usage(
    provider="custom-embeddings",
    model="custom-model",
    input_tokens=500,
    num_texts=10,
    embedding_dim=768,
    latency_ms=250
)
```

## Benefits Over Langfuse

| Feature | OpenTelemetry/OpenSearch | Langfuse |
|---------|--------------------------|----------|
| Span management | ❌ Not required | ✅ Required |
| Context errors | ❌ None | ⚠️ Common |
| Async support | ✅ Native | ⚠️ Complex |
| Query flexibility | ✅ Full SQL/DSL | ⚠️ Limited UI |
| Custom dashboards | ✅ Full control | ⚠️ Fixed UI |
| Cost tracking | ✅ Built-in | ⚠️ Manual |
| Multi-service | ✅ Unified logs | ⚠️ LLM-focused |
| Data retention | ✅ Self-hosted control | ⚠️ Service limits |

## Troubleshooting

### Not seeing service usage logs?

1. Check OTLP is enabled:
   ```python
   from core_lib.tracing import get_last_logging_config
   config = get_last_logging_config()
   print(config.get("otlp_enabled"))  # Should be True
   ```

2. Verify endpoint is reachable:
   ```bash
   curl http://localhost:4318/v1/logs
   ```

3. Check log level allows INFO:
   ```python
   import logging
   logging.getLogger().setLevel(logging.INFO)
   ```

### Missing cost data?

Cost is automatically calculated for known models. If `cost_usd` is 0:
- Model name might not match pricing database
- Add custom pricing in `service_usage.py`
- Or use `cost_override` parameter for manual logging

### Missing user context?

User/session data comes from `LoggingContext`:
```python
from core_lib.tracing import LoggingContext

with LoggingContext({"user_id": "user-123", "session_id": "sess-456"}):
    # All service usage here will include user context
    response = llm_client.chat(...)
```

## Example Integration

Complete example showing service usage tracking:

```python
from core_lib.config.logger_settings import LoggerSettings
from core_lib.tracing import setup_logging, LoggingContext
from core_lib.llm import create_client_from_env

# 1. Setup logging with OTLP
logger_settings = LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
    otlp_service_name="my-chat-app",
    log_level="INFO",
)
setup_logging(logger_settings=logger_settings)

# 2. Create LLM client
llm_client = create_client_from_env()

# 3. Use with context
async def process_request(user_id: str, session_id: str, prompt: str):
    context = {
        "user_id": user_id,
        "session_id": session_id,
        "company_id": "acme-corp"
    }
    
    with LoggingContext(context):
        # This automatically logs to OpenSearch with:
        # - Token usage
        # - Cost
        # - Latency
        # - User context
        response = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response
```

Now check OpenSearch dashboards for detailed usage analytics!
