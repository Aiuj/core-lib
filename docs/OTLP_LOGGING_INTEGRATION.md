# OpenTelemetry Protocol (OTLP) Logging Integration

## Overview

core-lib includes native support for sending logs to OpenTelemetry collectors via the OTLP (OpenTelemetry Protocol) over HTTP. This enables integration with observability platforms like OpenSearch, Elasticsearch, Grafana, Datadog, and cloud-native monitoring solutions.

## Architecture

```
┌──────────────────┐     OTLP/HTTP      ┌────────────────────┐
│  Your Application │ ─────────────────> │ OTLP Collector     │
│  (core-lib)       │   Port 4318        │ (otel-collector)   │
└──────────────────┘                     └────────────────────┘
                                                    │
                                                    ├──> OpenSearch
                                                    ├──> Prometheus
                                                    ├──> Grafana Loki
                                                    └──> Cloud Platforms
```

### Components

1. **OTLPHandler** (`core_lib/tracing/handlers/otlp_handler.py`)
   - Converts Python logging records to OTLP format
   - Batches logs for efficient transmission
   - Runs in background thread (non-blocking)
   - Automatic retry and error handling

2. **LoggerSettings** (`core_lib/config/logger_settings.py`)
   - Configuration for OTLP endpoint, headers, timeouts
   - Environment variable support
   - Integration with StandardSettings

3. **Lazy Loading** (`core_lib/tracing/logger.py`)
   - OTLP handler only imported when enabled
   - Zero overhead when disabled
   - Supports multiple handlers simultaneously (file + OTLP + GELF)

## Quick Start

### 1. Basic Setup

```python
from core_lib.config import LoggerSettings
from core_lib.tracing.logger import setup_logging

# Configure OTLP logging
logger_settings = LoggerSettings(
    log_level="INFO",
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
    otlp_service_name="my-service",
)

# Initialize logging
logger = setup_logging(logger_settings=logger_settings)

# Send logs
logger.info("Application started")
logger.error("Error occurred", extra={"error_code": "E001"})
```

### 2. Environment Variables

```bash
# Enable OTLP
export OTLP_ENABLED=true
export OTLP_ENDPOINT=http://localhost:4318/v1/logs
export OTLP_SERVICE_NAME=my-service
export OTLP_SERVICE_VERSION=1.0.0

# Optional: Authentication
export OTLP_HEADERS='{"Authorization": "Bearer token123"}'

# Optional: Advanced settings
export OTLP_TIMEOUT=15
export OTLP_INSECURE=false  # Skip SSL verification
```

```python
from core_lib.config import LoggerSettings
from core_lib.tracing.logger import setup_logging

# Load from environment
settings = LoggerSettings.from_env()
logger = setup_logging(logger_settings=settings)

logger.info("Configured from environment")
```

### 3. With StandardSettings

```python
from core_lib.config import StandardSettings
from core_lib.tracing.logger import setup_logging

# StandardSettings automatically includes logger config
settings = StandardSettings.from_env()

logger = setup_logging(
    app_settings=settings,
    logger_settings=settings.logger_safe,
)

logger.info("Integrated configuration")
```

## Configuration Reference

### LoggerSettings Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `otlp_enabled` | bool | `False` | Enable OTLP logging |
| `otlp_endpoint` | str | `http://localhost:4318/v1/logs` | OTLP collector endpoint |
| `otlp_headers` | dict | `{}` | HTTP headers (e.g., authentication) |
| `otlp_timeout` | int | `10` | Request timeout in seconds |
| `otlp_insecure` | bool | `False` | Skip SSL certificate verification |
| `otlp_service_name` | str | `core-lib` | Service name for resource attributes |
| `otlp_service_version` | str | `None` | Optional service version |

### Environment Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `OTLP_ENABLED` | `true` | Enable OTLP logging |
| `OTLP_ENDPOINT` | `http://otel:4318/v1/logs` | Collector endpoint URL |
| `OTLP_HEADERS` | `{"X-Key": "value"}` | JSON string of headers |
| `OTLP_TIMEOUT` | `15` | Request timeout (seconds) |
| `OTLP_INSECURE` | `true` | Skip SSL verification |
| `OTLP_SERVICE_NAME` | `my-app` | Service identifier |
| `OTLP_SERVICE_VERSION` | `1.0.0` | Service version |

## OpenTelemetry Collector Setup

### Using Your Provided Configuration

Your `otel-collector-config.yml` is already properly configured:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318  # ← core-lib sends logs here

exporters:
  opensearch:
    http:
      endpoint: http://opensearch:9200
      auth:
        authenticator: basicauth/opensearch
    logs_index: "otel-logs"  # ← Logs stored in this index

service:
  pipelines:
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [opensearch]  # ← Logs go to OpenSearch
```

### Point core-lib to Your Collector

```python
# If collector is on localhost
logger_settings = LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
    otlp_service_name="my-app",
)

# If collector is in Docker network
logger_settings = LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="http://otel-collector:4318/v1/logs",
    otlp_service_name="my-app",
)
```

### Docker Compose Integration

Add your app to the same Docker Compose stack:

```yaml
services:
  # Your existing services: otel-collector, opensearch, etc.
  
  my-app:
    build: .
    environment:
      - OTLP_ENABLED=true
      - OTLP_ENDPOINT=http://otel-collector:4318/v1/logs
      - OTLP_SERVICE_NAME=my-application
      - OTLP_SERVICE_VERSION=1.0.0
    depends_on:
      - otel-collector
```

## OTLP Log Format

core-lib sends logs in OTLP format:

```json
{
  "resourceLogs": [
    {
      "resource": {
        "attributes": [
          {"key": "service.name", "value": {"stringValue": "my-service"}},
          {"key": "service.version", "value": {"stringValue": "1.0.0"}}
        ]
      },
      "scopeLogs": [
        {
          "scope": {"name": "core-lib-logger"},
          "logRecords": [
            {
              "timeUnixNano": "1698249600000000000",
              "severityNumber": 9,
              "severityText": "INFO",
              "body": {"stringValue": "Application started"},
              "attributes": [
                {"key": "logger.name", "value": {"stringValue": "my_app"}},
                {"key": "source.file", "value": {"stringValue": "/app/main.py"}},
                {"key": "source.line", "value": {"intValue": "42"}},
                {"key": "source.function", "value": {"stringValue": "main"}}
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

## Viewing Logs in OpenSearch

### 1. OpenSearch Dashboards

Access at `http://localhost:5601` (from your docker-compose)

1. Go to **Discover**
2. Create index pattern: `otel-logs*`
3. Select `@timestamp` as time field
4. View logs with filters and queries

### 2. Query Examples

```bash
# View recent logs
curl -X GET "localhost:9200/otel-logs/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {"match_all": {}},
  "sort": [{"@timestamp": "desc"}],
  "size": 10
}
'

# Search by service name
curl -X GET "localhost:9200/otel-logs/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {
    "term": {"resource.attributes.service.name": "my-service"}
  }
}
'

# Search ERROR logs
curl -X GET "localhost:9200/otel-logs/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {
    "term": {"severityText": "ERROR"}
  }
}
'
```

## Advanced Usage

### 1. Authentication

```python
logger_settings = LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="https://secure-collector.example.com/v1/logs",
    otlp_headers={
        "Authorization": "Bearer secret-token",
        "X-API-Key": "api-key-123",
    },
)
```

### 2. Multiple Handlers

Combine file, GELF (OVH LDP), and OTLP:

```python
logger_settings = LoggerSettings(
    log_level="INFO",
    
    # File logging
    file_logging=True,
    file_path="logs/app.log",
    
    # OVH LDP (GELF)
    ovh_ldp_enabled=True,
    ovh_ldp_token="ovh-token",
    ovh_ldp_endpoint="gra1.logs.ovh.com",
    
    # OpenTelemetry
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
)

# All three handlers active simultaneously!
logger = setup_logging(logger_settings=logger_settings)
```

### 3. Custom Attributes

Add extra attributes to logs:

```python
logger.info(
    "User action",
    extra={
        "user_id": 12345,
        "action": "login",
        "ip_address": "192.168.1.1",
    }
)
```

These appear as OTLP attributes and are searchable in OpenSearch.

### 4. Trace Context Integration

If you're using OpenTelemetry tracing, add trace IDs:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("operation") as span:
    trace_id = span.get_span_context().trace_id
    span_id = span.get_span_context().span_id
    
    logger.info(
        "Operation completed",
        extra={
            "trace_id": format(trace_id, '032x'),
            "span_id": format(span_id, '016x'),
        }
    )
```

## Performance Considerations

### Batching

OTLPHandler batches logs for efficiency:

- **Batch size**: 100 records (configurable in handler)
- **Batch timeout**: 5 seconds
- Logs sent when either limit reached

### Background Processing

- Logs queued asynchronously (non-blocking)
- Background thread sends batches
- Queue size: 1000 records
- Full queue = silently drop (prevents app blocking)

### Overhead

- **Disabled**: Zero overhead (handler not imported)
- **Enabled**: ~1-2ms per log (batching + async)
- **Network**: Async, doesn't block application

## Running with ASGI Servers (FastAPI, FastMCP)

### Critical: Use uvicorn for OTLP Logging

When running ASGI applications (FastAPI, FastMCP, Starlette), you **must** use uvicorn (or another ASGI server) to ensure OTLP logs are properly sent. The OTLP handler uses background threads and batching, which requires proper server lifecycle management.

**✅ Correct - Using uvicorn:**

```python
# mcp_server.py or fastapi_app.py
import uvicorn
from core_lib.config import initialize_settings
from your_app import app

# Initialize settings with setup_logging=True
settings = initialize_settings(setup_logging=True)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,  # CRITICAL: Disable uvicorn's logging config
        log_level=settings.app.log_level.lower()
    )
```

**❌ Incorrect - Using app.run() directly:**

```python
# Don't do this for OTLP logging
app.run()  # Logs may be batched but never flushed
```

### Key Configuration

1. **Disable uvicorn's log_config:**
   ```python
   uvicorn.run(
       app,
       log_config=None,  # Preserves your OTLP handler setup
   )
   ```

2. **Flush logs on shutdown:**
   ```python
   from core_lib.tracing.logger import flush_logging
   import atexit
   
   # Register flush on exit
   atexit.register(flush_logging)
   ```

3. **Explicit flush before server start:**
   ```python
   logger.info("Starting server")
   flush_logging()  # Ensure initialization logs are sent
   
   uvicorn.run(app, ...)
   ```

### Complete FastAPI Example

```python
from fastapi import FastAPI
from core_lib.config import initialize_settings, Settings
from core_lib.tracing.logger import flush_logging, get_module_logger
import atexit
import uvicorn

# Initialize settings and logging
settings = initialize_settings(settings_class=Settings, setup_logging=True)
logger = get_module_logger()

# Register cleanup
atexit.register(flush_logging)

app = FastAPI()

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    flush_logging()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,  # Don't override our OTLP logging
        log_level=settings.app.log_level.lower()
    )
```

### Complete FastMCP Example

```python
from fastmcp import FastMCP
from core_lib.config import initialize_settings, Settings
from core_lib.tracing.logger import flush_logging, get_module_logger
import atexit
import uvicorn

# Initialize settings and logging
settings = initialize_settings(settings_class=Settings, setup_logging=True)
logger = get_module_logger()

# Register cleanup
atexit.register(flush_logging)

app = FastMCP("My MCP Server")

@app.tool()
def my_tool(query: str) -> str:
    logger.info(f"Tool called with query: {query}")
    return "result"

if __name__ == "__main__":
    logger.info("Starting MCP server")
    flush_logging()
    
    # Create ASGI app from FastMCP
    mcp_asgi_app = app.http_app(path="/mcp")
    
    # Run with uvicorn
    uvicorn.run(
        mcp_asgi_app,
        host="0.0.0.0",
        port=9980,
        log_config=None,  # Preserve OTLP logging setup
        log_level=settings.app.log_level.lower()
    )
```

### Why uvicorn is Required

The OTLP handler batches logs and sends them:
- When 100 records accumulate, OR
- After 5 seconds timeout, OR
- On explicit flush/shutdown

Without uvicorn's proper lifecycle management:
- Batched logs may never be sent
- Background threads may not be cleaned up properly
- Shutdown hooks may not execute

Using uvicorn ensures:
- Proper ASGI application lifecycle
- Graceful shutdown with cleanup
- Background threads are managed correctly
- atexit handlers execute reliably

## Troubleshooting

### Logs Not Appearing in OpenSearch

1. **Check collector is running:**
   ```bash
   docker ps | grep otel-collector
   ```

2. **Verify endpoint is reachable:**
   ```bash
   curl -X POST http://localhost:4318/v1/logs \
     -H "Content-Type: application/json" \
     -d '{"resourceLogs":[]}'
   ```

3. **Check collector logs:**
   ```bash
   docker logs otel-collector
   ```

4. **Verify OpenSearch index:**
   ```bash
   curl http://localhost:9200/_cat/indices?v
   # Should show otel-logs index
   ```

### Connection Errors

Errors are logged to stderr (not logged to avoid recursion):

```
OTLP send error: Connection refused
```

**Solutions:**
- Verify `OTLP_ENDPOINT` is correct
- Check collector is listening on 4318
- Verify network connectivity (Docker networks, firewalls)

### SSL Certificate Errors

```
OTLP send error: SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions:**
- For development: `otlp_insecure=True`
- For production: Use proper certificates or add CA bundle

## Migration from Other Systems

### From Python's logging.handlers

```python
# Before: StreamHandler
handler = logging.StreamHandler()

# After: Add OTLP
logger_settings = LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
)
```

### From ELK Stack

Replace Logstash with OTLP collector:

```python
# Before: Sending to Logstash
# (custom handler or tcp/udp)

# After: Send to OTLP collector → OpenSearch
logger_settings = LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="http://otel-collector:4318/v1/logs",
)
```

Collector exports to OpenSearch (your existing config ✓)

### From Fluentd/Fluent Bit

```python
# Before: Logging to file → Fluentd tails file

# After: Direct OTLP (more efficient)
logger_settings = LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
)
```

## Best Practices

1. **Use service name and version:**
   ```python
   otlp_service_name="user-service"
   otlp_service_version="2.1.0"
   ```

2. **Add context with extra attributes:**
   ```python
   logger.info("Event", extra={"user_id": 123, "tenant": "acme"})
   ```

3. **Use appropriate log levels:**
   - DEBUG: Detailed diagnostic info
   - INFO: General informational messages
   - WARNING: Warning but application continues
   - ERROR: Error events but application continues
   - CRITICAL: Severe errors, application may fail

4. **Enable multiple handlers for redundancy:**
   ```python
   file_logging=True,  # Local backup
   otlp_enabled=True,  # Central observability
   ```

5. **Monitor collector health:**
   - Collector metrics: `http://localhost:8888/metrics`
   - Use Prometheus scraping
   - Alert on export failures

## Examples

See `examples/example_otlp_logging.py` for:
- Basic setup
- Authentication
- Environment variable configuration
- Multi-handler setup
- Docker Compose integration
- OpenSearch query examples

## References

- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otlp/)
- [OTLP Log Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/)
- [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)
- [Your otel-collector-config.yml](../otel-collector-config.yml)
