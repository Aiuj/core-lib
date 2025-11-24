# OTLP Logging Quick Reference

## Enable OTLP Logging

### Quick Setup (Auto-Enable - Recommended)

**Minimal Configuration:**
```bash
# Only two settings needed - OTLP auto-enables!
export ENABLE_LOGGER=true
export OTLP_ENDPOINT=http://localhost:4318/v1/logs

# Optional: Service name defaults to APP_NAME
export APP_NAME=my-service

# Service version auto-detected from pyproject.toml
```

**Auto-Enable Logic:**
- OTLP automatically enables when:
  1. `ENABLE_LOGGER=true` (or `LOG_FILE_ENABLED=true`) AND
  2. `OTLP_ENDPOINT` is defined AND
  3. `OTLP_ENABLED` is not explicitly `false`

### Via Code (Explicit Configuration)
```python
from core_lib.config import LoggerSettings
from core_lib.tracing.logger import setup_logging

settings = LoggerSettings(
    log_level="DEBUG",              # Console shows DEBUG+
    otlp_enabled=True,
    otlp_endpoint="http://localhost:4318/v1/logs",
    otlp_service_name="my-app",     # Default: APP_NAME or "core-lib"
    otlp_log_level="INFO",          # OTLP only receives INFO+ (optional)
)
logger = setup_logging(logger_settings=settings)
```

### Via Environment (Full Control)
```bash
export LOG_LEVEL=DEBUG                # Console level
export OTLP_ENABLED=true              # Explicit enable (optional with auto-enable)
export OTLP_ENDPOINT=http://localhost:4318/v1/logs
export OTLP_SERVICE_NAME=my-app       # Default: APP_NAME or "core-lib"
export OTLP_SERVICE_VERSION=1.0.0     # Default: from pyproject.toml
export OTLP_LOG_LEVEL=INFO            # OTLP level (optional, defaults to LOG_LEVEL)
```

## Configuration Fields

| Field | Default | Description |
|-------|---------|-------------|
| `otlp_enabled` | `False` | Enable OTLP |
| `otlp_endpoint` | `http://localhost:4318/v1/logs` | Collector URL |
| `otlp_headers` | `{}` | Auth headers |
| `otlp_timeout` | `10` | Timeout (seconds) |
| `otlp_insecure` | `False` | Skip SSL check |
| `otlp_service_name` | `core-lib` | Service name |
| `otlp_service_version` | `None` | Version tag |
| `otlp_log_level` | Inherits from `log_level` | Independent log level for OTLP handler |

## Your Collector Setup

Your `otel-collector-config.yml` already configured:
- Receives: `0.0.0.0:4318` (OTLP/HTTP)
- Exports: OpenSearch at `otel-logs` index

## Docker Compose Integration

```yaml
my-app:
  environment:
    - OTLP_ENABLED=true
    - OTLP_ENDPOINT=http://otel-collector:4318/v1/logs
    - OTLP_SERVICE_NAME=my-app
  depends_on:
    - otel-collector
```

## View Logs

**OpenSearch Dashboards:** http://localhost:5601
- Index pattern: `otel-logs*`
- Time field: `@timestamp`

**Query API:**
```bash
curl http://localhost:9200/otel-logs/_search?pretty
```

## With Authentication

```python
LoggerSettings(
    otlp_enabled=True,
    otlp_endpoint="https://collector.example.com/v1/logs",
    otlp_headers={"Authorization": "Bearer token"},
)
```

Or:
```bash
export OTLP_HEADERS='{"Authorization": "Bearer token"}'
```

## Independent Log Levels

**Use Case:** DEBUG logs on console, only INFO+ logs to OTLP (reduce costs/noise)

```python
LoggerSettings(
    log_level="DEBUG",       # Console: DEBUG, INFO, WARNING, ERROR, CRITICAL
    otlp_enabled=True,
    otlp_log_level="INFO",   # OTLP:    INFO, WARNING, ERROR, CRITICAL only
)
```

```bash
# Environment variables
export LOG_LEVEL=DEBUG
export OTLP_LOG_LEVEL=WARNING  # Only send WARNING+ to reduce OTLP ingestion
```

**Behavior:**
- `logger.debug("...")` → Console ✓, OTLP ✗
- `logger.info("...")` → Console ✓, OTLP ✓ (if OTLP_LOG_LEVEL=INFO)
- `logger.warning("...")` → Console ✓, OTLP ✓

## Contextual Logging (Request Metadata)

Add request-specific metadata (user_id, session_id, company_id) to all logs:

```python
from core_lib.tracing import LoggingContext, parse_from

@app.post("/endpoint")
async def endpoint(from_: Optional[str] = Query(None, alias="from")):
    from_dict = parse_from(from_)  # Parses JSON from 'from' parameter
    
    # All logs within this context get metadata automatically
    with LoggingContext(from_dict):
        logger.info("Processing request")
        # OTLP log includes: session.id, user.id, organization.id, etc.
```

**What gets added:**
- `session_id` → `session.id` (OpenTelemetry convention)
- `user_id` → `user.id`
- `company_id` → `organization.id`
- `user_name` → `user.name`
- `app_name` → `client.app.name`

**Filter logs in OpenSearch:**
```
user.id: "user-456"
organization.id: "comp-789"
session.id: "session-123"
```

See **[Centralized Logging Guide](centralized-logging.md#contextual-logging-request-metadata)** for detailed examples.

## Multiple Handlers

```python
LoggerSettings(
    file_logging=True,      # → File
    ovh_ldp_enabled=True,   # → OVH
    otlp_enabled=True,      # → OpenTelemetry
)
```

All work simultaneously!

## Troubleshooting

**Logs not appearing?**
1. Check collector: `docker logs otel-collector`
2. Test endpoint: `curl http://localhost:4318/v1/logs`
3. Verify index: `curl http://localhost:9200/_cat/indices`

**Connection errors?**
- Verify `OTLP_ENDPOINT` is correct
- Check Docker network connectivity
- For dev: use `otlp_insecure=True`

## Full Documentation

- Architecture & setup: `docs/OTLP_LOGGING_INTEGRATION.md`
- Examples: `examples/example_otlp_logging.py`
- Implementation details: `docs/implementation/OTLP_IMPLEMENTATION_SUMMARY.md`
