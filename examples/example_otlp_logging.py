"""Example: OpenTelemetry Protocol (OTLP) logging integration.

This example demonstrates how to configure and use the OTLP handler to send
logs to an OpenTelemetry collector. The collector can then export logs to
various backends like OpenSearch, Elasticsearch, or cloud observability platforms.

Requirements:
    - OpenTelemetry collector running and accessible
    - requests library (installed with core-lib[dev])

Configuration:
    You can configure OTLP logging via:
    1. LoggerSettings class (programmatic)
    2. Environment variables (declarative)
    3. StandardSettings (integrated config)
"""

import time
from core_lib.config import LoggerSettings
from core_lib.tracing.logger import setup_logging


def example_basic_otlp():
    """Basic OTLP logging setup."""
    print("\n=== Example 1: Basic OTLP Logging ===\n")
    
    # Configure OTLP logging
    logger_settings = LoggerSettings(
        log_level="INFO",
        otlp_enabled=True,
        otlp_endpoint="http://localhost:4318/v1/logs",  # Default OTLP/HTTP endpoint
        otlp_service_name="my-application",
        otlp_service_version="1.0.0",
      otlp_log_channel="myfaq",
    )
    
    # Initialize logging
    logger = setup_logging(
        app_name="otlp_example",
        logger_settings=logger_settings,
        force=True,
    )
    
    # Send logs
    logger.info("Application started")
    logger.debug("This debug message won't appear (level is INFO)")
    logger.warning("Warning message", extra={"user_id": 123})
    logger.error("Error occurred", extra={"error_code": "E001"})
    
    # Give background thread time to send
    time.sleep(2)
    
    print("[OK] Logs sent to OTLP collector at http://localhost:4318/v1/logs")
    print("[OK] Resource attribute faciliter.log_channel=myfaq included for collector routing")


def example_otlp_with_authentication():
    """OTLP logging with authentication headers."""
    print("\n=== Example 2: OTLP with Authentication ===\n")
    
    # Configure OTLP with auth headers
    logger_settings = LoggerSettings(
        log_level="DEBUG",
        otlp_enabled=True,
        otlp_endpoint="http://localhost:4318/v1/logs",
        otlp_headers={
            "Authorization": "Bearer my-secret-token",
            "X-Custom-Header": "custom-value",
        },
        otlp_service_name="authenticated-service",
        otlp_timeout=15,  # Increase timeout if needed
    )
    
    logger = setup_logging(
        app_name="otlp_auth_example",
        logger_settings=logger_settings,
        force=True,
    )
    
    logger.info("Authenticated log message")
    logger.debug("Debug information", extra={"request_id": "req-12345"})
    
    time.sleep(2)
    
    print("[OK] Authenticated logs sent with custom headers")


def example_independent_log_levels():
    """OTLP with independent log level - reduce OTLP costs."""
    print("\n=== Example 3: Independent Log Levels (DEBUG console, INFO OTLP) ===\n")
    
    # Console shows DEBUG+, OTLP only receives INFO+
    logger_settings = LoggerSettings(
        log_level="DEBUG",           # Console shows all DEBUG and above
        otlp_enabled=True,
        otlp_endpoint="http://localhost:4318/v1/logs",
        otlp_log_level="INFO",       # OTLP only receives INFO and above
        otlp_service_name="cost-optimized-service",
    )
    
    logger = setup_logging(
        app_name="level_example",
        logger_settings=logger_settings,
        force=True,
    )
    
    print("Sending logs at different levels:\n")
    
    logger.debug("DEBUG: Detailed diagnostic (console only, not OTLP)")
    print("  ^ DEBUG log sent - appears in CONSOLE only\n")
    
    logger.info("INFO: General information (console + OTLP)")
    print("  ^ INFO log sent - appears in CONSOLE and OTLP\n")
    
    logger.warning("WARNING: Warning message (console + OTLP)")
    print("  ^ WARNING log sent - appears in CONSOLE and OTLP\n")
    
    logger.error("ERROR: Error occurred (console + OTLP)")
    print("  ^ ERROR log sent - appears in CONSOLE and OTLP\n")
    
    time.sleep(2)
    
    print("[OK] Independent log levels configured:")
    print("  - Console: DEBUG level (shows all logs)")
    print("  - OTLP: INFO level (only INFO, WARNING, ERROR sent)")
    print("  - Result: DEBUG logs saved to OTLP costs while visible locally")


def example_otlp_with_file_logging():
    """Combined file and OTLP logging."""
    print("\n=== Example 4: File + OTLP Logging ===\n")
    
    # Enable both file and OTLP logging
    logger_settings = LoggerSettings(
        log_level="INFO",
        file_logging=True,
        file_path="logs/combined.log",
        otlp_enabled=True,
        otlp_endpoint="http://localhost:4318/v1/logs",
        otlp_service_name="multi-output-service",
    )
    
    logger = setup_logging(
        app_name="combined_example",
        logger_settings=logger_settings,
        force=True,
    )
    
    logger.info("This log goes to BOTH file and OTLP collector")
    logger.warning("Multi-destination logging", extra={"destination": "file+otlp"})
    
    time.sleep(2)
    
    print("[OK] Logs sent to both file (logs/combined.log) and OTLP collector")


def example_otlp_from_env():
    """Configure OTLP via environment variables."""
    print("\n=== Example 5: OTLP from Environment Variables ===\n")
    
    import os
    
    # Set environment variables (including independent log level)
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["OTLP_ENABLED"] = "true"
    os.environ["OTLP_ENDPOINT"] = "http://localhost:4318/v1/logs"
    os.environ["OTLP_LOG_LEVEL"] = "WARNING"  # Only WARNING+ to OTLP
    os.environ["OTLP_SERVICE_NAME"] = "env-configured-service"
    os.environ["OTLP_SERVICE_VERSION"] = "2.0.0"
    os.environ["OTLP_LOG_CHANNEL"] = "faciliter"
    os.environ["OTLP_HEADERS"] = '{"X-API-Key": "my-api-key"}'
    
    # Load from environment
    logger_settings = LoggerSettings.from_env(load_dotenv=False)
    
    logger = setup_logging(
        app_name="env_example",
        logger_settings=logger_settings,
        force=True,
    )
    logger.info("Configured from environment variables")
    logger.debug("Service info (console only - OTLP_LOG_LEVEL=WARNING)", extra={
        "service": logger_settings.otlp_service_name,
        "version": logger_settings.otlp_service_version,
    })
    logger.warning("Warning message (console + OTLP)", extra={
        "note": "OTLP receives WARNING and above only"
    })
    
    time.sleep(2)
    
    print("[OK] OTLP configured from environment variables")


def example_otlp_with_standard_settings():
    """Use StandardSettings for integrated configuration."""
    print("\n=== Example 6: OTLP with StandardSettings ===\n")
    
    import os
    from core_lib.config import StandardSettings
    
    # Configure via environment
    os.environ["OTLP_ENABLED"] = "true"
    os.environ["OTLP_ENDPOINT"] = "http://localhost:4318/v1/logs"
    os.environ["OTLP_SERVICE_NAME"] = "standard-settings-service"
    os.environ["APP_NAME"] = "my_app"
    
    # Load StandardSettings (includes logger config)
    settings = StandardSettings.from_env(load_dotenv=False)
    
    # Use logger from StandardSettings
    logger = setup_logging(
        app_settings=settings,
        logger_settings=settings.logger_safe,
        force=True,
    )
    
    logger.info("Using StandardSettings integration")
    logger.info("App name from settings", extra={"app_name": settings.app_name})
    
    time.sleep(2)
    
    print("[OK] OTLP integrated with StandardSettings")


def example_contextual_logging():
    """Example: Contextual logging with parse_from for request metadata."""
    print("\n=== Example 7: Contextual Logging with parse_from ===\n")
    
    from core_lib.tracing import LoggingContext
    from core_lib.tracing import parse_from
    
    # Configure OTLP
    logger_settings = LoggerSettings(
        log_level="INFO",
        otlp_enabled=True,
        otlp_endpoint="http://localhost:4318/v1/logs",
        otlp_service_name="contextual-logging-service",
    )
    
    logger = setup_logging(
        app_name="context_example",
        logger_settings=logger_settings,
        force=True,
    )
    
    # Simulate a request with 'from' parameter (typically from FastAPI Query)
    from_json = """{
        "session_id": "session-abc-123",
        "user_id": "user-456",
        "user_name": "john.doe@example.com",
        "company_id": "company-789",
        "company_name": "Acme Corp",
        "app_name": "mobile-app",
        "app_version": "2.1.0"
    }"""
    
    # Parse the 'from' parameter
    from_dict = parse_from(from_json)
    
    print(f"Request metadata: {from_dict}\n")
    
    # Use LoggingContext to inject metadata into all logs
    with LoggingContext(from_dict):
        logger.info("Processing API request")
        logger.info("Looking up user data")
        logger.warning("Rate limit approaching", extra={"requests_remaining": 10})
        logger.info("Request completed successfully")
    
    # Logs outside context won't have metadata
    logger.info("Background cleanup task (no context)")
    
    time.sleep(2)
    
    print("\n[OK] Contextual logs sent with metadata:")
    print("  - session.id = session-abc-123")
    print("  - user.id = user-456")
    print("  - organization.id = company-789")
    print("  - client.app.name = mobile-app")
    print("\nIn OpenSearch/Grafana, you can filter by these fields!")


def example_docker_compose_setup():
    """Example Docker Compose setup for local testing."""
    print("\n=== Docker Compose Setup for Local Testing ===\n")
    
    docker_compose = """
# docker-compose.yml for local OTLP testing

version: '3.8'

services:
  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yml"]
    volumes:
      - ./otel-collector-config.yml:/etc/otel-collector-config.yml
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
      - "8888:8888"   # Prometheus metrics
      - "8889:8889"   # Prometheus exporter
    depends_on:
      - opensearch

  # OpenSearch for log storage
  opensearch:
    image: opensearchproject/opensearch:latest
    environment:
      - discovery.type=single-node
      - OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m
      - DISABLE_SECURITY_PLUGIN=true
    ports:
      - "9200:9200"
      - "9600:9600"
    volumes:
      - opensearch-data:/usr/share/opensearch/data

  # OpenSearch Dashboards for visualization
  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:latest
    ports:
      - "5601:5601"
    environment:
      - OPENSEARCH_HOSTS=http://opensearch:9200
      - DISABLE_SECURITY_DASHBOARDS_PLUGIN=true
    depends_on:
      - opensearch

volumes:
  opensearch-data:

# Usage:
# 1. Save this as docker-compose.yml
# 2. Create otel-collector-config.yml (see your config)
# 3. Run: docker-compose up -d
# 4. Access OpenSearch Dashboards at http://localhost:5601
# 5. Run Python examples to send logs
"""
    
    print(docker_compose)
    print("\n[OK] Copy the above docker-compose.yml to test locally")


def example_opensearch_queries():
    """Example OpenSearch queries for viewing logs."""
    print("\n=== Example OpenSearch Queries ===\n")
    
    queries = """
# 1. View all logs (from OpenSearch Dashboards Dev Tools or curl)

GET otel-logs/_search
{
  "query": {
    "match_all": {}
  },
  "sort": [
    { "@timestamp": "desc" }
  ],
  "size": 100
}

# 2. Search logs by service name

GET otel-logs/_search
{
  "query": {
    "term": {
      "resource.attributes.service.name": "my-application"
    }
  }
}

# 3. Search logs by severity (ERROR level)

GET otel-logs/_search
{
  "query": {
    "term": {
      "severityText": "ERROR"
    }
  }
}

# 4. Search logs by message content

GET otel-logs/_search
{
  "query": {
    "match": {
      "body.stringValue": "Application started"
    }
  }
}

# 5. Get logs from last 15 minutes with specific attribute

GET otel-logs/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "range": {
            "@timestamp": {
              "gte": "now-15m"
            }
          }
        },
        {
          "nested": {
            "path": "attributes",
            "query": {
              "bool": {
                "must": [
                  { "term": { "attributes.key": "user_id" } },
                  { "term": { "attributes.value.intValue": "123" } }
                ]
              }
            }
          }
        }
      ]
    }
  }
}
"""
    
    print(queries)
    print("\n[OK] Use these queries in OpenSearch Dashboards Dev Tools")


if __name__ == "__main__":
    print("=" * 70)
    print("OpenTelemetry Protocol (OTLP) Logging Examples")
    print("=" * 70)
    
    # Note: These examples require a running OTLP collector
    # Set SKIP_NETWORK_TESTS=true to skip actual network calls
    
    import os
    skip_network = os.getenv("SKIP_NETWORK_TESTS", "false").lower() == "true"
    
    if skip_network:
        print("\nSKIP_NETWORK_TESTS=true - showing configuration only\n")
        print("To run with actual OTLP collector:")
        print("1. Start OpenTelemetry collector (see Docker Compose example)")
        print("2. Unset SKIP_NETWORK_TESTS")
        print("3. Run: python examples/example_otlp_logging.py\n")
        try:
            example_basic_otlp()
            example_otlp_with_authentication()
            example_independent_log_levels()
            example_otlp_with_file_logging()
            example_otlp_from_env()
            example_otlp_with_standard_settings()
            example_contextual_logging()
            
            print("\n" + "=" * 70)
            print("All examples completed successfully!")
            print("=" * 70)
            
            example_docker_compose_setup()
            example_opensearch_queries()
            
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
            print("\nMake sure OpenTelemetry collector is running:")
            print("  docker run -p 4318:4318 otel/opentelemetry-collector-contrib")
            print("\nOr use Docker Compose (see docker-compose setup above)")

