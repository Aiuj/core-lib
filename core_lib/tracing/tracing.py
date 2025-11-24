"""Configure OpenTelemetry and Langfuse for the application.

This module centralizes all tracing concerns so callers don't import or depend
on Langfuse directly. It also exposes small helpers to integrate model
providers (e.g., Google Gemini) with Langfuse's standard provider tracing when
available, while remaining a no-op fallback if the integration isn't present.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

from langfuse import Langfuse, get_client
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

if TYPE_CHECKING:
    from core_lib.config.tracing_settings import TracingSettings


class TracingProvider(ABC):
    """Abstract base class for tracing providers."""
    
    @abstractmethod
    def add_metadata(self, metadata: Any) -> None:
        """Add metadata to the current trace. Accepts dict or JSON string."""
        pass


class LangfuseTracingProvider(TracingProvider):
    """Langfuse implementation of the tracing provider."""
    
    def __init__(self, langfuse_client: Langfuse):
        self._client = langfuse_client
    
    def add_metadata(self, metadata: Any) -> None:
        """Add metadata to the current trace. Accepts dict or JSON string.
        
        Args:
            metadata: Metadata to add. Can be:
                - None: no-op
                - dict: used directly
                - str: parsed as JSON
                - Any other type: ignored
        """
        if metadata is None:
            return
        
        from core_lib.tracing import parse_from
        
        # Convert to dict if needed
        if isinstance(metadata, str):
            try:
                metadata = parse_from(metadata)
            except Exception:
                # Invalid JSON string, ignore
                return
        elif not isinstance(metadata, dict):
            # Not a dict or string, ignore
            return
        
        # Only update if we have a valid dict
        if metadata:
            try:
                self._client.update_current_span(metadata=metadata)
            except Exception as e:
                # Silently ignore if no active span context exists or validation fails
                # This is expected in some execution contexts (e.g., MCP server)
                # Common errors: invalid session_id format, missing required fields, etc.
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to update trace metadata: {e}")
                pass


class NoOpTracingProvider(TracingProvider):
    """No-operation tracing provider for when tracing is disabled."""
    
    def add_metadata(self, metadata: Any) -> None:
        """No-op implementation of add_metadata.
        
        Args:
            metadata: Ignored in no-op implementation.
        """
        pass


class TracingManager:
    """Manages tracing configuration and provides a unified interface."""
    
    def __init__(self, service_name: Optional[str] = None, settings: Optional["TracingSettings"] = None):
        if settings:
            self.settings = settings
            # If service_name is explicitly provided, create new settings with updated service_name
            if service_name and service_name != settings.service_name:
                from core_lib.config.tracing_settings import TracingSettings
                self.settings = TracingSettings(
                    enabled=settings.enabled,
                    service_name=service_name,
                    service_version=settings.service_version,
                    langfuse_public_key=settings.langfuse_public_key,
                    langfuse_secret_key=settings.langfuse_secret_key,
                    langfuse_host=settings.langfuse_host,
                )
        else:
            # Create settings from environment variables, optionally overriding service_name
            from core_lib.config.tracing_settings import TracingSettings
            overrides = {}
            if service_name:
                overrides['service_name'] = service_name
            else:
                # If no explicit service_name and no APP_NAME env var, use "unknown"
                app_name = os.getenv("APP_NAME")
                if not app_name:
                    overrides['service_name'] = "unknown"
            self.settings = TracingSettings.from_env(load_dotenv=False, **overrides)
        
        self._provider: Optional[TracingProvider] = None
        self._initialized = False
    
    def setup(self) -> TracingProvider:
        """Setup tracing and return the tracing provider."""
        if self._initialized and self._provider is not None:
            return self._provider
        
        # Check if tracing is disabled
        if not self.settings.enabled:
            # Return a no-op provider for disabled tracing
            self._provider = NoOpTracingProvider()
            self._initialized = True
            return self._provider
        
        # Check if TracerProvider is already set
        if trace.get_tracer_provider() is not trace.ProxyTracerProvider():
            # Tracing already initialized, just return Langfuse client
            langfuse_client = get_client()
            self._provider = LangfuseTracingProvider(langfuse_client)
            self._initialized = True
            return self._provider
        
        # Configure OpenTelemetry
        service_name = self.settings.service_name or "unknown"
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": self.settings.service_version,
            }
        )
        
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        
        # Configure Langfuse
        langfuse_client = Langfuse(
            x_langfuse_sdk_name="Langfuse Python SDK",
            x_langfuse_sdk_version="1.0.0",
            x_langfuse_public_key=self.settings.langfuse_public_key,
            username=self.settings.langfuse_secret_key,
            password="",
            base_url=self.settings.langfuse_host,
        )
        
        self._provider = LangfuseTracingProvider(langfuse_client)
        self._initialized = True
        return self._provider
    
    def get_provider(self) -> Optional[TracingProvider]:
        """Get the current tracing provider."""
        return self._provider
    
    def add_metadata(self, metadata: Any) -> None:
        """Add metadata to the current trace. Accepts dict or JSON string."""
        if self._provider:
            self._provider.add_metadata(metadata)


# ------------------------- Convenience helpers -------------------------
def add_trace_metadata(metadata: Any, settings: Optional["TracingSettings"] = None) -> None:
    """Add metadata to the current trace in a provider-agnostic way.

    Args:
        metadata: Additional info to attach to the active span/trace.
            Can be a dict or a JSON string; implementation will parse strings.
        settings: Optional TracingSettings to use. If not provided, will use environment variables.
    """
    try:
        manager = TracingManager(settings=settings)
        provider = manager.get_provider() or manager.setup()
        provider.add_metadata(metadata)  # type: ignore[union-attr]
    except Exception:
        # Never fail application flow due to tracing
        pass

def suppress_otel_exporter_logs() -> None:
    """Suppress noisy OpenTelemetry exporter logs.
    
    This should be called after tracing is initialized to prevent retry warnings
    and transient error messages from cluttering application logs.
    """
    import logging
    
    try:
        logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(logging.CRITICAL)
        logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(logging.CRITICAL)
        logging.getLogger("opentelemetry.sdk.trace").setLevel(logging.CRITICAL)
        logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
    except Exception:
        pass


def setup_tracing(name: Optional[str] = None, settings: Optional["TracingSettings"] = None) -> TracingProvider:
    """Configure tracing and return the tracing provider.
    
    Args:
        name: Optional service name (backward compatibility).
        settings: Optional TracingSettings to use. If not provided, will use environment variables.
    
    This function maintains backward compatibility with the original API.
    """
    manager = TracingManager(service_name=name, settings=settings)
    provider = manager.setup()
    
    # Suppress noisy OTel exporter logs after initialization
    suppress_otel_exporter_logs()
    
    return provider