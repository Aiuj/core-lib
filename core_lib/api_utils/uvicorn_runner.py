"""Uvicorn server runner utilities for ASGI applications.

This module provides a standardized way to run ASGI applications (FastAPI, FastMCP, etc.)
with uvicorn, ensuring proper OTLP logging, graceful shutdown, and lifecycle management.
"""

from typing import Any, Optional, Union
import atexit


def run_uvicorn_server(
    app: Any,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
    reload: bool = False,
    app_module_path: Optional[str] = None,
    ws: str = "websockets-sansio",
    flush_logs_on_startup: bool = True,
    flush_logs_on_exit: bool = True,
) -> None:
    """Run an ASGI application with uvicorn with proper logging configuration.
    
    This function handles:
    - Disabling uvicorn's log_config to preserve OTLP/custom logging setup
    - Optional log flushing before server starts and on exit
    - Support for reload mode with module path
    - Consistent server configuration across services
    
    Args:
        app: ASGI application instance or callable
        host: Host to bind to (default: "0.0.0.0")
        port: Port to bind to (default: 8000)
        log_level: Uvicorn log level (default: "info")
        reload: Enable auto-reload on code changes (default: False)
        app_module_path: Module path for reload mode (e.g., "my_app:app")
        ws: WebSocket protocol implementation for uvicorn (default: "websockets-sansio")
        flush_logs_on_startup: Flush logs before starting server (default: True)
        flush_logs_on_exit: Register atexit handler to flush logs (default: True)
    
    Example:
        >>> from fastapi import FastAPI
        >>> from core_lib.api_utils import run_uvicorn_server
        >>> 
        >>> app = FastAPI()
        >>> 
        >>> if __name__ == "__main__":
        >>>     run_uvicorn_server(
        >>>         app,
        >>>         host="0.0.0.0",
        >>>         port=8000,
        >>>         log_level="info"
        >>>     )
    
    Note:
        For OTLP logging to work properly with batching, you MUST use uvicorn
        (or another ASGI server). Using app.run() directly may cause batched
        logs to never be flushed. See OTLP_LOGGING_INTEGRATION.md for details.
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "uvicorn is required to use run_uvicorn_server. "
            "Install it with: pip install uvicorn"
        ) from e
    
    # Import flush_logging lazily to avoid circular imports
    if flush_logs_on_startup or flush_logs_on_exit:
        try:
            from core_lib.tracing.logger import flush_logging
        except ImportError:
            flush_logging = None
            if flush_logs_on_startup or flush_logs_on_exit:
                import warnings
                warnings.warn(
                    "core_lib.tracing.logger not available, log flushing disabled"
                )
    
    # Register log flushing on exit
    if flush_logs_on_exit and flush_logging:
        atexit.register(flush_logging)
    
    # Flush logs before starting server
    if flush_logs_on_startup and flush_logging:
        flush_logging()
    
    # Determine what to pass to uvicorn.run()
    if reload:
        # When reload is True, use string path so uvicorn can reimport on file changes
        if not app_module_path:
            raise ValueError(
                "app_module_path is required when reload=True. "
                "Example: 'my_module:app'"
            )
        app_or_path: Union[str, Any] = app_module_path
    else:
        # When reload is False, pass the app object directly to avoid double import
        app_or_path = app
    
    # Run uvicorn with log_config=None to preserve custom logging setup
    uvicorn.run(
        app_or_path,
        host=host,
        port=port,
        reload=reload,
        ws=ws,
        log_config=None,  # CRITICAL: Preserves OTLP/custom logging handlers
        log_level=log_level.lower()
    )


def run_uvicorn_from_settings(
    app: Any,
    settings: Any,
    host_attr: str = "host",
    port_attr: str = "port",
    reload_attr: str = "reload",
    log_level_attr: str = "log_level",
    ws: str = "websockets-sansio",
    app_module_path: Optional[str] = None,
) -> None:
    """Run uvicorn server with configuration from settings object.
    
    This is a convenience wrapper that extracts host, port, reload, and log_level
    from a settings object (e.g., StandardSettings, FastAPISettings, MCPSettings).
    
    Args:
        app: ASGI application instance or callable
        settings: Settings object with server configuration
        host_attr: Attribute name for host (default: "host")
        port_attr: Attribute name for port (default: "port")
        reload_attr: Attribute name for reload flag (default: "reload")
        log_level_attr: Attribute name for log level (default: "log_level")
        ws: WebSocket protocol implementation for uvicorn (default: "websockets-sansio")
        app_module_path: Module path for reload mode (e.g., "my_app:app")
    
    Example:
        >>> from core_lib.config import StandardSettings
        >>> from core_lib.api_utils import run_uvicorn_from_settings
        >>> 
        >>> settings = StandardSettings.from_env()
        >>> 
        >>> run_uvicorn_from_settings(
        >>>     app,
        >>>     settings.fastapi,
        >>>     app_module_path="my_app:app" if settings.fastapi.reload else None
        >>> )
    """
    host = getattr(settings, host_attr, "0.0.0.0")
    port = getattr(settings, port_attr, 8000)
    reload = getattr(settings, reload_attr, False)
    log_level = getattr(settings, log_level_attr, "info")
    
    run_uvicorn_server(
        app=app,
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
        ws=ws,
        app_module_path=app_module_path,
    )
