"""Structured "usage event" logging for functional/user actions.

This is distinct from ``service_usage.py`` (LLM/embedding/OCR/search cost
tracking). A usage event marks the completion of one user- or system-facing
*action* -- a document ingested, an RFx questionnaire answered, a question
answered -- as exactly one INFO-level log line, so that a dashboard can list
"one row per action" the same way ``service_usage.py``'s functions let a
dashboard list "one row per provider call".

Usage:
    ```python
    from core_lib.tracing import log_usage_event
    from core_lib.tracing.usage_actions import ACTION_DOCUMENT_INGESTED

    log_usage_event(
        logger, ACTION_DOCUMENT_INGESTED, domain="ingestion",
        document_id=document_id, chunks_count=12, processing_time_ms=842.3,
    )
    ```

The emitted record follows the same OTel-semconv-style shape already used by
``service_usage.py`` and by mcp-doc-qa's answer-strategy logging: an
``event.name``/``event.domain`` pair plus arbitrary dotted attribute keys,
delivered via ``extra={"extra_attrs": event}`` so ``LoggingContextFilter``
merges in the ambient session/process/user/company context and the OTLP
handler ships it to OpenSearch like any other log record -- no new index or
pipeline is needed.
"""

import logging
from typing import Any, Optional

from .logging_context import get_current_logging_context


def _merge_context(event: dict[str, Any]) -> None:
    context = get_current_logging_context()
    for source, destination in {
        "process_id": "process.id",
        "session_id": "session.id",
        "user_id": "user.id",
        "user_name": "user.name",
        "company_id": "organization.id",
        "company_name": "organization.name",
        "project_id": "rfx.project.id",
        "generation_id": "rfx.generation.id",
        "operation": "rfx.operation",
        "document_type": "rfx.document.type",
    }.items():
        value = context.get(source)
        if value not in (None, ""):
            event.setdefault(destination, value)


def log_usage_event(
    logger: logging.Logger,
    action: str,
    domain: str,
    message: Optional[str] = None,
    **fields: Any,
) -> None:
    """Log one structured usage event marking a completed action.

    Args:
        logger: The module logger to emit on.
        action: Canonical action name, e.g. ``"document.ingested"``. See
            ``core_lib.tracing.usage_actions`` for the shared taxonomy.
        domain: Coarse grouping for the action, e.g. ``"ingestion"``, ``"rfx"``.
        message: Optional human-readable log message. Defaults to
            ``f"Usage event: {action}"``.
        **fields: Additional attributes to attach to the event (document id,
            counts, duration, etc). ``None`` values are dropped so callers can
            pass through optional/unavailable fields without polluting the
            record.
    """
    event = {"event.name": action, "event.domain": domain}
    for key, value in fields.items():
        if value is not None:
            event[key] = value

    _merge_context(event)

    logger.info(message or f"Usage event: {action}", extra={"extra_attrs": event})


def log_usage_error_event(
    logger: logging.Logger,
    action: str,
    domain: str,
    error_code: str,
    error: str,
    message: Optional[str] = None,
    **fields: Any,
) -> None:
    """Emit a structured, non-exception operational failure event.

    Use this when a completed provider call produces an unusable application
    result (for example a draft that fails a deterministic business rule).
    It intentionally does not raise or log a traceback: the caller can store
    an actionable status while OpenSearch receives a filterable error event.
    """
    event = {
        "event.name": action,
        "event.domain": domain,
        "status": "error",
        "error.code": error_code,
        "error": error,
    }
    for key, value in fields.items():
        if value is not None:
            event[key] = value
    _merge_context(event)
    logger.warning(
        message or f"Usage error: {action} ({error_code})",
        extra={"extra_attrs": event},
    )


__all__ = ["log_usage_event", "log_usage_error_event"]
