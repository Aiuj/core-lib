"""Optional capture of full LLM prompt/response payloads to object storage.

This is the content-capture counterpart to `service_usage.log_llm_usage()`,
which only ever logs numeric metrics (tokens, cost, latency) — never the
actual prompt or response text. When enabled via `PayloadCaptureSettings`
(env var `LLM_PAYLOAD_CAPTURE_ENABLED`), this module uploads a small gzip'd
JSON document per LLM call to S3-compatible object storage, keyed by the
`call_id` returned from `log_llm_usage()`, so the two can be correlated
later (e.g. by saas-admin's LLM Call Inspector).

Design goals:
- Best-effort only: any failure here must never break a live LLM call.
- Cheap by default: disabled unless explicitly configured, size-capped,
  gzip-compressed, and expected to be paired with an S3 lifecycle rule that
  auto-deletes objects after a configurable retention window.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3

from ..config.payload_capture_settings import PayloadCaptureSettings
from .logger import get_module_logger

logger = get_module_logger()


def _s3_key_for(call_id: str, when: Optional[datetime] = None) -> str:
    """Build the deterministic S3 key for a given call_id and timestamp.

    The key is derivable from `call_id` + the event's own timestamp alone,
    so no extra correlation field needs to be stored anywhere else.
    """
    when = when or datetime.now(timezone.utc)
    return f"llm-payloads/{when.strftime('%Y/%m/%d')}/{call_id}.json.gz"


def _truncate(value: Any, max_chars: int) -> Any:
    """Truncate a string (or the JSON-serialized form of a value) to max_chars."""
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) <= max_chars:
        return value
    return text[:max_chars] + "...[truncated]"


def capture_llm_payload(
    call_id: str,
    provider: str,
    model: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    response_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    settings: Optional[PayloadCaptureSettings] = None,
) -> None:
    """Best-effort upload of an LLM call's full prompt/response to S3.

    No-op unless capture is enabled via settings/env. Never raises: any
    error (missing credentials, network failure, etc.) is caught and logged
    so it can never break the calling LLM request.

    Args:
        call_id: The id returned by `log_llm_usage()` for this same call.
        provider: Provider name (e.g. "openai").
        model: Model name (e.g. "gpt-4").
        messages: The request message history sent to the model.
        response_text: The model's response text.
        metadata: Any additional context to store alongside the payload.
        settings: Optional pre-built settings (mainly for tests); defaults
            to `PayloadCaptureSettings.from_env()`.
    """
    try:
        settings = settings or PayloadCaptureSettings.from_env()
        if not settings.enabled or not settings.s3_bucket:
            return

        max_chars = settings.max_chars
        payload = {
            "call_id": call_id,
            "provider": provider,
            "model": model,
            "messages": _truncate(messages, max_chars) if messages is not None else None,
            "response": _truncate(response_text, max_chars) if response_text is not None else None,
            "metadata": metadata or {},
        }
        body = gzip.compress(json.dumps(payload, default=str).encode("utf-8"))

        client_kwargs: Dict[str, Any] = {}
        if settings.s3_endpoint_url:
            client_kwargs["endpoint_url"] = settings.s3_endpoint_url
        if settings.s3_region:
            client_kwargs["region_name"] = settings.s3_region
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        s3_client = boto3.client("s3", **client_kwargs)
        key = _s3_key_for(call_id)
        s3_client.put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
        )
    except Exception:
        logger.warning("Failed to capture LLM payload for call_id=%s", call_id, exc_info=True)
