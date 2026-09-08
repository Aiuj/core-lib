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

import atexit
import gzip
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ..config.payload_capture_settings import PayloadCaptureSettings
from .logger import get_module_logger

logger = get_module_logger()

_warned_missing_buckets: Set[str] = set()


def _reset_warned_missing_buckets() -> None:
    """Clear the cached set of missing buckets (used in tests)."""
    _warned_missing_buckets.clear()


# Uploads run on a small background pool so a slow/remote S3 endpoint never
# adds latency to the LLM call that triggered the capture (see log_llm_usage,
# which is already non-blocking for the same reason).
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()

# One boto3 S3 client per distinct endpoint/credential combination, reused
# across calls instead of rebuilt (and re-connected) on every capture.
_s3_client_cache: Dict[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]], Any] = {}
_s3_client_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="llm-payload-capture"
                )
                atexit.register(_executor.shutdown, wait=False)
    return _executor


def _get_s3_client(settings: PayloadCaptureSettings) -> Any:
    key = (
        settings.s3_endpoint_url,
        settings.s3_region,
        settings.aws_access_key_id,
        settings.aws_secret_access_key,
    )
    client = _s3_client_cache.get(key)
    if client is not None:
        return client
    with _s3_client_lock:
        client = _s3_client_cache.get(key)
        if client is None:
            client_kwargs: Dict[str, Any] = {}
            if settings.s3_endpoint_url:
                client_kwargs["endpoint_url"] = settings.s3_endpoint_url
            if settings.s3_region:
                client_kwargs["region_name"] = settings.s3_region
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
                client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            client = boto3.client("s3", **client_kwargs)
            _s3_client_cache[key] = client
        return client


def _reset_s3_client_cache() -> None:
    """Clear the cached S3 clients (used in tests)."""
    with _s3_client_lock:
        _s3_client_cache.clear()


def _wait_for_pending_captures() -> None:
    """Block until all queued payload-capture uploads finish (used in tests)."""
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=True)
            _executor = None


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
    force_enabled: Optional[bool] = None,
) -> None:
    """Best-effort, non-blocking upload of an LLM call's prompt/response to S3.

    No-op unless capture is enabled via settings/env (or forced on for this
    call). The actual upload runs on a background thread pool so a slow or
    remote S3 endpoint never adds latency to the LLM call that triggered the
    capture; any error (missing credentials, network failure, etc.) is caught
    and logged in that worker thread and can never break the calling LLM
    request.

    Args:
        call_id: The id returned by `log_llm_usage()` for this same call.
        provider: Provider name (e.g. "openai").
        model: Model name (e.g. "gpt-4").
        messages: The request message history sent to the model.
        response_text: The model's response text.
        metadata: Any additional context to store alongside the payload.
        settings: Optional pre-built settings (mainly for tests); defaults
            to `PayloadCaptureSettings.from_env()`.
        force_enabled: Per-model override (from `llm_providers.yaml`'s
            `payload_capture: true/false`) that takes precedence over
            `settings.enabled`. `None` (the default) means "use the global
            setting"; `True`/`False` forces capture on/off for this call
            regardless of the global `LLM_PAYLOAD_CAPTURE_ENABLED` value.
    """
    settings = settings or PayloadCaptureSettings.from_env()
    enabled = settings.enabled if force_enabled is None else force_enabled
    if not enabled:
        return
    if not settings.s3_bucket:
        if force_enabled:
            logger.warning(
                "LLM payload capture forced enabled for call_id=%s but no S3 bucket is "
                "configured (LLM_PAYLOAD_S3_BUCKET); skipping upload.",
                call_id,
            )
        return

    _get_executor().submit(
        _do_capture, call_id, provider, model, messages, response_text, metadata, settings
    )


def _do_capture(
    call_id: str,
    provider: str,
    model: str,
    messages: Optional[List[Dict[str, Any]]],
    response_text: Optional[str],
    metadata: Optional[Dict[str, Any]],
    settings: PayloadCaptureSettings,
) -> None:
    """Perform the actual S3 upload. Runs on the background pool."""
    try:
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

        s3_client = _get_s3_client(settings)
        key = _s3_key_for(call_id)
        s3_client.put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        error_msg = exc.response.get("Error", {}).get("Message", str(exc))
        bucket = settings.s3_bucket if settings else "unknown"
        if error_code == "NoSuchBucket":
            if bucket not in _warned_missing_buckets:
                _warned_missing_buckets.add(bucket)
                logger.warning(
                    "Failed to capture LLM payload for call_id=%s: S3 bucket '%s' does not exist (%s). "
                    "Further NoSuchBucket warnings for this bucket will be suppressed.",
                    call_id,
                    bucket,
                    error_msg,
                )
            else:
                logger.debug(
                    "Failed to capture LLM payload for call_id=%s: S3 bucket '%s' does not exist",
                    call_id,
                    bucket,
                )
        else:
            logger.warning(
                "Failed to capture LLM payload for call_id=%s (S3 %s: %s)",
                call_id,
                error_code or "ClientError",
                error_msg,
            )
        # ClientError is an expected S3 service response (for example, a
        # misconfigured access key), rather than an application exception.
        # The warning above contains the actionable code and message; do not
        # attach ``exc_info`` because that emits a misleading traceback for a
        # best-effort operation.
    except BotoCoreError as exc:
        logger.warning("Failed to capture LLM payload for call_id=%s: %s", call_id, exc)
        logger.debug("S3 payload capture traceback for call_id=%s:", call_id, exc_info=True)
    except Exception as exc:
        logger.warning("Failed to capture LLM payload for call_id=%s: %s", call_id, exc)
        logger.debug("Payload capture traceback for call_id=%s:", call_id, exc_info=True)
