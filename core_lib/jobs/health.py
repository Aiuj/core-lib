"""Background-worker liveness and job-queue health checks.

Workers run in a separate process from the API, so the API cannot inspect
them directly.  :class:`WorkerHeartbeat` (started by :meth:`JobWorker.start`)
periodically publishes a TTL'd heartbeat through the job queue; API health
endpoints call :func:`check_job_system_health` to count the heartbeats that
are still alive.

The heartbeat runs on its own daemon thread so long-running jobs never make
a busy worker look dead.  It only beats while the poll loop is actually making
progress (a recent successful poll, or a job in flight), so a process that is
alive but no longer consuming the queue is reported as down.

The same thread doubles as a watchdog: if the idle poll loop makes no
successful poll for ``stall_timeout`` seconds it terminates the process, so
the container's restart policy brings up a fresh worker instead of leaving it
wedged.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

from core_lib.tracing.logger import get_module_logger

logger = get_module_logger()

# Heartbeat TTL = interval * this factor, so one or two missed beats (GC
# pause, slow Redis) don't flap the health status.
HEARTBEAT_TTL_FACTOR = 3

_UNSET: Any = object()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exit_process() -> None:
    # os._exit, not sys.exit: this runs on the watchdog thread, and the main
    # thread is by definition stuck, so a normal shutdown cannot be relied on.
    os._exit(1)


class WorkerHeartbeat:
    """Publishes a worker's liveness through the job queue and kills it if it stalls.

    Args:
        job_queue: Queue implementing ``publish_worker_heartbeat`` /
            ``remove_worker_heartbeat``.
        interval: Seconds between heartbeats / watchdog checks.
        liveness: ``JobWorker.liveness`` — returns ``last_poll_age`` and
            ``current_job_id``.  When omitted, every tick beats unconditionally.
        stall_timeout: Seconds without a successful poll (while idle) after
            which the process is terminated.  0 disables the watchdog.
        on_stall: Called when the stall timeout is hit (default: exit(1)).
    """

    def __init__(
        self,
        job_queue: Any,
        interval: float,
        liveness: Optional[Callable[[], Dict[str, Any]]] = None,
        stall_timeout: float = 0.0,
        on_stall: Callable[[], None] = _exit_process,
    ) -> None:
        self._job_queue = job_queue
        self._interval = max(1.0, float(interval))
        self._ttl = max(1, int(self._interval * HEARTBEAT_TTL_FACTOR))
        # A healthy idle loop polls every poll_interval (seconds); allow for a
        # few missed polls before withholding the heartbeat.
        self._poll_stale_after = max(60.0, 2 * self._interval)
        self._liveness = liveness
        self._stall_timeout = max(0.0, float(stall_timeout))
        self._on_stall = on_stall
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._started_at = _utc_now_iso()
        self._started_monotonic = time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _check_loop(self) -> Tuple[bool, Dict[str, Any]]:
        """Return (loop is making progress, liveness snapshot); fire the watchdog."""
        if self._liveness is None:
            return True, {}
        state = self._liveness()
        if state.get("current_job_id"):
            return True, state

        poll_age = state.get("last_poll_age")
        if poll_age is None:  # no successful poll yet: measure from startup
            poll_age = time.monotonic() - self._started_monotonic
        if self._stall_timeout and poll_age > self._stall_timeout:
            logger.critical(
                "[WorkerHeartbeat] No successful queue poll for %.0fs (limit %.0fs); "
                "terminating so the container restarts",
                poll_age,
                self._stall_timeout,
            )
            self._on_stall()
            return False, state
        return poll_age <= self._poll_stale_after, state

    def _beat(self, state: Dict[str, Any]) -> None:
        poll_age = state.get("last_poll_age")
        payload = {
            "worker_id": self.worker_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": self._started_at,
            "last_seen": _utc_now_iso(),
            "last_poll_age_s": None if poll_age is None else round(poll_age, 1),
            "current_job_id": state.get("current_job_id"),
        }
        self._job_queue.publish_worker_heartbeat(self.worker_id, payload, self._ttl)

    def tick(self) -> None:
        """One heartbeat/watchdog cycle (exposed for tests)."""
        alive, state = self._check_loop()
        if not alive:
            logger.warning(
                "[WorkerHeartbeat] Poll loop not making progress; withholding heartbeat"
            )
            return
        try:
            self._beat(state)
        except Exception as exc:  # noqa: BLE001 — heartbeat is best-effort
            logger.warning("[WorkerHeartbeat] Failed to publish heartbeat: %s", exc)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 — never let the watchdog die
                logger.warning("[WorkerHeartbeat] Tick failed: %s", exc)
            self._stop.wait(self._interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="worker-heartbeat", daemon=True
        )
        self._thread.start()
        logger.info(
            "[WorkerHeartbeat] Started (id=%s, interval=%ss, ttl=%ss, stall_timeout=%ss)",
            self.worker_id,
            self._interval,
            self._ttl,
            self._stall_timeout or "off",
        )

    def stop(self) -> None:
        """Stop beating and remove the heartbeat so health reflects shutdown at once."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._job_queue.remove_worker_heartbeat(self.worker_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[WorkerHeartbeat] Could not remove heartbeat: %s", exc)


def ping_job_queue(job_queue: Any) -> Dict[str, Any]:
    """Ping the job-queue backend: ``{"healthy", "latency_ms", "error"?}``."""
    client = getattr(job_queue, "client", None) if job_queue is not None else None
    if client is None:
        return {"healthy": False, "error": "Job queue not initialized"}
    start = time.perf_counter()
    try:
        ok = bool(client.ping())
        result: Dict[str, Any] = {"healthy": ok}
    except Exception as exc:  # noqa: BLE001
        result = {"healthy": False, "error": str(exc)}
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result


def check_worker_health(job_queue: Any) -> Dict[str, Any]:
    """Live background workers (from heartbeats) plus queue depth."""
    if job_queue is None:
        return {"healthy": False, "active_workers": 0, "error": "Job queue not initialized"}
    try:
        workers = job_queue.list_worker_heartbeats()
        result: Dict[str, Any] = {
            "healthy": len(workers) > 0,
            "active_workers": len(workers),
            "workers": workers,
            **job_queue.get_queue_stats(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"healthy": False, "active_workers": 0, "error": str(exc)}
    if not workers:
        result["error"] = "No background worker heartbeat found"
    return result


def check_job_system_health(job_queue: Any = _UNSET, enabled: bool = True) -> Dict[str, Any]:
    """Health components for an API that hands work to a background worker.

    Returns ``{"job_queue": {...}, "background_worker": {...}}``, each with a
    ``healthy`` flag.  When ``enabled`` is False (the app runs without a job
    queue) both report healthy and ``enabled: False``.  ``job_queue`` defaults
    to the global queue from :func:`core_lib.jobs.get_job_queue`.
    """
    if not enabled:
        disabled = {"healthy": True, "enabled": False}
        return {"job_queue": dict(disabled), "background_worker": dict(disabled)}
    if job_queue is _UNSET:
        from .job_manager import get_job_queue

        job_queue = get_job_queue()
    return {
        "job_queue": ping_job_queue(job_queue),
        "background_worker": check_worker_health(job_queue),
    }


def install_queue_unavailable_handler(app: Any, detail: Any = None) -> None:
    """Make Redis/Valkey outages on a FastAPI app a retryable 503.

    The job queue keeps its client through outages (it reconnects on the next
    command), so endpoints see ``redis`` connection errors instead of empty
    results.  Without this handler those surface as 500s — or, in code that
    maps a missing result to 404, as a misleading "job not found".

    Args:
        app: The FastAPI application.
        detail: Response ``detail`` payload (defaults to a generic message).
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    body = {
        "detail": detail
        if detail is not None
        else "The job queue is temporarily unavailable; retry shortly."
    }

    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        logger.warning("Redis/Valkey unavailable while handling %s: %s", request.url.path, exc)
        return JSONResponse(status_code=503, content=body, headers={"Retry-After": "10"})

    app.add_exception_handler(RedisConnectionError, _handler)
    app.add_exception_handler(RedisTimeoutError, _handler)
