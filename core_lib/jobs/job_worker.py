"""Background job worker for processing queued jobs."""

import os
import time
import signal
import threading
from typing import Callable, Dict, Any, Optional
from abc import ABC, abstractmethod

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from .base_job_queue import BaseJobQueue, JobStatus, Job
from .health import WorkerHeartbeat
from .job_manager import get_job_queue
from core_lib.tracing.logger import get_module_logger
from core_lib.tracing import LoggingContext, parse_from, generate_process_id
from core_lib.exceptions import ConfigurationError


logger = get_module_logger()

# Errors that mean "the queue backend is unreachable right now" rather than a
# bug.  The worker backs off and keeps polling instead of exiting on these.
TRANSIENT_QUEUE_ERRORS = (RedisConnectionError, RedisTimeoutError, OSError)


class JobQueueUnavailableError(RuntimeError):
    """Raised when the job queue is still unreachable after the startup timeout."""


class JobHandler(ABC):
    """Abstract base class for job handlers."""
    
    @abstractmethod
    def handle(self, job: Job) -> Dict[str, Any]:
        """Handle a job and return the result.
        
        Args:
            job: Job to process
            
        Returns:
            Result dictionary
            
        Raises:
            Exception: If job processing fails
        """
        pass
    
    def get_job_type(self) -> str:
        """Get the job type this handler processes.
        
        Returns:
            Job type string
        """
        # Default: use class name without 'Handler' suffix
        class_name = self.__class__.__name__
        if class_name.endswith('Handler'):
            return class_name[:-7].lower()
        return class_name.lower()

    def on_job_lost(self, job: Job, requeued: bool, error: str) -> None:
        """Called when a job of this type was abandoned by a dead worker.

        The worker that ran it crashed or was killed mid-job, so :meth:`handle`
        never got to clean up.  ``requeued`` is True when the job was put back
        on the queue (it will run again), False when it was marked failed for
        good.  Override to update app-side records (audit rows, temp files).
        This may run in a different worker process than the one that died.
        """


class JobWorker:
    """Background worker for processing queued jobs."""
    
    def __init__(
        self,
        job_queue: Optional[BaseJobQueue] = None,
        poll_interval: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        heartbeat_interval: float = 30.0,
        startup_timeout: float = 300.0,
        max_backoff: float = 30.0,
        liveness_interval: Optional[float] = None,
        stall_timeout: Optional[float] = None,
        stale_job_timeout: Optional[float] = None,
    ):
        """Initialize job worker.
        
        Args:
            job_queue: Job queue instance (uses global if not provided)
            poll_interval: Seconds to wait between queue polls
            max_retries: Maximum number of retries for failed jobs
            retry_delay: Delay between retries in seconds
            heartbeat_interval: Seconds between liveness updates while a handler runs.
                Set to 0 to disable heartbeats.
            startup_timeout: Seconds :meth:`start` waits for the queue backend to
                become reachable before raising :class:`JobQueueUnavailableError`.
                Set to 0 to skip the check.
            max_backoff: Upper bound (seconds) for the exponential backoff used
                while the queue backend is unreachable.
            liveness_interval: Seconds between worker heartbeats published for
                API health checks (see :mod:`core_lib.jobs.health`).  Defaults
                to ``JOB_WORKER_HEARTBEAT_INTERVAL`` or 15.  0 disables them
                (and the stall watchdog).
            stall_timeout: Seconds the idle poll loop may go without a
                successful poll before the process exits (so the container
                restarts).  Defaults to ``JOB_WORKER_STALL_TIMEOUT`` or 600.
                0 disables the watchdog.  Raised to ``startup_timeout + 60`` if
                lower, so waiting for the queue at startup never trips it.
            stale_job_timeout: Seconds a PROCESSING job may go without a
                per-job heartbeat before it is considered lost (its worker
                died mid-job) and re-enqueued or failed.  Checked at startup
                and every minute.  Defaults to ``JOB_WORKER_STALE_JOB_TIMEOUT``
                or 600 (longer than a typical Redis outage, during which
                heartbeats cannot be written).  0 disables it; also disabled
                when ``heartbeat_interval`` is 0, since jobs then never refresh.
        """
        self.job_queue = job_queue or get_job_queue()
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.heartbeat_interval = max(0.0, float(heartbeat_interval))
        self.startup_timeout = max(0.0, float(startup_timeout))
        self.max_backoff = max(1.0, float(max_backoff))
        if liveness_interval is None:
            liveness_interval = float(os.getenv("JOB_WORKER_HEARTBEAT_INTERVAL", "15"))
        if stall_timeout is None:
            stall_timeout = float(os.getenv("JOB_WORKER_STALL_TIMEOUT", "600"))
        self.liveness_interval = max(0.0, float(liveness_interval))
        self.stall_timeout = max(0.0, float(stall_timeout))
        if self.stall_timeout and self.stall_timeout <= self.startup_timeout:
            self.stall_timeout = self.startup_timeout + 60
        if stale_job_timeout is None:
            stale_job_timeout = float(os.getenv("JOB_WORKER_STALE_JOB_TIMEOUT", "600"))
        self.stale_job_timeout = max(0.0, float(stale_job_timeout))
        if self.heartbeat_interval <= 0:
            self.stale_job_timeout = 0.0
        elif self.stale_job_timeout:
            # Several missed per-job heartbeats before a job counts as lost.
            self.stale_job_timeout = max(self.stale_job_timeout, 5 * self.heartbeat_interval)
        self._last_stale_check: Optional[float] = None

        self._handlers: Dict[str, JobHandler] = {}
        self._running = False
        self._stop_requested = False
        self._wake = threading.Event()

        # Liveness state, read by external watchdogs via liveness().
        self._last_poll_at: Optional[float] = None
        self._current_job_id: Optional[str] = None
        self._job_started_at: Optional[float] = None
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def register_handler(self, handler: JobHandler):
        """Register a job handler.
        
        Args:
            handler: Job handler instance
        """
        job_type = handler.get_job_type()
        self._handlers[job_type] = handler
        logger.info(f"[JobWorker] Registered handler for job type: {job_type}")
    
    def register_function_handler(
        self,
        job_type: str,
        handler_func: Callable[[Job], Dict[str, Any]]
    ):
        """Register a function as a job handler.
        
        Args:
            job_type: Job type to handle
            handler_func: Function that processes the job
        """
        class FunctionHandler(JobHandler):
            def __init__(self, func, jtype):
                self._func = func
                self._job_type = jtype
            
            def handle(self, job: Job) -> Dict[str, Any]:
                return self._func(job)
            
            def get_job_type(self) -> str:
                return self._job_type
        
        handler = FunctionHandler(handler_func, job_type)
        self.register_handler(handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"[JobWorker] Received signal {signum}, shutting down...")
        self._stop_requested = True
        self._wake.set()

    def _sleep(self, seconds: float) -> None:
        """Sleep that returns early when a stop is requested."""
        self._wake.wait(seconds)

    def _backoff(self, attempt: int) -> float:
        # Cap the exponent: attempts keep counting during a long outage and
        # 2.0 ** 1024 overflows.
        return min(self.max_backoff, 2.0 ** min(max(0, attempt - 1), 16))

    def _wait_for_queue(self) -> None:
        """Block until the queue backend answers, or raise after startup_timeout."""
        health_check = getattr(self.job_queue, "health_check", None)
        if health_check is None or self.startup_timeout <= 0:
            return

        deadline = time.monotonic() + self.startup_timeout
        attempt = 0
        while not self._stop_requested:
            if health_check():
                if attempt:
                    logger.info(
                        "[JobWorker] Job queue reachable after %d failed attempt(s)", attempt
                    )
                return
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise JobQueueUnavailableError(
                    f"Job queue unreachable after {self.startup_timeout:.0f}s"
                )
            delay = min(self._backoff(attempt), remaining)
            logger.warning(
                "[JobWorker] Job queue unreachable (attempt %d), retrying in %.0fs",
                attempt,
                delay,
            )
            self._sleep(delay)

    STALE_CHECK_INTERVAL = 60.0

    def _notify_job_lost(self, job: Job, requeued: bool, error: str) -> None:
        handler = self._handlers.get(job.job_type)
        if handler is not None:
            handler.on_job_lost(job, requeued, error)

    def _recover_stale_jobs(self) -> None:
        """Reclaim jobs whose worker died mid-job (at most once a minute)."""
        if not self.stale_job_timeout:
            return
        now = time.monotonic()
        if (
            self._last_stale_check is not None
            and now - self._last_stale_check < self.STALE_CHECK_INTERVAL
        ):
            return
        self._last_stale_check = now
        recover = getattr(self.job_queue, "recover_stale_processing_jobs", None)
        if recover is None:
            return
        counts = recover(
            self.stale_job_timeout, self.max_retries, on_reclaimed=self._notify_job_lost
        )
        if counts.get("requeued") or counts.get("failed"):
            logger.warning(
                "[JobWorker] Reclaimed jobs lost by a dead worker: %s requeued, %s failed",
                counts.get("requeued", 0),
                counts.get("failed", 0),
            )

    def liveness(self) -> Dict[str, Any]:
        """Snapshot of the poll loop's progress, for external watchdogs.

        ``last_poll_age`` is the number of seconds since the queue was last polled
        successfully (``None`` before the first poll).  While a job runs the loop
        does not poll, so watchdogs should also look at ``current_job_id``.
        """
        now = time.monotonic()
        return {
            "running": self._running,
            "last_poll_age": None if self._last_poll_at is None else now - self._last_poll_at,
            "current_job_id": self._current_job_id,
            "job_running_for": None
            if self._job_started_at is None
            else now - self._job_started_at,
        }
    
    def _process_job(self, job: Job) -> bool:
        """Process a single job.
        
        Args:
            job: Job to process
            
        Returns:
            True if job was processed successfully
        """
        job_id = job.job_id
        job_type = job.job_type
        
        # Extract logging context from job payload
        raw_input = job.input_data or {}
        from_raw = raw_input.get("from") or raw_input.get("from_")
        if not from_raw and job.metadata:
            from_raw = job.metadata.get("from") or job.metadata.get("from_")
        from_dict = parse_from(from_raw) if from_raw else {}
        if "process_id" not in from_dict:
            if raw_input.get("process_id"):
                from_dict["process_id"] = str(raw_input["process_id"])
            elif job.metadata and job.metadata.get("process_id"):
                from_dict["process_id"] = str(job.metadata["process_id"])
            else:
                from_dict["process_id"] = generate_process_id()
        if "company_id" not in from_dict and (job.company_id or raw_input.get("company_id")):
            from_dict["company_id"] = str(job.company_id or raw_input.get("company_id"))
        if "user_id" not in from_dict and (job.user_id or raw_input.get("user_id")):
            from_dict["user_id"] = str(job.user_id or raw_input.get("user_id"))

        with LoggingContext(from_dict):
            # Check if handler exists
            handler = self._handlers.get(job_type)
            if not handler:
                error_msg = f"No handler registered for job type: {job_type}"
                logger.error(f"[JobWorker] {error_msg}")
                self.job_queue.fail_job(job_id, error_msg)
                return False
            
            logger.info(f"[JobWorker] Processing job {job_id} (type: {job_type})")
            
            heartbeat_stop: Optional[threading.Event] = None
            heartbeat_thread: Optional[threading.Thread] = None
            try:
                # Update progress
                self.job_queue.update_job_progress(job_id, 10, "Starting job processing")
                job.progress = 10
                job.progress_message = "Starting job processing"
                if self.heartbeat_interval > 0:
                    heartbeat_stop = threading.Event()
                    heartbeat_thread = threading.Thread(
                        target=self._heartbeat_job,
                        args=(job, heartbeat_stop, from_dict),
                        name=f"job-heartbeat-{job_id}",
                        daemon=True,
                    )
                    heartbeat_thread.start()
                
                # Call handler
                result = handler.handle(job)

                # Stop heartbeating before publishing a terminal state. This avoids
                # a late heartbeat overwriting the final progress value.
                if heartbeat_stop is not None:
                    heartbeat_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=min(self.heartbeat_interval, 1.0))
                heartbeat_stop = None
                heartbeat_thread = None

                # Handlers use a structured result for expected processing errors.
                # Treat an explicit failure as a terminal job failure instead of
                # publishing it as a completed job and making callers infer failure
                # from result.success.  This lets polling clients stop immediately.
                if isinstance(result, dict) and result.get("success") is False:
                    error_msg = str(
                        result.get("error")
                        or result.get("message")
                        or "Job handler reported an unsuccessful result"
                    )
                    self.job_queue.fail_job(job_id, error_msg)
                    logger.error(
                        "[JobWorker] Job %s failed with a handler result: %s",
                        job_id,
                        error_msg,
                    )
                    return False
                
                # Mark as completed
                self.job_queue.complete_job(job_id, result)
                logger.info(f"[JobWorker] Job {job_id} completed successfully")
                return True
                
            except ConfigurationError as e:
                # Handle configuration errors - these should NOT be retried
                error_type = getattr(e, 'error_type', 'CONFIGURATION_ERROR')
                error_msg = str(e)
                
                logger.warning(
                    f"[JobWorker] Job {job_id} failed due to configuration error [{error_type}]. "
                    f"Marking as failed without retrying: {error_msg}"
                )
                self.job_queue.fail_job(job_id, f"Configuration error: {error_msg}")
                return False
            except (NameError, AttributeError, KeyError, TypeError, AssertionError) as e:
                # These errors indicate a programming or input-contract defect, not
                # a transient dependency failure. Retrying them only extends the
                # time that API and Celery pollers report a job as pending.
                error_msg = f"Job processing failed: {e}"
                logger.error(
                    "[JobWorker] Job %s failed with a non-retryable programming error: %s",
                    job_id,
                    error_msg,
                    exc_info=True,
                )
                self.job_queue.fail_job(job_id, error_msg)
                return False
            except Exception as e:
                error_msg = f"Job processing failed: {str(e)}"
                logger.error(f"[JobWorker] Job {job_id} failed: {error_msg}", exc_info=True)

                # Backward compatibility: Check if this is a configuration error via string prefix
                is_config_error = str(e).startswith("CONFIG_ERROR:")

                if is_config_error:
                    # Configuration errors should not be retried
                    clean_error = str(e).replace("CONFIG_ERROR: ", "")
                    logger.warning(
                        f"[JobWorker] Job {job_id} failed due to configuration error. "
                        f"Marking as failed without retrying: {clean_error}"
                    )
                    self.job_queue.fail_job(job_id, f"Configuration error: {clean_error}")
                    return False

                # Check retry count for other errors
                retry_count = job.metadata.get('retry_count', 0) if job.metadata else 0

                if retry_count < self.max_retries:
                    # Update retry count and requeue
                    metadata = job.metadata or {}
                    metadata['retry_count'] = retry_count + 1
                    metadata['last_error'] = error_msg

                    logger.info(f"[JobWorker] Retrying job {job_id} (attempt {retry_count + 1}/{self.max_retries})")

                    # Persist retry metadata and put the job back on the queue.
                    # Updating only the status would orphan the job after its
                    # original queue entry was popped.
                    if not self.job_queue.requeue_job(job_id, metadata, error_msg):
                        self.job_queue.fail_job(
                            job_id,
                            f"Could not requeue retryable job: {error_msg}",
                        )
                        return False

                    # Wait before retry
                    time.sleep(self.retry_delay)
                else:
                    # Max retries reached, mark as failed
                    self.job_queue.fail_job(job_id, f"{error_msg} (after {retry_count} retries)")

                return False
            finally:
                if heartbeat_stop is not None:
                    heartbeat_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=min(self.heartbeat_interval, 1.0))

    def _heartbeat_job(
        self,
        job: Job,
        stop_event: threading.Event,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Refresh ``updated_at`` while a synchronous handler is still active."""
        with LoggingContext(context or {}):
            while not stop_event.wait(self.heartbeat_interval):
                try:
                    self.job_queue.heartbeat_job(job.job_id)
                except Exception as exc:  # noqa: BLE001 - heartbeat is best effort
                    logger.warning(
                        "[JobWorker] Could not update heartbeat for job %s: %s",
                        job.job_id,
                        exc,
                    )
    
    def start(self, max_jobs: Optional[int] = None):
        """Start the worker loop.
        
        Args:
            max_jobs: Optional maximum number of jobs to process before stopping
        """
        if not self.job_queue:
            raise RuntimeError("Job queue not initialized")
        
        if not self._handlers:
            logger.warning("[JobWorker] No handlers registered")
        
        self._running = True
        self._stop_requested = False
        self._wake.clear()
        jobs_processed = 0

        logger.info("[JobWorker] Worker started")
        logger.info(f"[JobWorker] Registered handlers: {list(self._handlers.keys())}")

        consecutive_errors = 0
        outage_started: Optional[float] = None
        liveness_monitor: Optional[WorkerHeartbeat] = None
        if self.liveness_interval > 0:
            liveness_monitor = WorkerHeartbeat(
                self.job_queue,
                interval=self.liveness_interval,
                liveness=self.liveness,
                stall_timeout=self.stall_timeout,
            )
            liveness_monitor.start()
        try:
            self._wait_for_queue()
            try:
                recovered = self.job_queue.recover_pending_jobs()
                if recovered:
                    logger.info("[JobWorker] Recovered %s pending job(s)", recovered)
                self._recover_stale_jobs()
            except TRANSIENT_QUEUE_ERRORS as e:
                logger.warning("[JobWorker] Could not recover pending jobs: %s", e)

            while self._running and not self._stop_requested:
                # Check if max jobs reached
                if max_jobs and jobs_processed >= max_jobs:
                    logger.info(f"[JobWorker] Reached max jobs limit ({max_jobs})")
                    break

                try:
                    self._recover_stale_jobs()
                    job = self.job_queue.get_pending_job()
                    self._last_poll_at = time.monotonic()
                    if job:
                        self._current_job_id = job.job_id
                        self._job_started_at = time.monotonic()
                        try:
                            self._process_job(job)
                        finally:
                            self._current_job_id = None
                            self._job_started_at = None
                        jobs_processed += 1
                except TRANSIENT_QUEUE_ERRORS as e:
                    # Queue backend unreachable (e.g. Redis/Valkey restarting):
                    # back off and keep polling rather than exiting the loop.
                    consecutive_errors += 1
                    if consecutive_errors == 1:
                        outage_started = time.monotonic()
                        logger.warning("[JobWorker] Job queue unreachable, backing off: %s", e)
                    elif consecutive_errors % 10 == 0:
                        logger.warning(
                            "[JobWorker] Job queue still unreachable after %.0fs: %s",
                            time.monotonic() - (outage_started or time.monotonic()),
                            e,
                        )
                    self._sleep(self._backoff(consecutive_errors))
                    continue

                if consecutive_errors:
                    logger.info(
                        "[JobWorker] Job queue reachable again after %.0fs",
                        time.monotonic() - (outage_started or time.monotonic()),
                    )
                    consecutive_errors = 0
                    outage_started = None

                if not job:
                    # No pending jobs, wait before polling again
                    self._sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.info("[JobWorker] Interrupted by user")
        except Exception as e:
            # Unexpected errors are bugs: re-raise so the process exits non-zero
            # and a supervisor restarts it, instead of returning as if stopped
            # cleanly.
            logger.error(f"[JobWorker] Unexpected error: {e}", exc_info=True)
            raise
        finally:
            if liveness_monitor is not None:
                liveness_monitor.stop()
            self._running = False
            logger.info(f"[JobWorker] Worker stopped (processed {jobs_processed} jobs)")
    
    def stop(self):
        """Stop the worker loop."""
        self._stop_requested = True
        self._wake.set()
    
    def is_running(self) -> bool:
        """Check if worker is running.
        
        Returns:
            True if worker is running
        """
        return self._running
