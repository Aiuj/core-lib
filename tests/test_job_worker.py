import time

from core_lib.jobs.base_job_queue import Job, JobStatus
from core_lib.jobs.job_worker import JobHandler, JobWorker


class _FailureHandler(JobHandler):
    def get_job_type(self) -> str:
        return "failure-test"

    def handle(self, job: Job) -> dict:
        return {
            "success": False,
            "message": "The questionnaire could not be processed",
            "error": "unexpected parser failure",
        }


class _ProgrammingErrorHandler(JobHandler):
    def get_job_type(self) -> str:
        return "programming-error-test"

    def handle(self, job: Job) -> dict:
        raise NameError("payload is not defined")


class _TransientErrorHandler(JobHandler):
    def get_job_type(self) -> str:
        return "transient-error-test"

    def handle(self, job: Job) -> dict:
        raise RuntimeError("temporary service outage")


class _SlowSuccessHandler(JobHandler):
    def get_job_type(self) -> str:
        return "slow-success-test"

    def handle(self, job: Job) -> dict:
        time.sleep(0.04)
        return {"success": True}


class _Queue:
    def __init__(self):
        self.progress_updates = []
        self.heartbeats = []
        self.completed = []
        self.failed = []
        self.requeued = []

    def update_job_progress(self, job_id, progress, message):
        self.progress_updates.append((job_id, progress, message))

    def heartbeat_job(self, job_id):
        self.heartbeats.append(job_id)
        return True

    def complete_job(self, job_id, result):
        self.completed.append((job_id, result))

    def fail_job(self, job_id, error):
        self.failed.append((job_id, error))

    def requeue_job(self, job_id, metadata=None, error=None):
        self.requeued.append((job_id, metadata, error))
        return True

    def recover_pending_jobs(self):
        return 0


def test_worker_marks_unsuccessful_handler_results_as_failed_jobs():
    queue = _Queue()
    worker = JobWorker(job_queue=queue, max_retries=3)
    worker.register_handler(_FailureHandler())
    job = Job(
        job_id="job-123",
        job_type="failure-test",
        status=JobStatus.PROCESSING,
        created_at="2026-08-02T00:00:00Z",
        updated_at="2026-08-02T00:00:00Z",
    )

    assert worker._process_job(job) is False
    assert queue.completed == []
    assert queue.failed == [("job-123", "unexpected parser failure")]


def test_worker_does_not_retry_programming_errors():
    queue = _Queue()
    worker = JobWorker(job_queue=queue, max_retries=3)
    worker.register_handler(_ProgrammingErrorHandler())
    job = Job(
        job_id="job-456",
        job_type="programming-error-test",
        status=JobStatus.PROCESSING,
        created_at="2026-08-02T00:00:00Z",
        updated_at="2026-08-02T00:00:00Z",
    )

    assert worker._process_job(job) is False
    assert queue.completed == []
    assert queue.failed == [("job-456", "Job processing failed: payload is not defined")]


def test_worker_requeues_transient_errors_with_persisted_retry_metadata():
    queue = _Queue()
    worker = JobWorker(job_queue=queue, max_retries=3, retry_delay=0)
    worker.register_handler(_TransientErrorHandler())
    job = Job(
        job_id="job-789",
        job_type="transient-error-test",
        status=JobStatus.PROCESSING,
        created_at="2026-08-02T00:00:00Z",
        updated_at="2026-08-02T00:00:00Z",
        metadata={},
    )

    assert worker._process_job(job) is False
    assert queue.failed == []
    assert queue.requeued == [
        (
            "job-789",
            {
                "retry_count": 1,
                "last_error": "Job processing failed: temporary service outage",
            },
            "Job processing failed: temporary service outage",
        )
    ]


def test_worker_refreshes_job_heartbeat_while_handler_runs():
    queue = _Queue()
    worker = JobWorker(job_queue=queue, heartbeat_interval=0.005)
    worker.register_handler(_SlowSuccessHandler())
    job = Job(
        job_id="job-heartbeat",
        job_type="slow-success-test",
        status=JobStatus.PROCESSING,
        created_at="2026-08-02T00:00:00Z",
        updated_at="2026-08-02T00:00:00Z",
    )

    assert worker._process_job(job) is True
    assert queue.progress_updates == [(
        "job-heartbeat",
        10,
        "Starting job processing",
    )]
    assert len(queue.heartbeats) >= 1
    assert set(queue.heartbeats) == {"job-heartbeat"}


class _ContextVerifyingHandler(JobHandler):
    def __init__(self):
        self.captured_ctx = None

    def get_job_type(self) -> str:
        return "context-verify-test"

    def handle(self, job: Job) -> dict:
        from core_lib.tracing import get_current_logging_context
        self.captured_ctx = get_current_logging_context()
        return {"success": True}


def test_worker_establishes_logging_context():
    from core_lib.tracing import clear_logging_context
    clear_logging_context()

    queue = _Queue()
    worker = JobWorker(job_queue=queue)
    handler = _ContextVerifyingHandler()
    worker.register_handler(handler)

    job = Job(
        job_id="job-ctx-1",
        job_type="context-verify-test",
        status=JobStatus.PROCESSING,
        created_at="2026-08-02T00:00:00Z",
        updated_at="2026-08-02T00:00:00Z",
        company_id="comp-999",
        user_id="user-888",
        input_data={"from": '{"process_id": "pid-777", "session_id": "sess-666"}'},
    )

    assert worker._process_job(job) is True
    assert handler.captured_ctx is not None
    assert handler.captured_ctx["process_id"] == "pid-777"
    assert handler.captured_ctx["session_id"] == "sess-666"
    assert handler.captured_ctx["company_id"] == "comp-999"
    assert handler.captured_ctx["user_id"] == "user-888"


class _FlakyQueue(_Queue):
    """Queue whose polls follow a script of results and exceptions."""

    def __init__(self, script, healthy=None):
        super().__init__()
        self.script = list(script)
        self.healthy = list(healthy or [])
        self.polls = 0

    def health_check(self):
        return self.healthy.pop(0) if self.healthy else True

    def get_pending_job(self):
        self.polls += 1
        step = self.script.pop(0) if self.script else None
        if isinstance(step, BaseException):
            raise step
        return step


def _job(job_id="job-1"):
    return Job(
        job_id=job_id,
        job_type="slow-success-test",
        status=JobStatus.PROCESSING,
        created_at="2026-09-25T00:00:00Z",
        updated_at="2026-09-25T00:00:00Z",
    )


def _fast_worker(queue, **kwargs):
    worker = JobWorker(job_queue=queue, heartbeat_interval=0, **kwargs)
    worker._sleep = lambda seconds: None
    return worker


def test_worker_keeps_polling_through_redis_outage():
    import redis

    queue = _FlakyQueue(
        [
            redis.exceptions.ConnectionError("Error -2 connecting to valkey:6379"),
            redis.exceptions.TimeoutError("timed out"),
            _job(),
        ]
    )
    worker = _fast_worker(queue)
    worker.register_handler(_SlowSuccessHandler())

    worker.start(max_jobs=1)

    assert queue.polls == 3
    assert [job_id for job_id, _ in queue.completed] == ["job-1"]


def test_worker_reraises_unexpected_loop_errors():
    import pytest

    queue = _FlakyQueue([ValueError("corrupt job payload")])
    worker = _fast_worker(queue)

    with pytest.raises(ValueError):
        worker.start()
    assert worker.is_running() is False


def test_worker_waits_for_queue_at_startup():
    queue = _FlakyQueue([_job()], healthy=[False, False, True])
    worker = _fast_worker(queue)
    worker.register_handler(_SlowSuccessHandler())

    worker.start(max_jobs=1)

    assert queue.healthy == []
    assert len(queue.completed) == 1


def test_worker_fails_fast_when_queue_never_becomes_reachable():
    import pytest

    from core_lib.jobs import JobQueueUnavailableError

    class _DownQueue(_FlakyQueue):
        def health_check(self):
            return False

    worker = _fast_worker(_DownQueue([]), startup_timeout=0.05)

    with pytest.raises(JobQueueUnavailableError):
        worker.start()


def test_liveness_reports_last_poll_and_current_job():
    worker = _fast_worker(_FlakyQueue([]))
    assert worker.liveness()["last_poll_age"] is None

    seen = {}

    class _ObservingHandler(JobHandler):
        def get_job_type(self):
            return "slow-success-test"

        def handle(self, job):
            seen.update(worker.liveness())
            return {"success": True}

    worker.register_handler(_ObservingHandler())
    worker.job_queue.script = [_job("job-7")]
    worker.start(max_jobs=1)

    assert seen["current_job_id"] == "job-7"
    assert seen["last_poll_age"] is not None
    assert worker.liveness()["current_job_id"] is None


def test_redis_queue_keeps_client_when_redis_is_down_at_startup():
    from core_lib.jobs import JobConfig, RedisJobQueue

    queue = RedisJobQueue(JobConfig(host="127.0.0.1", port=1, socket_timeout=1))
    queue.connect()

    assert queue.connected is False
    assert queue.client is not None
    assert queue.health_check() is False


def test_worker_publishes_liveness_heartbeat_while_running():
    published = []

    class _HeartbeatQueue(_FlakyQueue):
        def publish_worker_heartbeat(self, worker_id, payload, ttl):
            published.append(payload)

        def remove_worker_heartbeat(self, worker_id):
            published.append("removed")

    queue = _HeartbeatQueue([_job()])
    worker = _fast_worker(queue, liveness_interval=1)
    worker.register_handler(_SlowSuccessHandler())

    worker.start(max_jobs=1)

    assert published[-1] == "removed"


def test_stall_timeout_never_below_startup_wait():
    worker = JobWorker(job_queue=_Queue(), startup_timeout=300, stall_timeout=120)
    assert worker.stall_timeout == 360


def test_worker_reclaims_stale_jobs_at_startup_and_throttles_checks():
    calls = []

    class _StaleQueue(_FlakyQueue):
        def recover_stale_processing_jobs(self, stale_after, max_retries, on_reclaimed=None):
            calls.append((stale_after, max_retries))
            return {"requeued": 1, "failed": 0}

    worker = JobWorker(
        job_queue=_StaleQueue([None, None, _job()]),
        heartbeat_interval=30,
        stale_job_timeout=900,
        max_retries=2,
        liveness_interval=0,
    )
    worker._sleep = lambda seconds: None
    worker.register_handler(_SlowSuccessHandler())

    worker.start(max_jobs=1)

    # Once at startup; the per-poll checks are throttled to once a minute.
    assert calls == [(900, 2)]


def test_stale_job_recovery_disabled_without_per_job_heartbeats():
    worker = JobWorker(job_queue=_Queue(), heartbeat_interval=0, stale_job_timeout=900)
    assert worker.stale_job_timeout == 0


def test_lost_jobs_are_reported_to_their_handler():
    lost = []

    class _AuditedHandler(_SlowSuccessHandler):
        def on_job_lost(self, job, requeued, error):
            lost.append((job.job_id, requeued, error))

    class _StaleQueue(_FlakyQueue):
        def recover_stale_processing_jobs(self, stale_after, max_retries, on_reclaimed=None):
            on_reclaimed(_job("lost-1"), False, "Worker lost while processing")
            other = _job("lost-2")
            other.job_type = "no-handler-for-this"
            on_reclaimed(other, True, "Worker lost while processing")
            return {"requeued": 1, "failed": 1}

    worker = JobWorker(
        job_queue=_StaleQueue([_job()]),
        heartbeat_interval=30,
        stale_job_timeout=900,
        liveness_interval=0,
    )
    worker._sleep = lambda seconds: None
    worker.register_handler(_AuditedHandler())

    worker.start(max_jobs=1)

    assert lost == [("lost-1", False, "Worker lost while processing")]
