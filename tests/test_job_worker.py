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


class _Queue:
    def __init__(self):
        self.progress_updates = []
        self.completed = []
        self.failed = []
        self.requeued = []

    def update_job_progress(self, job_id, progress, message):
        self.progress_updates.append((job_id, progress, message))

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
