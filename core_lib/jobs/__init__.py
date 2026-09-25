"""Job queue module for core-lib."""

from .job_manager import (
    create_job_queue,
    set_job_queue,
    get_job_queue,
    submit_job,
    get_job_status,
    get_job_result,
    update_job_status,
    update_job_progress,
    complete_job,
    fail_job,
    cancel_job,
    list_jobs,
    cleanup_old_jobs,
)
from .base_job_queue import BaseJobQueue, JobConfig, JobStatus, Job
from .redis_job_queue import RedisJobQueue
from .job_worker import JobWorker, JobHandler, JobQueueUnavailableError
from .health import (
    WorkerHeartbeat,
    check_job_system_health,
    check_worker_health,
    install_queue_unavailable_handler,
    ping_job_queue,
)

__all__ = [
    'BaseJobQueue', 'JobConfig', 'JobStatus', 'Job',
    'RedisJobQueue',
    'JobWorker', 'JobHandler', 'JobQueueUnavailableError',
    'WorkerHeartbeat', 'check_job_system_health', 'check_worker_health',
    'install_queue_unavailable_handler', 'ping_job_queue',
    'create_job_queue', 'set_job_queue', 'get_job_queue',
    'submit_job', 'get_job_status', 'get_job_result',
    'update_job_status', 'update_job_progress',
    'complete_job', 'fail_job', 'cancel_job',
    'list_jobs', 'cleanup_old_jobs'
]
