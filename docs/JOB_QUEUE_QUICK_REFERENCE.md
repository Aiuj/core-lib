# Job Queue System - Quick Reference

## Overview

The `core-lib` job queue system provides a Redis-based async job processing framework for distributed applications. It follows the same design pattern as the cache module.

## Installation

Already included in `core-lib`:

```python
from core_lib.jobs import (
    submit_job, get_job_status, get_job_result,
    update_job_progress, complete_job, fail_job,
    JobWorker, JobHandler, JobStatus, Job
)
```

## Environment Variables

```env
# Job Queue Configuration (falls back to REDIS_* if not set)
JOB_QUEUE_HOST=localhost
JOB_QUEUE_PORT=6379
JOB_QUEUE_DB=0
JOB_QUEUE_PASSWORD=
JOB_QUEUE_PREFIX=jobs:
JOB_QUEUE_TTL=86400                    # 24 hours in seconds
JOB_QUEUE_MAX_CONNECTIONS=10
JOB_QUEUE_RETRY_ON_TIMEOUT=true
JOB_QUEUE_SOCKET_TIMEOUT=5

# Worker liveness (read by JobWorker when not passed explicitly)
JOB_WORKER_HEARTBEAT_INTERVAL=15       # seconds; 0 disables heartbeat + watchdog
JOB_WORKER_STALL_TIMEOUT=600           # seconds; 0 disables the stall watchdog
JOB_WORKER_STALE_JOB_TIMEOUT=600       # seconds; reclaim PROCESSING jobs of dead workers; 0 = off
```

## Quick Start

### 1. Submit a Job

```python
from core_lib.jobs import submit_job

job_id = submit_job(
    job_type="my_task",
    input_data={
        "param1": "value1",
        "param2": "value2"
    },
    company_id="company1",
    user_id="user1",
    session_id="session1",
    metadata={"source": "api"},
    ttl=3600  # Optional, 1 hour
)

print(f"Job submitted: {job_id}")
```

### 2. Check Job Status

```python
from core_lib.jobs import get_job_status, JobStatus

job = get_job_status(job_id)

if job:
    print(f"Status: {job.status.value}")
    print(f"Progress: {job.progress}%")
    print(f"Message: {job.progress_message}")
    
    if job.status == JobStatus.COMPLETED:
        print(f"Result: {job.result}")
    elif job.status == JobStatus.FAILED:
        print(f"Error: {job.error}")
```

### 3. Create a Worker

```python
from core_lib.jobs import JobWorker, JobHandler, Job

# Define a handler
class MyTaskHandler(JobHandler):
    def get_job_type(self) -> str:
        return "my_task"
    
    def handle(self, job: Job) -> dict:
        # Extract input data
        param1 = job.input_data.get("param1")
        param2 = job.input_data.get("param2")
        
        # Process the job
        result = process_something(param1, param2)
        
        # Return result
        return {
            "success": True,
            "data": result
        }

# Create and start worker
worker = JobWorker(
    poll_interval=1.0,
    max_retries=3,
    retry_delay=5.0
)

worker.register_handler(MyTaskHandler())
worker.start()
```

### 4. Function-Based Handler

```python
from core_lib.jobs import JobWorker, Job

def process_my_task(job: Job) -> dict:
    """Simple function handler."""
    param1 = job.input_data.get("param1")
    result = do_something(param1)
    return {"result": result}

worker = JobWorker()
worker.register_function_handler("my_task", process_my_task)
worker.start()
```

## Core API

### Job Submission

```python
submit_job(
    job_type: str,                      # Required: type identifier
    input_data: Optional[Dict] = None,  # Job parameters
    company_id: Optional[str] = None,   # Multi-tenancy
    user_id: Optional[str] = None,      # Audit trail
    session_id: Optional[str] = None,   # Tracking
    metadata: Optional[Dict] = None,    # Additional info
    ttl: Optional[int] = None,          # Time-to-live (seconds)
) -> str  # Returns job_id
```

### Job Status & Results

```python
# Get full job object
job = get_job_status(job_id)

# Get just the result (if completed)
result = get_job_result(job_id)  # Returns None if not completed
```

### Update Job Progress (from worker)

```python
from core_lib.jobs import update_job_progress

update_job_progress(
    job_id="...",
    progress=50,  # 0-100
    message="Processing step 5 of 10"
)
```

### Complete or Fail Job (from worker)

```python
from core_lib.jobs import complete_job, fail_job

# Success
complete_job(job_id, result={"data": "..."})

# Failure
fail_job(job_id, error="Something went wrong")
```

### List Jobs

```python
from core_lib.jobs import list_jobs, JobStatus

# All pending jobs
pending = list_jobs(status=JobStatus.PENDING)

# Jobs for a specific company
company_jobs = list_jobs(company_id="company1", limit=50)

# Jobs for a user
user_jobs = list_jobs(user_id="user1")
```

### Cleanup

```python
from core_lib.jobs import cleanup_old_jobs

# Delete completed/failed jobs older than 24 hours
deleted_count = cleanup_old_jobs(older_than_seconds=86400)
```

## Job Object

```python
@dataclass
class Job:
    job_id: str
    job_type: str
    status: JobStatus  # PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED
    created_at: str
    updated_at: str
    
    # Optional fields
    company_id: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    input_data: Optional[Dict[str, Any]]
    
    # Progress tracking
    progress: int = 0  # 0-100
    progress_message: Optional[str] = None
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
```

## JobStatus Enum

```python
from core_lib.jobs import JobStatus

JobStatus.PENDING       # Job submitted, waiting for worker
JobStatus.PROCESSING    # Worker is processing the job
JobStatus.COMPLETED     # Job completed successfully
JobStatus.FAILED        # Job failed with error
JobStatus.CANCELLED     # Job was cancelled
```

## Worker Configuration

```python
worker = JobWorker(
    job_queue=None,          # Optional: custom queue (auto-initializes if None)
    poll_interval=1.0,       # Seconds between queue checks
    max_retries=3,           # Max retry attempts for failed jobs
    retry_delay=5.0,         # Delay between retries (seconds)
    startup_timeout=300.0,   # Wait this long for Redis at startup, then raise
    max_backoff=30.0,        # Backoff cap while Redis is unreachable
    liveness_interval=None,  # Worker heartbeat period (env / 15s; 0 = off)
    stall_timeout=None,      # Exit if idle loop can't poll this long (env / 600s)
)

# Register handlers
worker.register_handler(MyHandler())
worker.register_function_handler("task_type", my_function)

# Start processing (blocks until stopped)
worker.start(max_jobs=None)  # Optional: limit number of jobs

# Stop worker (from another thread or signal handler)
worker.stop()

# Check if running
if worker.is_running():
    print("Worker is active")
```

### Resilience to Redis/Valkey outages

- **Startup:** `start()` waits up to `startup_timeout` for the queue, retrying with backoff. If Redis is still unreachable, it raises `JobQueueUnavailableError`, so the process exits and gets restarted instead of hanging.
- **While running:** Redis connection and timeout errors don't stop the loop. The worker backs off (1s up to `max_backoff`) and keeps polling. It logs when the outage starts and when it recovers.
- **Unexpected errors** are re-raised from `start()`, so the process exits non-zero. Run workers under a restart policy (e.g. `restart: unless-stopped`).
- `RedisJobQueue.connect()` keeps its client when the first ping fails, and redis-py reconnects on the next command.

### Liveness heartbeat and stall watchdog

While `start()` runs, a `WorkerHeartbeat` thread publishes `{prefix}worker:heartbeat:{host}:{pid}:{id}` with a TTL of 3× `liveness_interval`. It only beats while the poll loop is making progress: a successful poll within the last 60s, or a job in flight. A process that is alive but no longer consuming the queue therefore drops out.

### Lost jobs (worker died mid-job)

A job whose worker is killed mid-job (crash, OOM, deploy, container recreated) would otherwise stay `processing` forever. Running jobs refresh `updated_at` through the per-job heartbeat (`heartbeat_interval`, 30s). At startup and every minute, each worker reclaims a `processing` job if both of these hold:

- its `updated_at` is older than `stale_job_timeout` (default 600s, longer than a typical Redis outage, during which heartbeats can't be written);
- no live worker lists it as its `current_job_id`.

A reclaimed job is re-enqueued, or failed once `max_retries` is used up. An atomic `SREM` on the processing set makes sure only one worker reclaims each job. Handlers must tolerate being run again, as they already must for retries.

The dead worker never got to clean up, so the handler for that job type gets a callback, which may run in a different worker process:

```python
class MyHandler(JobHandler):
    def on_job_lost(self, job, requeued, error):
        # requeued=True: the job will run again. False: it was failed for good.
        # Update app-side records (audit rows), delete temp files, etc.
        ...
```

If the loop is idle and hasn't polled successfully for `stall_timeout` seconds, the watchdog calls `os._exit(1)` so the restart policy replaces the worker. A long-running job never triggers it. `worker.liveness()` exposes `last_poll_age` and `current_job_id`.

## Health Checks (API side)

```python
from core_lib.jobs import check_job_system_health, install_queue_unavailable_handler

# {"job_queue": {"healthy", "latency_ms"}, "background_worker": {"healthy",
#  "active_workers", "workers", "pending_jobs", "processing_jobs",
#  "oldest_pending_age_s"}}
components = check_job_system_health()           # uses the global queue
components = check_job_system_health(enabled=False)  # app runs without a queue

# Redis ConnectionError/TimeoutError on any endpoint -> 503 + Retry-After
install_queue_unavailable_handler(app, detail={"code": "QUEUE_UNAVAILABLE"})
```

`check_job_system_health` makes blocking Redis calls. In an `async def` endpoint, run it with `await asyncio.to_thread(check_job_system_health)`.

## Advanced: Custom Job Queue

```python
from core_lib.jobs import create_job_queue, set_job_queue, JobConfig

# Custom configuration
config = JobConfig(
    host="redis.example.com",
    port=6379,
    db=1,
    password="secret",
    prefix="myapp:jobs:",
    default_ttl=7200  # 2 hours
)

# Create and set custom queue
queue = create_job_queue("redis", config=config)
set_job_queue(queue)

# Now all submit_job() calls use this queue
```

## Error Handling

### In Workers

```python
class MyHandler(JobHandler):
    def handle(self, job: Job) -> dict:
        # Raise exceptions on errors
        if not job.input_data:
            raise ValueError("Missing input data")
        
        # Worker will:
        # 1. Catch the exception
        # 2. Retry up to max_retries
        # 3. Mark as FAILED after max retries
        
        return {"success": True}
```

### In Clients

```python
from core_lib.jobs import get_job_status, JobStatus

job = get_job_status(job_id)

if job.status == JobStatus.FAILED:
    print(f"Job failed: {job.error}")
    
    # Check retry count
    retry_count = job.metadata.get("retry_count", 0) if job.metadata else 0
    print(f"Retries attempted: {retry_count}")
```

## Multi-Tenancy

```python
# Submit job with company_id
job_id = submit_job(
    job_type="task",
    input_data={"data": "..."},
    company_id="company1"  # Tenant isolation
)

# List jobs for specific company
company_jobs = list_jobs(company_id="company1")
```

## Patterns

### Progress Reporting

```python
class LongRunningHandler(JobHandler):
    def handle(self, job: Job) -> dict:
        steps = 10
        for i in range(steps):
            # Update progress
            progress = int((i + 1) / steps * 100)
            update_job_progress(
                job.job_id,
                progress=progress,
                message=f"Processing step {i+1}/{steps}"
            )
            
            # Do work
            process_step(i)
        
        return {"completed": True}
```

### File Processing

```python
import base64

class FileHandler(JobHandler):
    def handle(self, job: Job) -> dict:
        # Get base64 file from input
        file_b64 = job.input_data.get("file_content")
        file_bytes = base64.b64decode(file_b64)
        
        # Process file
        result_bytes = process_file(file_bytes)
        
        # Return result as base64
        result_b64 = base64.b64encode(result_bytes).decode()
        return {
            "file_bytes_b64": result_b64,
            "filename": "result.xlsx"
        }
```

### Cleanup Schedule

```python
import schedule
import time
from core_lib.jobs import cleanup_old_jobs

def cleanup_task():
    deleted = cleanup_old_jobs(older_than_seconds=86400)  # 24h
    print(f"Cleaned up {deleted} old jobs")

# Run cleanup every 6 hours
schedule.every(6).hours.do(cleanup_task)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Testing

### Mock Queue

```python
from core_lib.jobs import set_job_queue
from unittest.mock import MagicMock

# Create mock queue
mock_queue = MagicMock()
mock_queue.submit_job.return_value = "test-job-id"
set_job_queue(mock_queue)

# Test code
job_id = submit_job("test", {"data": "..."})
assert job_id == "test-job-id"
```

### Test Handler

```python
from core_lib.jobs import Job, JobStatus

def test_my_handler():
    handler = MyTaskHandler()
    
    # Create test job
    job = Job(
        job_id="test-123",
        job_type="my_task",
        status=JobStatus.PROCESSING,
        created_at="2025-10-01T00:00:00",
        updated_at="2025-10-01T00:00:00",
        input_data={"param1": "value1"}
    )
    
    # Test handler
    result = handler.handle(job)
    assert result["success"] is True
```

## Comparison with Cache Module

Both follow similar patterns:

| Feature | Cache | Jobs |
|---------|-------|------|
| Backend | Redis/Valkey | Redis/Valkey |
| Singleton | `set_cache()`, `get_cache()` | `set_job_queue()`, `get_job_queue()` |
| Config | `CacheConfig` | `JobConfig` |
| Factory | `create_cache()` | `create_job_queue()` |
| Convenience | `cache_get()`, `cache_set()` | `submit_job()`, `get_job_status()` |

## See Also

- Agent-RFx async API documentation: `docs/ASYNC_JOB_PROCESSING.md`
- Cache module documentation: `docs/cache.md`
- Redis configuration: `docs/ENV_VARIABLES.md`
