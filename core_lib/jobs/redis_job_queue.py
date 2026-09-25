"""Redis-based job queue implementation."""

import json
import redis
from typing import Any, Callable, Optional, Dict, List
from datetime import datetime, timedelta, timezone

from .base_job_queue import BaseJobQueue, JobConfig, JobStatus, Job
from core_lib.tracing.logger import get_module_logger


logger = get_module_logger()


class RedisJobQueue(BaseJobQueue):
    """Redis-based job queue implementation with connection pooling."""
    
    def __init__(self, config: Optional[JobConfig] = None):
        """Initialize Redis job queue."""
        super().__init__(config)
        self.client: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        
        # Redis key patterns
        self._job_key_prefix = f"{self.config.prefix}job:"
        self._pending_queue_key = f"{self.config.prefix}queue:pending"
        self._processing_set_key = f"{self.config.prefix}set:processing"
        self._status_index_prefix = f"{self.config.prefix}index:status:"
        self._company_index_prefix = f"{self.config.prefix}index:company:"
        self._user_index_prefix = f"{self.config.prefix}index:user:"
        self._worker_heartbeat_prefix = f"{self.config.prefix}worker:heartbeat:"
    
    def _create_connection_pool(self) -> redis.ConnectionPool:
        """Create Redis connection pool."""
        pool_kwargs = {
            'host': self.config.host,
            'port': self.config.port,
            'db': self.config.db,
            'decode_responses': True,
            'socket_connect_timeout': self.config.socket_timeout,
            'socket_timeout': self.config.socket_timeout,
            'max_connections': self.config.max_connections,
            'retry_on_timeout': self.config.retry_on_timeout
        }
        if self.config.password:
            pool_kwargs['password'] = self.config.password
        return redis.ConnectionPool(**pool_kwargs)
    
    def connect(self):
        """Establish connection to Redis server.

        The client is kept even when the initial ping fails: redis-py opens
        connections lazily, so once the server is reachable again the next
        command reconnects on its own.  Dropping the client here would leave
        the queue permanently disconnected after a transient outage at
        startup (e.g. a Valkey container being recreated).
        """
        if self._connection_pool is None:
            self._connection_pool = self._create_connection_pool()
        self.client = redis.Redis(connection_pool=self._connection_pool)

        try:
            self.connected = bool(self.client.ping())
        except Exception as e:
            self.connected = False
            logger.error(
                f"[RedisJobQueue] Could not connect to Redis (will retry on next command): {e}"
            )
            return

        if self.connected:
            logger.info("[RedisJobQueue] Connected to Redis")
        else:
            logger.error("[RedisJobQueue] Redis ping failed")
    
    def close(self):
        """Close connection pool and cleanup resources."""
        if self._connection_pool:
            try:
                self._connection_pool.disconnect()
                logger.info("[RedisJobQueue] Connection pool closed")
            except Exception as e:
                logger.warning(f"[RedisJobQueue] Error closing connection pool: {e}")
            finally:
                self._connection_pool = None
                self.connected = False
                self.client = None
    
    def health_check(self) -> bool:
        """Check if Redis server is healthy."""
        if not self.client:
            return False
        try:
            self.connected = bool(self.client.ping())
        except Exception as e:
            logger.error(f"[RedisJobQueue] Health check failed: {e}")
            self.connected = False
        return self.connected
    
    def _get_job_key(self, job_id: str) -> str:
        """Get Redis key for job data."""
        return f"{self._job_key_prefix}{job_id}"
    
    def _get_status_index_key(self, status: JobStatus) -> str:
        """Get Redis key for status index."""
        return f"{self._status_index_prefix}{status.value}"
    
    def _get_company_index_key(self, company_id: str) -> str:
        """Get Redis key for company index."""
        return f"{self._company_index_prefix}{company_id}"
    
    def _get_user_index_key(self, user_id: str) -> str:
        """Get Redis key for user index."""
        return f"{self._user_index_prefix}{user_id}"
    
    def _add_to_indexes(self, job: Job):
        """Add job to various indexes for efficient querying."""
        if not self.client:
            return
        
        job_id = job.job_id
        
        # Add to status index
        status_key = self._get_status_index_key(job.status)
        self.client.sadd(status_key, job_id)
        
        # Add to company index if present
        if job.company_id:
            company_key = self._get_company_index_key(job.company_id)
            self.client.sadd(company_key, job_id)
        
        # Add to user index if present
        if job.user_id:
            user_key = self._get_user_index_key(job.user_id)
            self.client.sadd(user_key, job_id)
    
    def _remove_from_indexes(self, job: Job, old_status: Optional[JobStatus] = None):
        """Remove job from indexes (used when status changes)."""
        if not self.client:
            return
        
        job_id = job.job_id
        
        # Remove from old status index if provided
        if old_status:
            old_status_key = self._get_status_index_key(old_status)
            self.client.srem(old_status_key, job_id)
    
    def submit_job(
        self,
        job_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Submit a new job to the queue."""
        if not self.client:
            raise RuntimeError("Job queue not connected")
        
        job_id = self._generate_job_id()
        now = self._get_timestamp()
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            company_id=company_id,
            user_id=user_id,
            session_id=session_id,
            input_data=input_data,
            metadata=metadata,
        )
        
        # Store job data
        job_key = self._get_job_key(job_id)
        job_data = json.dumps(job.to_dict())
        
        ttl_value = ttl if ttl is not None else self.config.default_ttl
        self.client.setex(job_key, ttl_value, job_data)
        
        # Add to pending queue (using list for FIFO)
        self.client.rpush(self._pending_queue_key, job_id)
        
        # Add to indexes
        self._add_to_indexes(job)
        
        logger.info(f"[RedisJobQueue] Job {job_id} submitted (type: {job_type})")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        if not self.client:
            return None
        
        job_key = self._get_job_key(job_id)
        job_data = self.client.get(job_key)
        
        if not job_data:
            return None
        
        try:
            job_dict = json.loads(job_data)
            return Job.from_dict(job_dict)
        except Exception as e:
            logger.error(f"[RedisJobQueue] Error parsing job {job_id}: {e}")
            return None
    
    def _update_job(self, job: Job, old_status: Optional[JobStatus] = None) -> bool:
        """Internal method to update job in Redis."""
        if not self.client:
            return False
        
        job.updated_at = self._get_timestamp()
        job_key = self._get_job_key(job.job_id)
        
        # Get TTL from existing key to preserve it
        ttl = self.client.ttl(job_key)
        if ttl <= 0:
            ttl = self.config.default_ttl
        
        # Update job data
        job_data = json.dumps(job.to_dict())
        self.client.setex(job_key, ttl, job_data)
        
        # Update indexes if status changed
        if old_status and old_status != job.status:
            self._remove_from_indexes(job, old_status)
            self._add_to_indexes(job)
        
        return True
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None
    ) -> bool:
        """Update job status."""
        job = self.get_job(job_id)
        if not job:
            return False
        
        old_status = job.status
        job.status = status
        if error:
            job.error = error
        
        return self._update_job(job, old_status)

    def requeue_job(
        self,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Persist retry metadata and return a job to the FIFO pending queue."""
        if not self.client:
            return False

        job = self.get_job(job_id)
        if not job:
            return False

        old_status = job.status
        job.status = JobStatus.PENDING
        job.progress_message = "Retry scheduled"
        job.metadata = metadata if metadata is not None else job.metadata
        job.error = error

        self.client.srem(self._processing_set_key, job_id)
        if not self._update_job(job, old_status):
            return False

        # A retried job has already been popped from the list. Remove any stale
        # duplicate before adding the single authoritative retry entry.
        self.client.lrem(self._pending_queue_key, 0, job_id)
        self.client.rpush(self._pending_queue_key, job_id)
        logger.info("[RedisJobQueue] Job %s re-enqueued for retry", job_id)
        return True

    def recover_pending_jobs(self) -> int:
        """Restore pending jobs orphaned by an interrupted retry or restart."""
        if not self.client:
            return 0

        pending_ids = set(self.client.smembers(self._get_status_index_key(JobStatus.PENDING)))
        queued_ids = set(self.client.lrange(self._pending_queue_key, 0, -1))
        orphaned_ids = pending_ids - queued_ids
        for job_id in orphaned_ids:
            self.client.rpush(self._pending_queue_key, job_id)

        if orphaned_ids:
            logger.warning(
                "[RedisJobQueue] Re-enqueued %s orphaned pending job(s): %s",
                len(orphaned_ids),
                ", ".join(sorted(orphaned_ids)),
            )
        return len(orphaned_ids)
    
    def update_job_progress(
        self,
        job_id: str,
        progress: int,
        message: Optional[str] = None
    ) -> bool:
        """Update job progress."""
        job = self.get_job(job_id)
        if not job:
            return False
        
        job.progress = max(0, min(100, progress))  # Clamp to 0-100
        if message:
            job.progress_message = message
        
        return self._update_job(job)

    def heartbeat_job(self, job_id: str) -> bool:
        """Refresh ``updated_at`` without overwriting handler-owned progress."""
        job = self.get_job(job_id)
        if not job or job.status != JobStatus.PROCESSING:
            return False
        return self._update_job(job)
    
    def complete_job(
        self,
        job_id: str,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Mark job as completed."""
        if not self.client:
            return False
        
        job = self.get_job(job_id)
        if not job:
            return False
        
        old_status = job.status
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.result = result
        
        # Remove from processing set if present
        self.client.srem(self._processing_set_key, job_id)
        
        return self._update_job(job, old_status)
    
    def fail_job(
        self,
        job_id: str,
        error: str
    ) -> bool:
        """Mark job as failed."""
        if not self.client:
            return False
        
        job = self.get_job(job_id)
        if not job:
            return False
        
        old_status = job.status
        job.status = JobStatus.FAILED
        job.error = error
        
        # Remove from processing set if present
        self.client.srem(self._processing_set_key, job_id)
        
        return self._update_job(job, old_status)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or processing job."""
        if not self.client:
            return False
        
        job = self.get_job(job_id)
        if not job:
            return False
        
        # Can only cancel pending or processing jobs
        if job.status not in [JobStatus.PENDING, JobStatus.PROCESSING]:
            return False
        
        old_status = job.status
        job.status = JobStatus.CANCELLED
        
        # Remove from pending queue if present
        self.client.lrem(self._pending_queue_key, 0, job_id)
        
        # Remove from processing set if present
        self.client.srem(self._processing_set_key, job_id)
        
        return self._update_job(job, old_status)
    
    def get_pending_job(self) -> Optional[Job]:
        """Get the next pending job from the queue."""
        if not self.client:
            return None
        
        # Pop job from pending queue (FIFO)
        job_id = self.client.lpop(self._pending_queue_key)
        if not job_id:
            return None
        
        # Get job data
        job = self.get_job(job_id)
        if not job:
            logger.warning(f"[RedisJobQueue] Job {job_id} in queue but not found in storage")
            return None
        
        # Update status to processing
        old_status = job.status
        job.status = JobStatus.PROCESSING
        
        # Add to processing set
        self.client.sadd(self._processing_set_key, job_id)
        
        # Update job
        self._update_job(job, old_status)
        
        logger.info(f"[RedisJobQueue] Job {job_id} moved to processing")
        return job
    
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Job]:
        """List jobs with optional filtering."""
        if not self.client:
            return []
        
        job_ids = set()
        
        # Get job IDs from appropriate indexes
        if status:
            status_key = self._get_status_index_key(status)
            job_ids = set(self.client.smembers(status_key))
        elif company_id:
            company_key = self._get_company_index_key(company_id)
            job_ids = set(self.client.smembers(company_key))
        elif user_id:
            user_key = self._get_user_index_key(user_id)
            job_ids = set(self.client.smembers(user_key))
        else:
            # Get all job IDs (scan for job keys)
            cursor = 0
            pattern = f"{self._job_key_prefix}*"
            while True:
                cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=100)
                for key in keys:
                    job_id = key.replace(self._job_key_prefix, '')
                    job_ids.add(job_id)
                if cursor == 0:
                    break
        
        # Fetch jobs
        jobs = []
        for job_id in list(job_ids)[:limit]:
            job = self.get_job(job_id)
            if job:
                # Apply additional filters
                if status and job.status != status:
                    continue
                if company_id and job.company_id != company_id:
                    continue
                if user_id and job.user_id != user_id:
                    continue
                jobs.append(job)
        
        # Sort by created_at (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs[:limit]
    
    def cleanup_old_jobs(self, older_than_seconds: int = 86400) -> int:
        """Clean up completed/failed jobs older than specified time."""
        if not self.client:
            return 0
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
        deleted_count = 0
        
        # Get completed and failed jobs
        for status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            status_key = self._get_status_index_key(status)
            job_ids = self.client.smembers(status_key)
            
            for job_id in job_ids:
                job = self.get_job(job_id)
                if not job:
                    # Job key expired, remove from index
                    self.client.srem(status_key, job_id)
                    continue
                
                # Check if job is old enough
                try:
                    created_at = datetime.fromisoformat(job.created_at)
                    if created_at < cutoff_time:
                        # Delete job
                        job_key = self._get_job_key(job_id)
                        self.client.delete(job_key)
                        
                        # Remove from indexes
                        self._remove_from_indexes(job, job.status)
                        if job.company_id:
                            self.client.srem(self._get_company_index_key(job.company_id), job_id)
                        if job.user_id:
                            self.client.srem(self._get_user_index_key(job.user_id), job_id)
                        
                        deleted_count += 1
                except Exception as e:
                    logger.error(f"[RedisJobQueue] Error cleaning up job {job_id}: {e}")
        
        if deleted_count > 0:
            logger.info(f"[RedisJobQueue] Cleaned up {deleted_count} old jobs")
        
        return deleted_count

    # --- Worker liveness -------------------------------------------------

    def _worker_heartbeat_key(self, worker_id: str) -> str:
        return f"{self._worker_heartbeat_prefix}{worker_id}"

    def publish_worker_heartbeat(
        self, worker_id: str, payload: Dict[str, Any], ttl: int
    ) -> None:
        """Write the worker's heartbeat with a TTL; it vanishes if not refreshed."""
        if not self.client:
            raise RuntimeError("Job queue not connected")
        self.client.set(self._worker_heartbeat_key(worker_id), json.dumps(payload), ex=ttl)

    def remove_worker_heartbeat(self, worker_id: str) -> None:
        if self.client:
            self.client.delete(self._worker_heartbeat_key(worker_id))

    def list_worker_heartbeats(self) -> List[Dict[str, Any]]:
        if not self.client:
            raise RuntimeError("Job queue not connected")
        keys = list(self.client.scan_iter(match=f"{self._worker_heartbeat_prefix}*", count=100))
        workers: List[Dict[str, Any]] = []
        for raw in self.client.mget(keys) if keys else []:
            if not raw:
                continue  # expired between SCAN and MGET
            try:
                workers.append(json.loads(raw))
            except (TypeError, ValueError):
                continue
        return workers

    def recover_stale_processing_jobs(
        self,
        stale_after: float,
        max_retries: int,
        on_reclaimed: Optional[Callable[[Job, bool, str], None]] = None,
    ) -> Dict[str, int]:
        """Reclaim jobs left in PROCESSING by a worker that died mid-job.

        A running job's ``updated_at`` is refreshed by the worker's per-job
        heartbeat, so a job whose ``updated_at`` is older than ``stale_after``
        seconds *and* that no live worker reports as its ``current_job_id``
        has lost its worker (crash, OOM, container recreated, deploy).  It is
        re-enqueued, or failed once it has used up ``max_retries``, instead of
        showing as "processing" forever.

        The atomic SREM on the processing set is the claim: when several
        workers run this concurrently only one reclaims each job.

        ``on_reclaimed(job, requeued, error)`` is called after each job's new
        state is stored, so the app can update its own records (audit rows,
        temp files).  Errors raised by it are logged and ignored.
        """
        counts = {"requeued": 0, "failed": 0}
        if not self.client:
            return counts

        active = {
            w.get("current_job_id") for w in self.list_worker_heartbeats()
        } - {None}
        now = datetime.now(timezone.utc)

        for job_id in list(self.client.smembers(self._processing_set_key)):
            if job_id in active:
                continue
            job = self.get_job(job_id)
            if job is None:
                # Job record expired: drop the dangling processing entry.
                self.client.srem(self._processing_set_key, job_id)
                continue
            if job.status != JobStatus.PROCESSING:
                continue
            try:
                updated = datetime.fromisoformat(job.updated_at)
            except (TypeError, ValueError):
                continue
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            idle = (now - updated).total_seconds()
            if idle <= stale_after:
                continue
            if not self.client.srem(self._processing_set_key, job_id):
                continue  # another worker claimed it first

            error = f"Worker lost while processing (no heartbeat for {idle:.0f}s)"
            metadata = dict(job.metadata or {})
            retry_count = metadata.get("retry_count", 0)
            if retry_count < max_retries:
                metadata["retry_count"] = retry_count + 1
                metadata["last_error"] = error
                if self.requeue_job(job_id, metadata, error):
                    counts["requeued"] += 1
                    logger.warning("[RedisJobQueue] Re-enqueued stale job %s: %s", job_id, error)
                    self._notify_reclaimed(on_reclaimed, job, True, error)
                    continue
            error = f"{error} (after {retry_count} retries)"
            self.fail_job(job_id, error)
            counts["failed"] += 1
            logger.warning("[RedisJobQueue] Failed stale job %s: %s", job_id, error)
            self._notify_reclaimed(on_reclaimed, job, False, error)
        return counts

    @staticmethod
    def _notify_reclaimed(
        callback: Optional[Callable[[Job, bool, str], None]],
        job: Job,
        requeued: bool,
        error: str,
    ) -> None:
        if callback is None:
            return
        try:
            callback(job, requeued, error)
        except Exception as exc:  # noqa: BLE001 — app hook must not break recovery
            logger.warning(
                "[RedisJobQueue] on_reclaimed hook failed for job %s: %s", job.job_id, exc
            )

    def get_queue_stats(self) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("Job queue not connected")
        stats: Dict[str, Any] = {
            "pending_jobs": self.client.llen(self._pending_queue_key),
            "processing_jobs": self.client.scard(self._processing_set_key),
            "oldest_pending_age_s": None,
        }
        # Head of the list is the next job to be popped, i.e. the one that has
        # waited longest.  A large age with no progress means no worker is
        # consuming the queue, even if a heartbeat is present.
        head_id = self.client.lindex(self._pending_queue_key, 0)
        head = self.get_job(head_id) if head_id else None
        if head is not None:
            try:
                created = datetime.fromisoformat(head.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                stats["oldest_pending_age_s"] = round(
                    (datetime.now(timezone.utc) - created).total_seconds(), 1
                )
            except (TypeError, ValueError):
                pass
        return stats
