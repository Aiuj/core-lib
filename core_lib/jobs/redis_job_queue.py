"""Redis-based job queue implementation."""

import json
import redis
from typing import Any, Optional, Dict, List
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
        """Establish connection to Redis server."""
        try:
            if self._connection_pool is None:
                self._connection_pool = self._create_connection_pool()
            self.client = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            if self.client.ping():
                self.connected = True
                logger.info("[RedisJobQueue] Connected to Redis")
            else:
                self.connected = False
                logger.error("[RedisJobQueue] Redis ping failed")
        except Exception as e:
            logger.error(f"[RedisJobQueue] Could not connect to Redis: {e}")
            self.connected = False
            self.client = None
    
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
        if not self.client or not self.connected:
            return False
        try:
            return bool(self.client.ping())
        except Exception as e:
            logger.error(f"[RedisJobQueue] Health check failed: {e}")
            return False
    
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
