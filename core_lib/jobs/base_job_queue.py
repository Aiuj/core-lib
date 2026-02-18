"""Base classes for job queue system."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Dict, List
import uuid


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Job data structure."""
    job_id: str
    job_type: str
    status: JobStatus
    created_at: str
    updated_at: str
    company_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Job input data
    input_data: Optional[Dict[str, Any]] = None
    
    # Progress tracking
    progress: int = 0  # 0-100
    progress_message: Optional[str] = None
    
    # Job result
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        data = asdict(self)
        # Convert enum to string
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Create job from dictionary."""
        # Convert status string to enum
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = JobStatus(data['status'])
        return cls(**data)


@dataclass
class JobConfig:
    """Configuration for job queue."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    prefix: str = "jobs:"
    default_ttl: int = 86400  # 24 hours in seconds
    max_connections: int = 10
    retry_on_timeout: bool = True
    socket_timeout: int = 5
    
    @classmethod
    def from_env(cls) -> 'JobConfig':
        """Load configuration from environment variables."""
        import os
        return cls(
            host=os.getenv("JOB_QUEUE_HOST", os.getenv("REDIS_HOST", "localhost")),
            port=int(os.getenv("JOB_QUEUE_PORT", os.getenv("REDIS_PORT", "6379"))),
            db=int(os.getenv("JOB_QUEUE_DB", os.getenv("REDIS_DB", "0"))),
            password=os.getenv("JOB_QUEUE_PASSWORD", os.getenv("REDIS_PASSWORD")),
            prefix=os.getenv("JOB_QUEUE_PREFIX", "jobs:"),
            default_ttl=int(os.getenv("JOB_QUEUE_TTL", "86400")),
            max_connections=int(os.getenv("JOB_QUEUE_MAX_CONNECTIONS", "10")),
            retry_on_timeout=os.getenv("JOB_QUEUE_RETRY_ON_TIMEOUT", "true").lower() == "true",
            socket_timeout=int(os.getenv("JOB_QUEUE_SOCKET_TIMEOUT", "5")),
        )


class BaseJobQueue(ABC):
    """Abstract base class for job queue implementations."""
    
    def __init__(self, config: Optional[JobConfig] = None):
        """Initialize job queue with configuration."""
        self.config = config or JobConfig.from_env()
        self.connected = False
    
    @abstractmethod
    def connect(self):
        """Establish connection to job queue backend."""
        pass
    
    @abstractmethod
    def close(self):
        """Close connection to job queue backend."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if job queue backend is healthy."""
        pass
    
    @abstractmethod
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
        """Submit a new job to the queue.
        
        Args:
            job_type: Type of job (e.g., "ingest_excel", "answer_questionnaire")
            input_data: Input data for the job
            company_id: Optional company identifier for multi-tenancy
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional metadata
            ttl: Optional time-to-live in seconds
            
        Returns:
            Job ID (UUID)
        """
        pass
    
    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job object or None if not found
        """
        pass
    
    @abstractmethod
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None
    ) -> bool:
        """Update job status.
        
        Args:
            job_id: Job identifier
            status: New status
            error: Optional error message (for failed jobs)
            
        Returns:
            True if updated successfully
        """
        pass
    
    @abstractmethod
    def update_job_progress(
        self,
        job_id: str,
        progress: int,
        message: Optional[str] = None
    ) -> bool:
        """Update job progress.
        
        Args:
            job_id: Job identifier
            progress: Progress percentage (0-100)
            message: Optional progress message
            
        Returns:
            True if updated successfully
        """
        pass
    
    @abstractmethod
    def complete_job(
        self,
        job_id: str,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Mark job as completed.
        
        Args:
            job_id: Job identifier
            result: Job result data
            
        Returns:
            True if updated successfully
        """
        pass
    
    @abstractmethod
    def fail_job(
        self,
        job_id: str,
        error: str
    ) -> bool:
        """Mark job as failed.
        
        Args:
            job_id: Job identifier
            error: Error message
            
        Returns:
            True if updated successfully
        """
        pass
    
    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or processing job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancelled successfully
        """
        pass
    
    @abstractmethod
    def get_pending_job(self) -> Optional[Job]:
        """Get the next pending job from the queue.
        
        Returns:
            Job object or None if no pending jobs
        """
        pass
    
    @abstractmethod
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Job]:
        """List jobs with optional filtering.
        
        Args:
            status: Optional status filter
            company_id: Optional company filter
            user_id: Optional user filter
            limit: Maximum number of jobs to return
            
        Returns:
            List of Job objects
        """
        pass
    
    @abstractmethod
    def cleanup_old_jobs(self, older_than_seconds: int = 86400) -> int:
        """Clean up completed/failed jobs older than specified time.
        
        Args:
            older_than_seconds: Delete jobs older than this (default 24h)
            
        Returns:
            Number of jobs deleted
        """
        pass
    
    def _generate_job_id(self) -> str:
        """Generate a unique job ID."""
        return str(uuid.uuid4())
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()
