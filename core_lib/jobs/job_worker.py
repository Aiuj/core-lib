"""Background job worker for processing queued jobs."""

import time
import signal
from typing import Callable, Dict, Any, Optional
from abc import ABC, abstractmethod

from .base_job_queue import BaseJobQueue, JobStatus, Job
from .job_manager import get_job_queue
from core_lib.tracing.logger import get_module_logger
from core_lib.exceptions import ConfigurationError


logger = get_module_logger()


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


class JobWorker:
    """Background worker for processing queued jobs."""
    
    def __init__(
        self,
        job_queue: Optional[BaseJobQueue] = None,
        poll_interval: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """Initialize job worker.
        
        Args:
            job_queue: Job queue instance (uses global if not provided)
            poll_interval: Seconds to wait between queue polls
            max_retries: Maximum number of retries for failed jobs
            retry_delay: Delay between retries in seconds
        """
        self.job_queue = job_queue or get_job_queue()
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._handlers: Dict[str, JobHandler] = {}
        self._running = False
        self._stop_requested = False
        
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
    
    def _process_job(self, job: Job) -> bool:
        """Process a single job.
        
        Args:
            job: Job to process
            
        Returns:
            True if job was processed successfully
        """
        job_id = job.job_id
        job_type = job.job_type
        
        # Check if handler exists
        handler = self._handlers.get(job_type)
        if not handler:
            error_msg = f"No handler registered for job type: {job_type}"
            logger.error(f"[JobWorker] {error_msg}")
            self.job_queue.fail_job(job_id, error_msg)
            return False
        
        logger.info(f"[JobWorker] Processing job {job_id} (type: {job_type})")
        
        try:
            # Update progress
            self.job_queue.update_job_progress(job_id, 10, "Starting job processing")
            
            # Call handler
            result = handler.handle(job)
            
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
                
                # Update job status back to pending for retry
                self.job_queue.update_job_status(job_id, JobStatus.PENDING)
                
                # Wait before retry
                time.sleep(self.retry_delay)
            else:
                # Max retries reached, mark as failed
                self.job_queue.fail_job(job_id, f"{error_msg} (after {retry_count} retries)")
            
            return False
    
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
        jobs_processed = 0
        
        logger.info("[JobWorker] Worker started")
        logger.info(f"[JobWorker] Registered handlers: {list(self._handlers.keys())}")
        
        try:
            while self._running and not self._stop_requested:
                # Check if max jobs reached
                if max_jobs and jobs_processed >= max_jobs:
                    logger.info(f"[JobWorker] Reached max jobs limit ({max_jobs})")
                    break
                
                # Get next pending job
                job = self.job_queue.get_pending_job()
                
                if job:
                    # Process job
                    success = self._process_job(job)
                    jobs_processed += 1
                else:
                    # No pending jobs, wait before polling again
                    time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("[JobWorker] Interrupted by user")
        except Exception as e:
            logger.error(f"[JobWorker] Unexpected error: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info(f"[JobWorker] Worker stopped (processed {jobs_processed} jobs)")
    
    def stop(self):
        """Stop the worker loop."""
        self._stop_requested = True
    
    def is_running(self) -> bool:
        """Check if worker is running.
        
        Returns:
            True if worker is running
        """
        return self._running
