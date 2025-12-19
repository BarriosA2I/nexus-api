"""
Nexus Assistant Unified - Async Job Store
In-memory job queue with async processing
"""
import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

from ..config import settings

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Job data structure"""
    id: str
    type: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.id,
            "type": self.type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "brief": self.payload.get("brief", ""),
        }


class JobStore:
    """
    In-memory job store with async processing.

    Features:
    - Async job queue with concurrent execution limit
    - Job status tracking and progress updates
    - Automatic cleanup of old jobs
    - Type-based handler registration
    """

    def __init__(
        self,
        max_concurrent: int = None,
        retention_hours: int = None,
    ):
        self.max_concurrent = max_concurrent or settings.JOB_MAX_CONCURRENT
        self.retention_hours = retention_hours or settings.JOB_RETENTION_HOURS

        self._jobs: Dict[str, Job] = {}
        self._handlers: Dict[str, Callable] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def job_count(self) -> int:
        return len(self._jobs)

    @property
    def pending_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)

    @property
    def running_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)

    def register_handler(self, job_type: str, handler: Callable):
        """
        Register async handler for job type.

        Handler signature: async def handler(job: Job) -> Dict[str, Any]
        """
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    async def start(self):
        """Start job processing workers"""
        if self._running:
            return

        self._running = True
        self._queue = asyncio.Queue()

        # Start worker tasks
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        # Start cleanup task
        cleanup = asyncio.create_task(self._cleanup_worker())
        self._workers.append(cleanup)

        logger.info(f"Job store started with {self.max_concurrent} workers")

    async def stop(self):
        """Stop job processing"""
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

        logger.info("Job store stopped")

    async def submit(
        self,
        job_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """
        Submit a new job for processing.

        Returns the created Job object.
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        job = Job(
            id=job_id,
            type=job_type,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            payload=payload,
            metadata=metadata or {},
        )

        async with self._lock:
            self._jobs[job_id] = job

        # Add to processing queue
        await self._queue.put(job_id)

        logger.info(f"Job submitted: {job_id} (type={job_type})")
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[Job]:
        """List jobs with optional filtering"""
        jobs = list(self._jobs.values())

        if job_type:
            jobs = [j for j in jobs if j.type == job_type]

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def update_progress(self, job_id: str, progress: float):
        """Update job progress (0-100)"""
        job = self._jobs.get(job_id)
        if job:
            job.progress = min(max(progress, 0), 100)
            job.updated_at = datetime.utcnow()

    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending job"""
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            job.updated_at = datetime.utcnow()
            logger.info(f"Job cancelled: {job_id}")
            return True
        return False

    async def _worker(self, worker_id: int):
        """Worker task that processes jobs from queue"""
        logger.info(f"Worker {worker_id} started")

        while self._running:
            try:
                # Wait for job with timeout
                try:
                    job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                job = self._jobs.get(job_id)
                if not job or job.status != JobStatus.PENDING:
                    continue

                # Process job
                await self._process_job(job, worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Worker {worker_id} stopped")

    async def _process_job(self, job: Job, worker_id: int):
        """Process a single job"""
        handler = self._handlers.get(job.type)

        if not handler:
            job.status = JobStatus.FAILED
            job.error = f"No handler registered for job type: {job.type}"
            job.updated_at = datetime.utcnow()
            logger.error(f"No handler for job {job.id}: {job.type}")
            return

        try:
            # Update status to running
            job.status = JobStatus.RUNNING
            job.updated_at = datetime.utcnow()
            logger.info(f"Worker {worker_id} processing job: {job.id}")

            # Execute handler
            result = await handler(job)

            # Update with result
            job.status = JobStatus.COMPLETED
            job.result = result
            job.progress = 100
            job.updated_at = datetime.utcnow()

            logger.info(f"Job completed: {job.id}")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.updated_at = datetime.utcnow()
            logger.error(f"Job failed: {job.id} - {e}")

    async def _cleanup_worker(self):
        """Cleanup old jobs periodically"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
                to_remove = []

                async with self._lock:
                    for job_id, job in self._jobs.items():
                        if job.created_at < cutoff and job.status in (
                            JobStatus.COMPLETED,
                            JobStatus.FAILED,
                            JobStatus.CANCELLED,
                        ):
                            to_remove.append(job_id)

                    for job_id in to_remove:
                        del self._jobs[job_id]

                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old jobs")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def health_info(self) -> Dict[str, Any]:
        """Get health information"""
        return {
            "status": "running" if self._running else "stopped",
            "total_jobs": self.job_count,
            "pending": self.pending_count,
            "running": self.running_count,
            "workers": len(self._workers) - 1,  # Exclude cleanup worker
            "handlers": list(self._handlers.keys()),
        }


# Global job store instance
_job_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    """Get or create job store instance"""
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store
