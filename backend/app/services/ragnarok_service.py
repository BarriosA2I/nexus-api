"""
Nexus Assistant Unified - RAGNAROK Service
Production-grade integration with circuit breaker, polling, and trace propagation.
"""
import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from .circuit_breaker import CircuitBreaker, CircuitBreakerOpen, get_circuit_breaker

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

RAGNAROK_BASE_URL = os.getenv("RAGNAROK_API_URL", "http://localhost:8001")
POLL_INTERVAL_SECONDS = float(os.getenv("RAGNAROK_POLL_INTERVAL", "3"))
MAX_WAIT_SECONDS = float(os.getenv("RAGNAROK_MAX_WAIT", "360"))
REQUEST_TIMEOUT = float(os.getenv("RAGNAROK_TIMEOUT", "30"))

# Endpoints discovered from OpenAPI
GENERATE_ENDPOINT = "/api/v1/commercial/generate"
STATUS_ENDPOINT = "/api/v1/workflows/{workflow_id}"
HEALTH_ENDPOINT = "/health"


# =============================================================================
# DATA MODELS
# =============================================================================

class JobStatus(str, Enum):
    """RAGNAROK job status values"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class RagnarokJobResult:
    """Normalized output from RAGNAROK pipeline"""
    workflow_id: str
    status: JobStatus
    video_url: Optional[str] = None
    tracking_url: Optional[str] = None
    script: Optional[str] = None
    scene_prompts: Optional[List[str]] = None
    cost_usd: Optional[float] = None
    duration_seconds: Optional[float] = None
    quality_score: Optional[float] = None
    provider: str = "ragnarok_v7_apex"
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# =============================================================================
# RAGNAROK SERVICE
# =============================================================================

class RagnarokService:
    """
    Production-grade RAGNAROK integration service.

    Features:
    - Circuit breaker protection (5 failures → open for 30s)
    - Async job submission with instant response
    - Background polling with configurable intervals
    - Trace ID propagation
    - Normalized output schema
    """

    def __init__(
        self,
        base_url: str = RAGNAROK_BASE_URL,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.circuit_breaker = circuit_breaker or get_circuit_breaker("ragnarok_service")
        self._client: Optional[httpx.AsyncClient] = None
        self._active_polls: Dict[str, asyncio.Task] = {}
        self._results_cache: Dict[str, RagnarokJobResult] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with extended timeout configuration"""
        if self._client is None or self._client.is_closed:
            # Extended timeout config:
            # - connect: 10s (fail fast if service is down)
            # - read: 300s (wait for blocking generation if necessary)
            # - write: 30s (sending payload)
            timeout_config = httpx.Timeout(
                timeout=300.0,  # Default for operations
                connect=10.0,
                read=300.0,
                write=30.0,
            )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout_config,
                headers={"User-Agent": "NexusAssistant/1.0"},
            )
        return self._client

    async def close(self):
        """Cleanup resources"""
        # Cancel all active polls
        for task in self._active_polls.values():
            task.cancel()
        self._active_polls.clear()

        # Close HTTP client
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> Dict[str, Any]:
        """Check RAGNAROK health"""
        try:
            client = await self._get_client()
            response = await client.get(HEALTH_ENDPOINT)
            response.raise_for_status()
            return {
                "status": "healthy",
                "base_url": self.base_url,
                "circuit_state": self.circuit_breaker.state.value,
                "response": response.json(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "base_url": self.base_url,
                "circuit_state": self.circuit_breaker.state.value,
                "error": str(e),
            }

    async def submit_job(
        self,
        business_name: str,
        user_id: str = "nexus_user",
        formats: Optional[List[str]] = None,
        priority: int = 5,
        trace_id: Optional[str] = None,
    ) -> RagnarokJobResult:
        """
        Submit a job to RAGNAROK and start background polling.

        Returns immediately with pending status.
        Use get_job_result() to check completion.
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerOpen(
                f"Circuit breaker is {self.circuit_breaker.state.value}"
            )

        try:
            client = await self._get_client()

            # Build request
            payload = {
                "business_name": business_name,
                "user_id": user_id,
                "formats": formats or ["1080p"],
                "priority": priority,
            }

            headers = {}
            if trace_id:
                headers["X-Trace-Id"] = trace_id

            # Submit to RAGNAROK
            logger.info(
                f"[ragnarok.submit] trace_id={trace_id} | "
                f"Submitting job for {business_name}"
            )

            response = await client.post(
                GENERATE_ENDPOINT,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            # Record success
            self.circuit_breaker.record_success()

            # Create result object
            workflow_id = data.get("workflow_id")
            result = RagnarokJobResult(
                workflow_id=workflow_id,
                status=JobStatus(data.get("status", "submitted")),
                tracking_url=data.get("tracking_url"),
                cost_usd=data.get("estimated_cost_dollars"),
                duration_seconds=data.get("estimated_duration_seconds"),
                trace_id=trace_id,
                raw=data,
            )

            # Cache and start polling
            self._results_cache[workflow_id] = result
            self._start_polling(workflow_id, trace_id)

            logger.info(
                f"[ragnarok.submit] trace_id={trace_id} | "
                f"Job submitted: workflow_id={workflow_id}"
            )

            return result

        except httpx.HTTPStatusError as e:
            self.circuit_breaker.record_failure()
            # Try to extract error detail from response
            error_detail = str(e)
            try:
                error_json = e.response.json()
                error_detail = error_json.get("detail", error_json.get("error", str(e)))
            except Exception:
                error_detail = e.response.text or str(e)
            logger.error(
                f"[ragnarok.submit] trace_id={trace_id} | "
                f"HTTP error {e.response.status_code}: {error_detail}"
            )
            raise Exception(f"RAGNAROK API error ({e.response.status_code}): {error_detail}")

        except httpx.ReadTimeout:
            # ReadTimeout means RAGNAROK is working but slow (blocking API)
            # Don't crash; assume the job is running on their side
            logger.warning(
                f"[ragnarok.submit] trace_id={trace_id} | "
                f"Read timeout on submission - RAGNAROK may be processing synchronously"
            )
            # Create a pending result - job may be running downstream
            pending_workflow_id = f"pending_{uuid.uuid4().hex[:8]}"
            result = RagnarokJobResult(
                workflow_id=pending_workflow_id,
                status=JobStatus.RUNNING,
                trace_id=trace_id,
                error="Submission timed out, but job may be running on RAGNAROK side",
            )
            self._results_cache[pending_workflow_id] = result
            return result

        except httpx.ConnectError as e:
            self.circuit_breaker.record_failure()
            logger.error(
                f"[ragnarok.submit] trace_id={trace_id} | "
                f"Connection error: {e}"
            )
            raise Exception(f"Cannot connect to RAGNAROK at {self.base_url}: {e}")

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(
                f"[ragnarok.submit] trace_id={trace_id} | "
                f"Error: {e}"
            )
            raise

    def _start_polling(self, workflow_id: str, trace_id: Optional[str] = None):
        """Start background polling for workflow status"""
        if workflow_id in self._active_polls:
            return  # Already polling

        task = asyncio.create_task(
            self._poll_status(workflow_id, trace_id)
        )
        self._active_polls[workflow_id] = task

        # Cleanup when done
        task.add_done_callback(
            lambda t: self._active_polls.pop(workflow_id, None)
        )

    async def _poll_status(
        self,
        workflow_id: str,
        trace_id: Optional[str] = None,
    ):
        """Poll RAGNAROK for workflow status until completion"""
        start_time = time.time()
        poll_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > MAX_WAIT_SECONDS:
                logger.warning(
                    f"[ragnarok.poll] trace_id={trace_id} | "
                    f"Timeout after {elapsed:.1f}s for {workflow_id}"
                )
                if workflow_id in self._results_cache:
                    self._results_cache[workflow_id].status = JobStatus.TIMEOUT
                    self._results_cache[workflow_id].error = f"Timeout after {MAX_WAIT_SECONDS}s"
                return

            try:
                poll_count += 1
                client = await self._get_client()

                endpoint = STATUS_ENDPOINT.format(workflow_id=workflow_id)
                response = await client.get(endpoint)
                response.raise_for_status()
                data = response.json()

                status = data.get("status", "running")

                # Update cached result
                if workflow_id in self._results_cache:
                    result = self._results_cache[workflow_id]
                    result.status = JobStatus(status)
                    result.raw = data

                    if status == "completed":
                        result.completed_at = datetime.utcnow()
                        result.cost_usd = data.get("cost_dollars")
                        result.duration_seconds = data.get("duration_seconds")
                        result.quality_score = data.get("quality_score")

                        # Extract video URL from outputs
                        outputs = data.get("outputs", {})
                        if outputs:
                            result.video_url = list(outputs.values())[0]

                        logger.info(
                            f"[ragnarok.poll] trace_id={trace_id} | "
                            f"Completed: workflow_id={workflow_id}, "
                            f"polls={poll_count}, elapsed={elapsed:.1f}s"
                        )
                        return

                    elif status == "failed":
                        result.error = data.get("error", "Unknown error")
                        logger.error(
                            f"[ragnarok.poll] trace_id={trace_id} | "
                            f"Failed: workflow_id={workflow_id}, error={result.error}"
                        )
                        return

                # Continue polling
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

            except asyncio.CancelledError:
                logger.info(
                    f"[ragnarok.poll] trace_id={trace_id} | "
                    f"Polling cancelled for {workflow_id}"
                )
                return
            except Exception as e:
                logger.warning(
                    f"[ragnarok.poll] trace_id={trace_id} | "
                    f"Poll error for {workflow_id}: {e}"
                )
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def get_job_result(self, workflow_id: str) -> Optional[RagnarokJobResult]:
        """Get cached job result"""
        return self._results_cache.get(workflow_id)

    def get_circuit_state(self) -> Dict[str, Any]:
        """Get circuit breaker state for debugging"""
        return {
            "name": self.circuit_breaker.name,
            "state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count,
            "last_failure": (
                self.circuit_breaker.last_failure_time.isoformat()
                if self.circuit_breaker.last_failure_time else None
            ),
            "active_polls": len(self._active_polls),
            "cached_results": len(self._results_cache),
        }

    async def wait_for_completion(
        self,
        workflow_id: str,
        timeout: float = MAX_WAIT_SECONDS,
    ) -> RagnarokJobResult:
        """
        Wait for a job to complete.

        Blocks until job completes, fails, or times out.
        """
        start = time.time()

        while time.time() - start < timeout:
            result = self.get_job_result(workflow_id)
            if result:
                if result.status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.TIMEOUT,
                ):
                    return result
            await asyncio.sleep(0.5)

        # Timeout
        result = self.get_job_result(workflow_id)
        if result:
            result.status = JobStatus.TIMEOUT
            result.error = f"Wait timeout after {timeout}s"
            return result

        return RagnarokJobResult(
            workflow_id=workflow_id,
            status=JobStatus.TIMEOUT,
            error=f"Wait timeout after {timeout}s",
        )


# =============================================================================
# SINGLETON
# =============================================================================

_ragnarok_service: Optional[RagnarokService] = None


def get_ragnarok_service() -> RagnarokService:
    """Get or create RAGNAROK service singleton"""
    global _ragnarok_service
    if _ragnarok_service is None:
        _ragnarok_service = RagnarokService()
    return _ragnarok_service


async def shutdown_ragnarok_service():
    """Shutdown RAGNAROK service gracefully"""
    global _ragnarok_service
    if _ragnarok_service:
        await _ragnarok_service.close()
        _ragnarok_service = None
