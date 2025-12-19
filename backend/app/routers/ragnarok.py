"""
Nexus Assistant Unified - Ragnarok Router
Job management endpoints for RAGNAROK video generation
"""
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ..schemas import (
    RagnarokGenerateRequest,
    RagnarokJob,
    RagnarokJobList,
    JobStatus,
)
from ..services.job_store import get_job_store, JobStatus as JSStatus
from ..services.ragnarok_service import get_ragnarok_service, RagnarokService
from ..services.circuit_breaker import CircuitBreakerOpen

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/nexus/ragnarok", tags=["ragnarok"])


@router.post("/generate", response_model=RagnarokJob)
async def generate(request: RagnarokGenerateRequest):
    """
    Queue a new commercial generation job.

    The job will be processed asynchronously by the RAGNAROK pipeline.
    Poll the job status endpoint to check progress.
    """
    job_store = get_job_store()

    job = await job_store.submit(
        job_type="ragnarok_generate",
        payload={
            "brief": request.brief,
            "industry": request.industry,
            "duration_seconds": request.duration_seconds,
            "platform": request.platform,
            "style": request.style,
            "voice_style": request.voice_style,
            "metadata": request.metadata,
        },
    )

    logger.info(f"RAGNAROK job created: {job.id}")

    return RagnarokJob(
        job_id=job.id,
        status=JobStatus(job.status.value),
        created_at=job.created_at,
        updated_at=job.updated_at,
        progress=job.progress,
        brief=request.brief,
        result=job.result,
        error=job.error,
    )


@router.get("/jobs/{job_id}", response_model=RagnarokJob)
async def get_job(job_id: str):
    """
    Get job status and results.

    Returns current status, progress, and results if completed.
    """
    job_store = get_job_store()
    job = job_store.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return RagnarokJob(
        job_id=job.id,
        status=JobStatus(job.status.value),
        created_at=job.created_at,
        updated_at=job.updated_at,
        progress=job.progress,
        brief=job.payload.get("brief", ""),
        result=job.result,
        error=job.error,
    )


@router.get("/jobs", response_model=RagnarokJobList)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
):
    """
    List recent RAGNAROK jobs.

    Optionally filter by status: pending, running, completed, failed, cancelled
    """
    job_store = get_job_store()

    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = JSStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    jobs = job_store.list_jobs(
        job_type="ragnarok_generate",
        status=status_filter,
        limit=limit,
    )

    return RagnarokJobList(
        jobs=[
            RagnarokJob(
                job_id=job.id,
                status=JobStatus(job.status.value),
                created_at=job.created_at,
                updated_at=job.updated_at,
                progress=job.progress,
                brief=job.payload.get("brief", ""),
                result=job.result,
                error=job.error,
            )
            for job in jobs
        ],
        total=len(jobs),
    )


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a pending job.

    Only jobs in 'pending' status can be cancelled.
    """
    job_store = get_job_store()
    job = job_store.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status != JSStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in {job.status.value} status"
        )

    success = await job_store.cancel(job_id)

    if success:
        return {"message": f"Job {job_id} cancelled", "job_id": job_id}
    else:
        raise HTTPException(status_code=400, detail="Failed to cancel job")


# =============================================================================
# RAGNAROK SERVICE ENDPOINTS (Direct API integration with circuit breaker)
# =============================================================================


@router.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """
    Get RAGNAROK workflow status by workflow_id.

    Returns status, progress, and outputs when complete.
    Uses circuit breaker protection.
    """
    service = get_ragnarok_service()
    result = service.get_job_result(workflow_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow not found: {workflow_id}"
        )

    return {
        "workflow_id": result.workflow_id,
        "status": result.status.value,
        "video_url": result.video_url,
        "script": result.script,
        "cost_usd": result.cost_usd,
        "duration_seconds": result.duration_seconds,
        "quality_score": result.quality_score,
        "error": result.error,
        "created_at": result.created_at.isoformat(),
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        "trace_id": result.trace_id,
    }


@router.post("/direct/submit")
async def direct_submit(
    request: Request,
    business_name: str = Query(..., description="Business name for commercial"),
    formats: str = Query("1080p", description="Output format"),
    priority: int = Query(5, ge=1, le=10, description="Priority 1-10"),
) -> Dict[str, Any]:
    """
    Submit job directly to RAGNAROK via API mode.

    Uses the production RagnarokService with:
    - Circuit breaker protection
    - Trace ID propagation
    - Background polling
    """
    service = get_ragnarok_service()

    # Get trace ID from request state
    trace_id = getattr(request.state, "trace_id", None)

    try:
        result = await service.submit_job(
            business_name=business_name,
            user_id="nexus_user",
            formats=[formats],
            priority=priority,
            trace_id=trace_id,
        )

        return {
            "workflow_id": result.workflow_id,
            "status": result.status.value,
            "tracking_url": result.tracking_url,
            "estimated_cost_usd": result.cost_usd,
            "estimated_duration_seconds": result.duration_seconds,
            "trace_id": result.trace_id,
            "message": "Job submitted. Poll /workflows/{workflow_id} for status.",
        }

    except CircuitBreakerOpen as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAGNAROK service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Direct submit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/service/health")
async def service_health() -> Dict[str, Any]:
    """
    Check RAGNAROK service health including circuit breaker state.
    """
    service = get_ragnarok_service()
    health = await service.health_check()
    circuit = service.get_circuit_state()

    return {
        "ragnarok": health,
        "circuit_breaker": circuit,
    }


@router.get("/service/circuit")
async def circuit_state() -> Dict[str, Any]:
    """
    Get detailed circuit breaker state for RAGNAROK service.
    """
    service = get_ragnarok_service()
    return service.get_circuit_state()
