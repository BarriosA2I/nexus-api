"""
Nexus Assistant Unified - Ragnarok Router
Gateway Proxy for RAGNAROK video generation.

ARCHITECTURE PRINCIPLE: Nexus is the TRUE GATEWAY.
- Frontend ONLY calls Nexus endpoints
- Frontend ONLY polls Nexus job status
- Frontend NEVER needs to know about RAGNAROK's port or API schema
"""
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ..schemas import (
    RagnarokGenerateRequest,
    RagnarokJob,
    RagnarokJobList,
    JobStatus,
)
from ..services.job_store import get_job_store, JobStatus as JSStatus
from ..services.ragnarok_service import (
    get_ragnarok_service,
    RagnarokService,
    RagnarokAPIError,
)
from ..services.circuit_breaker import CircuitBreakerOpen
from ..services.trinity_service import get_trinity_service

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


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    """
    Get job status and results.

    Returns current status, progress, and results if completed.
    This endpoint UNIFIES polling - it checks both the Nexus job store
    AND the RAGNAROK workflow status if the job has a workflow_id.
    """
    job_store = get_job_store()
    service = get_ragnarok_service()

    job = job_store.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Build base response
    response = {
        "job_id": job.id,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "progress": job.progress,
        "brief": job.payload.get("brief", ""),
        "error": job.error,
    }

    # If this job has a workflow_id, fetch the RAGNAROK status
    workflow_id = job.metadata.get("workflow_id") or job.payload.get("workflow_id")
    if workflow_id:
        response["workflow_id"] = workflow_id

        # Check our cached result from background polling
        ragnarok_result = service.get_job_result(workflow_id)
        if ragnarok_result:
            response["ragnarok_status"] = ragnarok_result.status.value
            response["video_url"] = ragnarok_result.video_url
            response["cost_usd"] = ragnarok_result.cost_usd
            response["quality_score"] = ragnarok_result.quality_score
            response["duration_seconds"] = ragnarok_result.duration_seconds

            # Sync status with RAGNAROK result
            if ragnarok_result.status.value == "completed":
                response["status"] = "completed"
                response["result"] = {
                    "video_url": ragnarok_result.video_url,
                    "cost_usd": ragnarok_result.cost_usd,
                    "quality_score": ragnarok_result.quality_score,
                    "duration_seconds": ragnarok_result.duration_seconds,
                }
            elif ragnarok_result.status.value == "failed":
                response["status"] = "failed"
                response["error"] = ragnarok_result.error
            elif ragnarok_result.status.value in ("running", "submitted"):
                response["status"] = "running"

    # Include enrichment data if present
    if job.payload.get("enrichment"):
        response["enrichment"] = job.payload["enrichment"]

    # Include trace_id if present
    if job.metadata.get("trace_id"):
        response["trace_id"] = job.metadata["trace_id"]

    return response


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
    - Unified Nexus job tracking

    POLLING: Use GET /api/nexus/ragnarok/jobs/{job_id} - NEVER poll RAGNAROK directly.
    """
    service = get_ragnarok_service()
    job_store = get_job_store()

    # Get trace ID from request state
    trace_id = getattr(request.state, "trace_id", f"nxs_{int(time.time())}_{__import__('uuid').uuid4().hex[:8]}")

    try:
        ragnarok_result = await service.submit_job(
            business_name=business_name,
            user_id="nexus_user",
            formats=[formats],
            priority=priority,
            trace_id=trace_id,
        )

        # Create a Nexus tracking job that wraps the RAGNAROK workflow
        nexus_job = await job_store.submit(
            job_type="ragnarok_direct",
            payload={
                "brief": f"Commercial for {business_name}",
                "business_name": business_name,
                "formats": [formats],
                "workflow_id": ragnarok_result.workflow_id,
            },
            metadata={
                "workflow_id": ragnarok_result.workflow_id,
                "trace_id": trace_id,
                "type": "direct",
            }
        )

        # Store the mapping for unified polling
        service._results_cache[nexus_job.id] = ragnarok_result

        logger.info(
            f"Direct job created | job_id={nexus_job.id} | "
            f"workflow_id={ragnarok_result.workflow_id} | trace={trace_id}"
        )

        return {
            "job_id": nexus_job.id,
            "workflow_id": ragnarok_result.workflow_id,
            "status": "queued",
            "tracking_url": f"/api/nexus/ragnarok/jobs/{nexus_job.id}",
            "estimated_cost_usd": ragnarok_result.cost_usd,
            "estimated_duration_seconds": ragnarok_result.duration_seconds,
            "trace_id": trace_id,
        }

    except RagnarokAPIError:
        # Re-raise - FastAPI will handle the proper status code (429, 400, etc.)
        raise
    except CircuitBreakerOpen as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAGNAROK service unavailable: {str(e)}"
        )


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


# =============================================================================
# TRINITY-ENRICHED ENDPOINTS
# =============================================================================


@router.post("/generate/enriched")
async def generate_enriched_commercial(
    request: Request,
    payload: dict,
    enable_trinity: bool = True
):
    """
    Generate commercial with Trinity market intelligence enrichment.

    This endpoint:
    1. Calls Trinity for market intelligence (trends, competitors, market data)
    2. Enriches the payload with Trinity insights
    3. Creates a Nexus tracking job
    4. Submits to RAGNAROK for data-driven commercial generation
    5. Returns BOTH job_id (for Nexus polling) and workflow_id (for reference)

    POLLING: Use GET /api/nexus/ragnarok/jobs/{job_id} - NEVER poll RAGNAROK directly.
    """
    trace_id = getattr(request.state, 'trace_id', f"nxs_{int(time.time())}_{__import__('uuid').uuid4().hex[:8]}")
    service = get_ragnarok_service()
    trinity = get_trinity_service()
    job_store = get_job_store()

    # Extract context for Trinity
    industry = payload.get("industry", "B2B Technology")
    product_description = payload.get("product_description", payload.get("company_description", ""))
    target_audience = payload.get("target_audience", "business professionals")
    company_name = payload.get("company_name", "Barrios A2I")

    enriched_payload = payload.copy()
    enrichment_metadata = {
        "trinity_enabled": enable_trinity,
        "trace_id": trace_id
    }

    if enable_trinity:
        try:
            intel = await trinity.gather_intelligence(
                industry=industry,
                product_description=product_description,
                target_audience=target_audience,
                company_name=company_name,
                trace_id=trace_id
            )

            # Merge Trinity context into payload
            trinity_context = intel.to_ragnarok_context()
            enriched_payload["market_context"] = trinity_context["market_intelligence"]
            enriched_payload["strategic_context"] = trinity_context.get("strategic_context", {})

            enrichment_metadata.update({
                "trinity_source": intel.source,
                "trinity_latency_ms": intel.latency_ms,
                "trinity_cost_usd": intel.cost_usd,
                "confidence": intel.confidence,
            })

            logger.info(f"Payload enriched with Trinity | trace={trace_id} | source={intel.source}")

        except Exception as e:
            logger.warning(f"Trinity enrichment failed: {e}")
            enrichment_metadata["trinity_error"] = str(e)

    enriched_payload["_enrichment"] = enrichment_metadata

    # Submit to RAGNAROK (this now raises RagnarokAPIError on failure)
    try:
        ragnarok_result = await service.submit_job(
            business_name=enriched_payload.get("company_name", "Barrios A2I"),
            user_id="nexus_user",
            formats=["1080p"],
            priority=5,
            trace_id=trace_id,
        )
    except RagnarokAPIError:
        # Re-raise - FastAPI will handle the proper status code (429, 400, etc.)
        raise
    except CircuitBreakerOpen as e:
        raise HTTPException(status_code=503, detail=f"RAGNAROK service unavailable: {e}")

    # Create a Nexus tracking job that wraps the RAGNAROK workflow
    nexus_job = await job_store.submit(
        job_type="ragnarok_enriched",
        payload={
            "brief": f"Enriched commercial for {company_name}",
            "company_name": company_name,
            "industry": industry,
            "workflow_id": ragnarok_result.workflow_id,
            "enrichment": enrichment_metadata,
        },
        metadata={
            "workflow_id": ragnarok_result.workflow_id,
            "trace_id": trace_id,
            "type": "enriched",
        }
    )

    # Store the mapping for unified polling
    service._results_cache[nexus_job.id] = ragnarok_result

    logger.info(
        f"Enriched job created | job_id={nexus_job.id} | "
        f"workflow_id={ragnarok_result.workflow_id} | trace={trace_id}"
    )

    return {
        "job_id": nexus_job.id,
        "workflow_id": ragnarok_result.workflow_id,
        "status": "queued",
        "tracking_url": f"/api/nexus/ragnarok/jobs/{nexus_job.id}",
        "enrichment": enrichment_metadata,
        "trace_id": trace_id,
    }


@router.get("/trinity/health")
async def trinity_health():
    """Check Trinity service health."""
    trinity = get_trinity_service()
    return await trinity.health_check()


@router.post("/trinity/preview")
async def preview_trinity_intelligence(request: Request, payload: dict):
    """Preview Trinity intelligence without generating commercial."""
    trace_id = getattr(request.state, 'trace_id', None)
    trinity = get_trinity_service()

    intel = await trinity.gather_intelligence(
        industry=payload.get("industry", "Technology"),
        product_description=payload.get("product_description", ""),
        target_audience=payload.get("target_audience", "professionals"),
        company_name=payload.get("company_name", "Barrios A2I"),
        trace_id=trace_id,
        use_cache=payload.get("use_cache", True)
    )

    return {
        "ragnarok_context": intel.to_ragnarok_context(),
        "executive_summary": intel.executive_summary,
        "key_insights": intel.key_insights,
        "metadata": {
            "source": intel.source,
            "latency_ms": intel.latency_ms,
            "cost_usd": intel.cost_usd,
            "confidence": intel.confidence,
        }
    }


@router.post("/trinity/cache/clear")
async def clear_trinity_cache():
    """Clear Trinity intelligence cache."""
    trinity = get_trinity_service()
    cleared = trinity.clear_cache()
    return {"status": "cleared", "entries_cleared": cleared}
