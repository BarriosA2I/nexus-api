"""
Nexus Assistant Unified - Intake Router
Handles commercial intake flow and triggers voiceover generation.
"""
import logging
import time
import uuid
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.voiceover_service import (
    generate_voiceover_from_intake,
    generate_single_voiceover,
    get_service_health,
    is_initialized,
)
from ..services.job_store import get_job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/nexus/intake", tags=["intake"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class SceneConfig(BaseModel):
    """Scene configuration for video architecture"""
    type: str = Field(default="hook", description="Scene type: hook, problem, solution, benefits, proof, cta")
    text: str = Field(..., description="Voiceover text for this scene")
    duration: float = Field(default=5.0, description="Target duration in seconds")
    emotion: Optional[str] = Field(default=None, description="Emotion: urgent, confident, warm, empathetic")
    pacing: str = Field(default="normal", description="Pacing: slow, normal, fast")


class VideoArchitecture(BaseModel):
    """Video architecture with scene plan"""
    scene_plan: list[SceneConfig] = Field(default_factory=list)
    total_duration: float = Field(default=30.0)
    style: Optional[str] = None


class IntakeConfig(BaseModel):
    """Complete intake configuration"""
    client_id: str = Field(..., description="Unique client identifier")
    commercial_id: Optional[str] = Field(default=None, description="Commercial identifier")
    business_name: str = Field(..., description="Business/company name")
    product_description: str = Field(default="", description="Product or service description")
    target_audience: str = Field(default="businesses", description="Target audience")
    main_problem: Optional[str] = Field(default=None, description="Main problem to solve")
    key_benefit: Optional[str] = Field(default=None, description="Key benefit")
    brand_voice: Optional[str] = Field(default=None, description="Brand voice guidelines")
    video_architecture: Optional[VideoArchitecture] = None
    enable_voiceover: bool = Field(default=True, description="Generate voiceovers")
    voiceover_quality: str = Field(default="premium", description="Quality tier: premium, standard, budget")


class IntakeCompleteRequest(BaseModel):
    """Request to complete intake and trigger downstream processes"""
    session_id: str = Field(..., description="Chat session ID")
    client_config: IntakeConfig


class SingleVoiceoverRequest(BaseModel):
    """Request for single scene voiceover"""
    text: str = Field(..., description="Voiceover text")
    scene_type: str = Field(default="hook")
    duration: float = Field(default=5.0)
    emotion: str = Field(default="confident")
    quality_tier: str = Field(default="premium")


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/complete")
async def complete_intake(request: IntakeCompleteRequest) -> Dict[str, Any]:
    """
    Complete intake and trigger downstream processes.

    This endpoint:
    1. Validates the intake configuration
    2. Generates a commercial_id if not provided
    3. Triggers voiceover generation (if enabled)
    4. Creates a tracking job for the full pipeline
    5. Returns the enriched client config

    The client config will include voiceover data when generation completes.
    """
    trace_id = f"intake_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    session_id = request.session_id
    config = request.client_config.model_dump()

    logger.info(f"[{trace_id}] Intake complete for {config['business_name']} | session={session_id}")

    # Generate commercial_id if not provided
    if not config.get("commercial_id"):
        config["commercial_id"] = f"comm_{config['client_id']}_{int(time.time())}"

    # Convert video architecture to dict if present
    if config.get("video_architecture"):
        if hasattr(config["video_architecture"], "model_dump"):
            config["video_architecture"] = config["video_architecture"].model_dump()
        # Convert scene_plan items
        if "scene_plan" in config["video_architecture"]:
            config["video_architecture"]["scene_plan"] = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in config["video_architecture"]["scene_plan"]
            ]

    # Add metadata
    config["_metadata"] = {
        "session_id": session_id,
        "trace_id": trace_id,
        "completed_at": time.time(),
    }

    # Trigger voiceover generation if enabled
    if config.get("enable_voiceover", True):
        try:
            quality_tier = config.get("voiceover_quality", "premium")
            config = await generate_voiceover_from_intake(config, quality_tier=quality_tier)
            logger.info(f"[{trace_id}] Voiceover generation complete")
        except Exception as e:
            logger.error(f"[{trace_id}] Voiceover generation failed: {e}")
            config["voiceover"] = {
                "enabled": False,
                "status": "failed",
                "error": str(e),
            }

    # Create tracking job
    job_store = get_job_store()
    job = await job_store.submit(
        job_type="intake_complete",
        payload=config,
        metadata={
            "trace_id": trace_id,
            "session_id": session_id,
            "client_id": config["client_id"],
        },
    )

    # Mark job as completed since voiceover is done
    if config.get("voiceover", {}).get("status") == "completed":
        await job_store.complete(job.id, config)

    return {
        "status": "completed",
        "job_id": job.id,
        "trace_id": trace_id,
        "client_config": config,
    }


@router.post("/voiceover/single")
async def generate_single(request: SingleVoiceoverRequest) -> Dict[str, Any]:
    """
    Generate a single scene voiceover.

    Useful for testing or regenerating individual scenes.
    """
    trace_id = f"vo_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    try:
        result = await generate_single_voiceover(
            text=request.text,
            scene_type=request.scene_type,
            duration=request.duration,
            emotion=request.emotion,
            quality_tier=request.quality_tier,
        )

        return {
            "status": "completed",
            "trace_id": trace_id,
            "result": result,
        }

    except Exception as e:
        logger.error(f"[{trace_id}] Single voiceover failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voiceover/health")
async def voiceover_health() -> Dict[str, Any]:
    """
    Check voiceover service health.

    Returns provider status and circuit breaker states.
    """
    return get_service_health()


@router.post("/validate")
async def validate_intake(config: IntakeConfig) -> Dict[str, Any]:
    """
    Validate intake configuration without processing.

    Checks:
    - Required fields present
    - Scene plan structure valid
    - Quality tier supported
    """
    issues = []

    # Check required fields
    if not config.business_name:
        issues.append("business_name is required")

    if not config.client_id:
        issues.append("client_id is required")

    # Validate scene plan if present
    if config.video_architecture and config.video_architecture.scene_plan:
        valid_types = {"hook", "problem", "solution", "benefits", "proof", "cta"}
        for i, scene in enumerate(config.video_architecture.scene_plan):
            if scene.type not in valid_types:
                issues.append(f"Scene {i+1}: invalid type '{scene.type}'")
            if not scene.text:
                issues.append(f"Scene {i+1}: text is required")
            if scene.duration <= 0:
                issues.append(f"Scene {i+1}: duration must be positive")

    # Validate quality tier
    valid_tiers = {"premium", "standard", "budget"}
    if config.voiceover_quality not in valid_tiers:
        issues.append(f"Invalid voiceover_quality: {config.voiceover_quality}")

    if issues:
        return {
            "valid": False,
            "issues": issues,
        }

    # Estimate voiceover cost
    scene_count = len(config.video_architecture.scene_plan) if config.video_architecture else 5
    cost_per_scene = {"premium": 0.15, "standard": 0.05, "budget": 0.0}
    estimated_cost = scene_count * cost_per_scene.get(config.voiceover_quality, 0.15)

    return {
        "valid": True,
        "issues": [],
        "estimates": {
            "scene_count": scene_count,
            "voiceover_cost_usd": estimated_cost,
            "processing_time_seconds": scene_count * 8,  # ~8s per scene
        },
    }


@router.get("/jobs/{job_id}")
async def get_intake_job(job_id: str) -> Dict[str, Any]:
    """
    Get intake job status and results.
    """
    job_store = get_job_store()
    job = job_store.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return {
        "job_id": job.id,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "client_id": job.metadata.get("client_id"),
        "trace_id": job.metadata.get("trace_id"),
        "result": job.result,
        "error": job.error,
    }
