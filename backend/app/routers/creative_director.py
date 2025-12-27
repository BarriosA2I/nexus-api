"""
================================================================================
NEXUS - Creative Director API Router
================================================================================
FastAPI router for Creative Director endpoints.

Endpoints:
- POST /session          - Create new CD session
- POST /intake           - Process intake message
- POST /pipeline/start   - Start 6-agent pipeline
- GET  /pipeline/status  - Get pipeline status
- POST /concept/select   - Select concept
- GET  /video/{job_id}   - Get video status
- DELETE /session        - Close session

Author: Barrios A2I | Version: 6.0.0
================================================================================
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.creative_director.intent_detector import CreativeIntentDetector, CreativeIntent
from app.creative_director.session_manager import (
    CreativeSessionManager,
    CreativeDirectorSession,
    BriefData,
    WorkflowPhase,
)
from app.creative_director.pipeline_bridge import (
    CreativeDirectorBridge,
    PipelineEvent,
    PipelineEventData,
)

logger = logging.getLogger("nexus.api.creative_director")
router = APIRouter()

# Global instances (initialized on startup)
session_manager: CreativeSessionManager = None
pipeline_bridge: CreativeDirectorBridge = None
intent_detector: CreativeIntentDetector = None


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

async def initialize_creative_director(app_state):
    """Initialize Creative Director components"""
    global session_manager, pipeline_bridge, intent_detector
    
    session_manager = CreativeSessionManager()
    pipeline_bridge = CreativeDirectorBridge(use_mock=True)  # TODO: Wire real clients
    intent_detector = CreativeIntentDetector()
    
    await pipeline_bridge.initialize()
    logger.info("✓ Creative Director initialized")


async def shutdown_creative_director():
    """Shutdown Creative Director components"""
    logger.info("✓ Creative Director shutdown")


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class CreateSessionRequest(BaseModel):
    """Create session request"""
    user_id: str
    nexus_session_id: Optional[str] = None


class CreateSessionResponse(BaseModel):
    """Create session response"""
    session_id: str
    greeting: str
    phase: str


class IntakeMessageRequest(BaseModel):
    """Intake message request"""
    session_id: str
    message: str


class IntakeMessageResponse(BaseModel):
    """Intake message response"""
    response: str
    phase: str
    is_complete: bool
    brief: Optional[Dict[str, Any]] = None
    next_action: Optional[str] = None


class BriefDataRequest(BaseModel):
    """Direct brief submission"""
    session_id: str
    business_name: str
    industry: str
    target_audience: str
    target_platform: str = "tiktok"
    video_duration: int = 30
    key_message: str
    unique_selling_points: List[str] = []
    brand_tone: str = "professional"
    budget_tier: str = "standard"
    competitors: List[str] = []


class StartPipelineRequest(BaseModel):
    """Start pipeline request"""
    session_id: str
    auto_select_concept: bool = True


class PipelineStatusResponse(BaseModel):
    """Pipeline status response"""
    session_id: str
    workflow_phase: str
    brief_complete: bool
    has_research: bool
    num_concepts: int
    has_script: bool
    production_job: Optional[Dict[str, Any]] = None
    video_url: Optional[str] = None


class ConceptSelectRequest(BaseModel):
    """Concept selection request"""
    session_id: str
    concept_id: str


class IntentDetectRequest(BaseModel):
    """Intent detection request"""
    message: str
    session_context: Optional[Dict[str, Any]] = None


class IntentDetectResponse(BaseModel):
    """Intent detection response"""
    intent: str
    confidence: float
    should_handoff: bool
    suggested_response: Optional[str] = None


# =============================================================================
# SESSION ENDPOINTS
# =============================================================================

@router.post("/session", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """
    Create a new Creative Director session.
    
    Returns session ID and greeting to start intake conversation.
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    nexus_id = request.nexus_session_id or f"nexus-{datetime.utcnow().timestamp()}"
    session = session_manager.create_session(nexus_id, request.user_id)
    
    # Get initial greeting
    result = await session_manager.process_intake_message(session.session_id, "")
    
    return CreateSessionResponse(
        session_id=session.session_id,
        greeting=result["response"],
        phase=result["phase"],
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "workflow_phase": session.workflow_phase.value,
        "intake_phase": session.intake_phase.value,
        "brief_complete": session.brief.is_complete if session.brief else False,
        "video_url": session.video_url,
        "created_at": session.created_at.isoformat(),
    }


@router.delete("/session/{session_id}")
async def close_session(session_id: str):
    """Close and cleanup session"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    session_manager.close_session(session_id)
    return {"status": "closed", "session_id": session_id}


# =============================================================================
# INTAKE ENDPOINTS
# =============================================================================

@router.post("/intake", response_model=IntakeMessageResponse)
async def process_intake_message(request: IntakeMessageRequest):
    """
    Process an intake conversation message.
    
    Guides user through brief collection and returns next prompt.
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    result = await session_manager.process_intake_message(
        request.session_id,
        request.message,
    )
    
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result.get("response"))
    
    return IntakeMessageResponse(
        response=result["response"],
        phase=result["phase"],
        is_complete=result.get("is_complete", False),
        brief=result.get("brief"),
        next_action=result.get("next_action"),
    )


@router.post("/brief")
async def submit_brief(request: BriefDataRequest):
    """
    Submit a complete brief directly (skip intake conversation).
    
    Useful for API integrations or returning users.
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update brief
    session.brief = BriefData(
        session_id=session.session_id,
        business_name=request.business_name,
        industry=request.industry,
        target_audience=request.target_audience,
        target_platform=request.target_platform,
        video_duration=request.video_duration,
        key_message=request.key_message,
        unique_selling_points=request.unique_selling_points,
        brand_tone=request.brand_tone,
        budget_tier=request.budget_tier,
        competitors=request.competitors,
        is_complete=True,
    )
    
    session.workflow_phase = WorkflowPhase.RESEARCH
    
    return {
        "status": "brief_submitted",
        "session_id": session.session_id,
        "brief": session.brief.to_dict(),
        "next_action": "start_pipeline",
    }


# =============================================================================
# PIPELINE ENDPOINTS
# =============================================================================

@router.post("/pipeline/start")
async def start_pipeline(
    request: StartPipelineRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start the 6-agent Creative Director pipeline.
    
    Runs asynchronously and emits events for progress tracking.
    """
    if not session_manager or not pipeline_bridge:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.brief or not session.brief.is_complete:
        raise HTTPException(status_code=400, detail="Brief not complete")
    
    # Run pipeline in background
    async def run_pipeline_async():
        events = []
        async for event in pipeline_bridge.run_pipeline(
            session,
            concept_selector=None if request.auto_select_concept else lambda c: c[0]["concept_id"],
        ):
            events.append(event)
            logger.info(f"Pipeline event: {event.event.value}")
        return events
    
    # For now, run synchronously and return result
    # In production, this would use background tasks + webhooks
    events = []
    async for event in pipeline_bridge.run_pipeline(session):
        events.append({
            "event": event.event.value,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
        })
    
    return {
        "status": "complete" if session.workflow_phase == WorkflowPhase.COMPLETE else "error",
        "session_id": session.session_id,
        "video_url": session.video_url,
        "events": events,
    }


@router.get("/pipeline/status/{session_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(session_id: str):
    """Get current pipeline status"""
    if not session_manager or not pipeline_bridge:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    status = await pipeline_bridge.get_pipeline_status(session)
    return PipelineStatusResponse(**status)


@router.get("/pipeline/stream/{session_id}")
async def stream_pipeline_events(session_id: str):
    """
    Stream pipeline events via Server-Sent Events.
    
    Connect to this endpoint to receive real-time updates.
    """
    if not session_manager or not pipeline_bridge:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    async def event_generator():
        import json
        async for event in pipeline_bridge.run_pipeline(session):
            yield f"data: {json.dumps({'event': event.event.value, 'data': event.data})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# =============================================================================
# CONCEPT ENDPOINTS
# =============================================================================

@router.get("/concepts/{session_id}")
async def get_concepts(session_id: str):
    """Get generated concepts for a session"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "concepts": session.concepts,
        "selected_concept_id": session.selected_concept_id,
    }


@router.post("/concepts/select")
async def select_concept(request: ConceptSelectRequest):
    """Select a concept for script generation"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate concept exists
    concept = next((c for c in session.concepts if c["concept_id"] == request.concept_id), None)
    if not concept:
        raise HTTPException(status_code=400, detail="Concept not found")
    
    session.selected_concept_id = request.concept_id
    
    return {
        "status": "concept_selected",
        "session_id": session.session_id,
        "concept": concept,
    }


# =============================================================================
# VIDEO ENDPOINTS
# =============================================================================

@router.get("/video/{session_id}")
async def get_video_status(session_id: str):
    """Get video production status"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.production_job:
        return {
            "session_id": session.session_id,
            "status": "not_started",
            "video_url": None,
        }
    
    return {
        "session_id": session.session_id,
        "job_id": session.production_job.job_id,
        "status": session.production_job.status,
        "progress": session.production_job.progress,
        "video_url": session.production_job.video_url,
        "preview_url": session.production_job.preview_url,
    }


# =============================================================================
# INTENT DETECTION ENDPOINT
# =============================================================================

@router.post("/intent/detect", response_model=IntentDetectResponse)
async def detect_intent(request: IntentDetectRequest):
    """
    Detect if a message should be routed to Creative Director.
    
    Use this from the main NEXUS chat to determine routing.
    """
    if not intent_detector:
        raise HTTPException(status_code=503, detail="Creative Director not initialized")
    
    result = intent_detector.detect(
        request.message,
        request.session_context,
    )
    
    return IntentDetectResponse(
        intent=result.intent.value,
        confidence=result.confidence,
        should_handoff=result.should_handoff,
        suggested_response=result.suggested_response,
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get("/health")
async def health_check():
    """Creative Director health check"""
    return {
        "status": "healthy" if session_manager else "not_initialized",
        "components": {
            "session_manager": session_manager is not None,
            "pipeline_bridge": pipeline_bridge is not None,
            "intent_detector": intent_detector is not None,
        },
        "active_sessions": len(session_manager.get_active_sessions()) if session_manager else 0,
    }


