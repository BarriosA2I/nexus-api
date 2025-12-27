"""
Unified Chat Endpoint - Single API for all NEXUS interactions
Replaces: /api/nexus/chat + /api/legendary/* + /api/creative-director/*
"""
import logging
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from ..orchestrator.supergraph import get_supergraph
from ..orchestrator.state import create_initial_state

logger = logging.getLogger("nexus.unified_chat")
router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: Optional[str] = "anonymous"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    intent_confidence: float
    phase: Optional[str] = None  # For Creative Director flow
    video_url: Optional[str] = None  # If video is ready
    cost_usd: float
    latency_ms: float


class SessionState(BaseModel):
    session_id: str
    current_intent: Optional[str] = None
    cd_phase: Optional[str] = None
    message_count: int = 0
    total_cost_usd: float = 0.0


@router.post("/chat", response_model=ChatResponse)
async def unified_chat(request: ChatRequest):
    """
    Single unified endpoint for ALL NEXUS interactions.

    Automatically routes to:
    - RAG subgraph (general chat)
    - Creative Director subgraph (video creation)
    - Trinity subgraph (market research)
    - Escalation (human handoff)

    State persists across intent switches via PostgreSQL checkpointing.
    """
    start_time = time.perf_counter()

    try:
        supergraph = await get_supergraph()

        if supergraph is None:
            raise HTTPException(
                status_code=503,
                detail="SuperGraph not available. Install langgraph for full functionality."
            )

        # Config for checkpointer - thread_id enables state persistence
        config = {
            "configurable": {
                "thread_id": request.session_id
            }
        }

        # Create or update state
        state = create_initial_state(
            session_id=request.session_id,
            user_id=request.user_id,
            message=request.message
        )

        # Invoke the supergraph
        result = await supergraph.ainvoke(state, config)

        # Extract response
        messages = result.get("messages", [])
        last_assistant_message = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"),
            "I'm sorry, I couldn't generate a response."
        )

        total_latency = (time.perf_counter() - start_time) * 1000

        return ChatResponse(
            response=last_assistant_message,
            session_id=request.session_id,
            intent=result.get("current_intent", "general_chat"),
            intent_confidence=result.get("intent_confidence", 0.0),
            phase=result.get("cd_phase"),
            video_url=result.get("cd_video_url"),
            cost_usd=result.get("total_cost_usd", 0.0),
            latency_ms=total_latency,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}", response_model=SessionState)
async def get_session_state(session_id: str):
    """Get current state for a session (for debugging/admin)"""

    try:
        supergraph = await get_supergraph()

        if supergraph is None:
            raise HTTPException(
                status_code=503,
                detail="SuperGraph not available"
            )

        config = {"configurable": {"thread_id": session_id}}

        # Try to get state
        try:
            state = await supergraph.aget_state(config)
        except AttributeError:
            # Fallback orchestrator doesn't have aget_state
            state = None

        if not state:
            raise HTTPException(status_code=404, detail="Session not found")

        values = state.values if hasattr(state, 'values') else state

        return SessionState(
            session_id=session_id,
            current_intent=values.get("current_intent"),
            cd_phase=values.get("cd_phase"),
            message_count=len(values.get("messages", [])),
            total_cost_usd=values.get("total_cost_usd", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session state (start fresh)"""

    # TODO: Implement checkpoint deletion when using PostgreSQL checkpointer
    return {"status": "cleared", "session_id": session_id}


@router.get("/health")
async def health_check():
    """Unified chat health check"""
    try:
        supergraph = await get_supergraph()
        return {
            "status": "healthy",
            "supergraph_available": supergraph is not None,
            "langgraph_enabled": supergraph is not None and not hasattr(supergraph, 'sessions'),
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
        }
