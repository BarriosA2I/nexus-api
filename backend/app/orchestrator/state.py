"""
NEXUS SuperGraph - Unified State Schema
All subgraphs (RAG, Creative Director, Trinity) share this state
"""
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4
import operator


# ============================================================================
# UNIFIED STATE (Shared across ALL subgraphs)
# ============================================================================
class NexusState(TypedDict, total=False):
    """
    Single state object that flows through the entire SuperGraph.
    Subgraphs read/write only the fields they need.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # SESSION METADATA
    # ─────────────────────────────────────────────────────────────────────────
    session_id: str
    user_id: str
    created_at: str
    updated_at: str

    # ─────────────────────────────────────────────────────────────────────────
    # CONVERSATION HISTORY
    # ─────────────────────────────────────────────────────────────────────────
    messages: Annotated[List[Dict[str, Any]], operator.add]  # Append-only

    # ─────────────────────────────────────────────────────────────────────────
    # ROUTING & INTENT
    # ─────────────────────────────────────────────────────────────────────────
    current_intent: Literal["general_chat", "video_creation", "market_research", "intake", "escalate"]
    previous_intent: Optional[str]
    intent_confidence: float
    intent_history: List[Dict[str, Any]]

    # ─────────────────────────────────────────────────────────────────────────
    # RAG SUBGRAPH STATE
    # ─────────────────────────────────────────────────────────────────────────
    query: str
    retrieved_docs: List[Dict[str, Any]]
    compressed_context: str
    rag_confidence: float
    industry: Optional[str]
    industry_confidence: float

    # ─────────────────────────────────────────────────────────────────────────
    # CREATIVE DIRECTOR SUBGRAPH STATE
    # ─────────────────────────────────────────────────────────────────────────
    cd_phase: Optional[Literal["intake", "brief", "script", "review", "render", "deliver", "complete"]]
    cd_intake_complete: bool
    cd_brief: Optional[Dict[str, Any]]
    cd_script: Optional[Dict[str, Any]]
    cd_script_approved: bool
    cd_render_job_id: Optional[str]
    cd_video_url: Optional[str]
    cd_delivery_status: Optional[str]
    cd_intake_questions_asked: List[str]
    cd_intake_answers: Dict[str, str]
    cd_intake_missing_info: List[str]

    # ─────────────────────────────────────────────────────────────────────────
    # TRINITY SUBGRAPH STATE
    # ─────────────────────────────────────────────────────────────────────────
    trinity_query: Optional[str]
    trinity_competitor_data: Optional[Dict[str, Any]]
    trinity_sentiment_data: Optional[Dict[str, Any]]
    trinity_trends_data: Optional[Dict[str, Any]]
    trinity_synthesis: Optional[str]

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVABILITY & COST TRACKING
    # ─────────────────────────────────────────────────────────────────────────
    total_cost_usd: float
    total_latency_ms: float
    token_usage: Dict[str, int]
    model_calls: Annotated[List[Dict[str, Any]], operator.add]  # Append-only

    # ─────────────────────────────────────────────────────────────────────────
    # ERROR HANDLING & RECOVERY
    # ─────────────────────────────────────────────────────────────────────────
    errors: Annotated[List[Dict[str, Any]], operator.add]  # Append-only
    last_successful_node: str
    checkpoint_id: Optional[str]

    # ─────────────────────────────────────────────────────────────────────────
    # CIRCUIT BREAKER STATES
    # ─────────────────────────────────────────────────────────────────────────
    circuit_breaker_states: Dict[str, str]


def create_initial_state(session_id: str, user_id: str, message: str) -> NexusState:
    """Factory function to create initial state with defaults"""
    now = datetime.utcnow().isoformat()

    return NexusState(
        # Session
        session_id=session_id,
        user_id=user_id,
        created_at=now,
        updated_at=now,

        # Conversation
        messages=[{"role": "user", "content": message, "timestamp": now}],

        # Routing
        current_intent="general_chat",
        previous_intent=None,
        intent_confidence=0.0,
        intent_history=[],

        # RAG
        query=message,
        retrieved_docs=[],
        compressed_context="",
        rag_confidence=0.0,
        industry=None,
        industry_confidence=0.0,

        # Creative Director
        cd_phase=None,
        cd_intake_complete=False,
        cd_brief=None,
        cd_script=None,
        cd_script_approved=False,
        cd_render_job_id=None,
        cd_video_url=None,
        cd_delivery_status=None,
        cd_intake_questions_asked=[],
        cd_intake_answers={},
        cd_intake_missing_info=[],

        # Trinity
        trinity_query=None,
        trinity_competitor_data=None,
        trinity_sentiment_data=None,
        trinity_trends_data=None,
        trinity_synthesis=None,

        # Observability
        total_cost_usd=0.0,
        total_latency_ms=0.0,
        token_usage={"input": 0, "output": 0},
        model_calls=[],

        # Error handling
        errors=[],
        last_successful_node="start",
        checkpoint_id=None,

        # Circuit breakers
        circuit_breaker_states={},
    )
