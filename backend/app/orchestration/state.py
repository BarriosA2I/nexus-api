"""
NEXUS BRAIN v5.0 APEX - State Definitions
==========================================
Immutable state schema with Annotated reducers for LangGraph concurrency.

ConversationState flows through all pipeline nodes:
Classifier -> Router -> RAG -> Agent -> Publisher
"""

import operator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


class ComplexityLevel(str, Enum):
    """System 1 (fast) vs System 2 (slow) routing."""
    SYSTEM_1 = "system_1"  # Simple queries: greetings, FAQs, direct lookups
    SYSTEM_2 = "system_2"  # Complex queries: multi-step, reasoning, analysis


class ModelTier(str, Enum):
    """Available model tiers for Thompson Sampling."""
    HAIKU = "haiku"      # Fast, cheap - System 1
    SONNET = "sonnet"    # Balanced - default
    OPUS = "opus"        # Premium - complex reasoning


@dataclass
class RetrievedChunk:
    """A single retrieved knowledge chunk."""
    content: str
    score: float
    industry: str
    chunk_type: str
    source_title: str
    priority: str = "normal"
    quality_score: float = 0.5


@dataclass
class ClassifierResult:
    """Output from complexity classifier."""
    level: ComplexityLevel
    confidence: float
    signals: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class RouterResult:
    """Output from Thompson router."""
    selected_model: ModelTier
    model_id: str  # Full model ID for API call
    sample_value: float  # Thompson sampling draw
    reasoning: str = ""


@dataclass
class RAGResult:
    """Output from RAG retrieval."""
    chunks: List[RetrievedChunk] = field(default_factory=list)
    company_chunks: int = 0
    industry_chunks: int = 0
    total_chunks: int = 0
    avg_score: float = 0.0


@dataclass
class AgentResult:
    """Output from LLM agent."""
    response: str
    model_used: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    confidence: float = 0.0


class ConversationState(TypedDict, total=False):
    """
    Immutable conversation state for LangGraph pipeline.

    Uses Annotated reducers for safe concurrent updates.
    All node outputs are additive - nodes append to lists, don't overwrite.
    """

    # === INPUT (set once at start) ===
    session_id: str
    message: str
    timestamp: str

    # === CONVERSATION HISTORY ===
    # Reducer: operator.add - new messages append to list
    history: Annotated[List[Dict[str, str]], operator.add]
    history_length: int

    # === CLASSIFIER NODE OUTPUT ===
    complexity: Optional[ClassifierResult]
    detected_industry: str
    industry_confidence: float

    # === ROUTER NODE OUTPUT ===
    router_result: Optional[RouterResult]
    selected_model: str

    # === RAG NODE OUTPUT ===
    rag_result: Optional[RAGResult]
    context_chunks: Annotated[List[RetrievedChunk], operator.add]
    company_knowledge_found: bool

    # === AGENT NODE OUTPUT ===
    agent_result: Optional[AgentResult]
    response: str
    response_confidence: float

    # === PUBLISHER NODE OUTPUT ===
    published: bool
    support_code: str
    trace_id: str

    # === METRICS & DEBUGGING ===
    pipeline_start: float
    pipeline_end: float
    node_timings: Dict[str, float]
    errors: Annotated[List[str], operator.add]

    # === THOMPSON SAMPLING FEEDBACK ===
    # Set after response quality is evaluated
    thompson_success: Optional[bool]
    thompson_reward: Optional[float]


def create_initial_state(
    session_id: str,
    message: str,
    history: Optional[List[Dict[str, str]]] = None
) -> ConversationState:
    """
    Create initial state for a new conversation turn.

    Args:
        session_id: Unique session identifier
        message: User's input message
        history: Previous conversation history (optional)

    Returns:
        ConversationState ready for pipeline execution
    """
    import time
    import uuid

    return ConversationState(
        # Input
        session_id=session_id,
        message=message,
        timestamp=datetime.utcnow().isoformat(),

        # History
        history=history or [],
        history_length=len(history) if history else 0,

        # Classifier (to be filled)
        complexity=None,
        detected_industry="general",
        industry_confidence=0.0,

        # Router (to be filled)
        router_result=None,
        selected_model="claude-sonnet-4-20250514",

        # RAG (to be filled)
        rag_result=None,
        context_chunks=[],
        company_knowledge_found=False,

        # Agent (to be filled)
        agent_result=None,
        response="",
        response_confidence=0.0,

        # Publisher (to be filled)
        published=False,
        support_code=f"NX-{uuid.uuid4().hex[:8].upper()}",
        trace_id=f"trace-{uuid.uuid4().hex[:12]}",

        # Metrics
        pipeline_start=time.time(),
        pipeline_end=0.0,
        node_timings={},
        errors=[],

        # Thompson feedback
        thompson_success=None,
        thompson_reward=None,
    )


def state_to_dict(state: ConversationState) -> Dict[str, Any]:
    """Convert state to JSON-serializable dict for logging/persistence."""
    result = dict(state)

    # Convert dataclasses to dicts
    if result.get("complexity"):
        result["complexity"] = {
            "level": result["complexity"].level.value,
            "confidence": result["complexity"].confidence,
            "signals": result["complexity"].signals,
            "reasoning": result["complexity"].reasoning,
        }

    if result.get("router_result"):
        result["router_result"] = {
            "selected_model": result["router_result"].selected_model.value,
            "model_id": result["router_result"].model_id,
            "sample_value": result["router_result"].sample_value,
            "reasoning": result["router_result"].reasoning,
        }

    if result.get("rag_result"):
        result["rag_result"] = {
            "company_chunks": result["rag_result"].company_chunks,
            "industry_chunks": result["rag_result"].industry_chunks,
            "total_chunks": result["rag_result"].total_chunks,
            "avg_score": result["rag_result"].avg_score,
        }

    if result.get("agent_result"):
        result["agent_result"] = {
            "model_used": result["agent_result"].model_used,
            "tokens_used": result["agent_result"].tokens_used,
            "latency_ms": result["agent_result"].latency_ms,
            "confidence": result["agent_result"].confidence,
        }

    # Convert chunks to dicts
    if result.get("context_chunks"):
        result["context_chunks"] = [
            {
                "content": c.content[:100] + "..." if len(c.content) > 100 else c.content,
                "score": c.score,
                "industry": c.industry,
                "chunk_type": c.chunk_type,
            }
            for c in result["context_chunks"]
        ]

    return result
