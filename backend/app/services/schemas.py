"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NEXUS SHARED SCHEMAS v1.0                                 ║
║              Single Source of Truth for All Event Contracts                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Import this in: nexus_brain, trigger_system, oracle_service                 ║
║  Barrios A2I Cognitive Systems Division | December 2025                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# ROUTING KEYS (Canonical - DO NOT CHANGE WITHOUT UPDATING ALL CONSUMERS)
# =============================================================================

class RoutingKeys:
    """RabbitMQ routing keys - must match trigger_system.py consumers."""
    
    CONVERSATION_COMPLETED = "nexus.conversation.completed"
    CONFIDENCE_LOW = "nexus.confidence.low"
    INDUSTRY_UNKNOWN = "nexus.industry.unknown"
    RESEARCH_COMPLETED = "nexus.research.completed"
    
    # Exchange name (use default "" for simple routing, or named for topic routing)
    EXCHANGE = "nexus.events"


# =============================================================================
# CONVERSATION EVENTS (Nexus → Trigger)
# =============================================================================

class ConversationEvent(BaseModel):
    """
    Event published by Nexus Brain after each conversation turn.
    Consumed by Trigger System to detect gaps and queue research.
    
    CRITICAL: This schema MUST match between publisher and consumer.
    """
    
    conversation_id: str = Field(
        ..., 
        description="Persistent session ID (from X-Session-ID header)"
    )
    user_message: str = Field(
        ..., 
        description="The user's input message"
    )
    nexus_response: str = Field(
        ..., 
        description="Nexus's full response text"
    )
    detected_industry: Optional[str] = Field(
        None, 
        description="Industry detected from message (e.g., 'dental_practices', 'law_firms')"
    )
    confidence_score: float = Field(
        1.0, 
        ge=0.0, 
        le=1.0,
        description="Response confidence (0.0-1.0). Below 0.7 triggers research."
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp (UTC)"
    )
    
    # Optional metadata
    rag_chunks_used: int = Field(
        0,
        description="Number of RAG chunks retrieved for this response"
    )
    response_latency_ms: float = Field(
        0.0,
        description="Response generation latency in milliseconds"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


# =============================================================================
# RESEARCH EVENTS (Trigger → Oracle)
# =============================================================================

class ResearchPriority(str, Enum):
    """Research task priority levels."""
    CRITICAL = "critical"   # Immediate execution
    HIGH = "high"           # Within 1 hour
    MEDIUM = "medium"       # Within 4 hours
    LOW = "low"             # Next scheduled batch


class ResearchTriggerType(str, Enum):
    """What triggered the research request."""
    SCHEDULED = "scheduled"
    GAP_DETECTED = "gap_detected"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN_INDUSTRY = "unknown_industry"
    MANUAL = "manual"
    NEWS_UPDATE = "news_update"


class ResearchRequest(BaseModel):
    """
    Request to queue research task.
    Sent from Trigger System to Oracle Service.
    """
    
    industry: str = Field(
        ...,
        description="Industry to research (e.g., 'dental_practices')"
    )
    trigger_type: ResearchTriggerType = Field(
        ResearchTriggerType.MANUAL,
        description="What triggered this research"
    )
    priority: ResearchPriority = Field(
        ResearchPriority.MEDIUM,
        description="Task priority"
    )
    topic: Optional[str] = Field(
        None,
        description="Specific topic within industry (for gap filling)"
    )
    source_conversation_id: Optional[str] = Field(
        None,
        description="Conversation that triggered this research"
    )
    
    # Dedupe key generation
    def get_dedupe_key(self) -> str:
        """Generate dedupe key: industry + topic + date."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        topic_str = self.topic or "general"
        return f"{self.industry}:{topic_str}:{date_str}"


class ResearchResponse(BaseModel):
    """
    Response from Oracle after queueing research.
    """
    
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="queued | running | completed | failed")
    dedupe_hit: bool = Field(
        False, 
        description="True if this was a duplicate (not queued)"
    )
    message: str = Field("", description="Human-readable status message")


class ResearchResult(BaseModel):
    """
    Result of completed research task.
    Contains extracted knowledge ready for Qdrant upsert.
    """
    
    task_id: str
    industry: str
    status: str  # completed | failed
    
    # Extracted knowledge
    chunks_created: int = 0
    pain_points_found: int = 0
    opportunities_found: int = 0
    objections_found: int = 0
    
    # Quality metrics
    quality_score: float = 0.0
    duration_ms: int = 0
    
    # Errors if failed
    error_message: Optional[str] = None


# =============================================================================
# KNOWLEDGE SCHEMAS (Oracle → Qdrant)
# =============================================================================

class KnowledgeChunkType(str, Enum):
    """Types of knowledge chunks stored in Qdrant."""
    PAIN_POINT = "pain_point"
    AUTOMATION_OPPORTUNITY = "automation_opportunity"
    DECISION_MAKER = "decision_maker"
    OBJECTION_HANDLER = "objection_handler"
    ROI_DATA = "roi_data"
    COMPETITOR = "competitor"
    TERMINOLOGY = "terminology"
    CONVERSATION_STARTER = "conversation_starter"
    COMPANY_CORE = "company_core"  # Barrios A2I company info


class KnowledgeChunk(BaseModel):
    """
    Knowledge chunk for Qdrant storage.
    """
    
    id: str = Field(..., description="Unique chunk ID")
    industry: str = Field(..., description="Industry this applies to")
    chunk_type: KnowledgeChunkType
    content: str = Field(..., description="The actual knowledge text")
    
    # Metadata
    source: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    quality_score: float = 0.0
    
    # For ROI data
    metric_value: Optional[str] = None
    metric_source: Optional[str] = None


# =============================================================================
# HEALTH CHECK SCHEMAS
# =============================================================================

class ServiceHealth(BaseModel):
    """Health check response for any service."""
    
    status: str = Field(..., description="healthy | degraded | unhealthy")
    service: str = Field(..., description="Service name")
    version: str = Field("1.0.0", description="Service version")
    
    # Component health
    components: Dict[str, bool] = Field(
        default_factory=dict,
        description="Health of sub-components (e.g., qdrant: true)"
    )
    
    # Metrics
    uptime_seconds: float = 0.0
    requests_total: int = 0
    errors_total: int = 0


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_conversation_event(data: dict) -> ConversationEvent:
    """Validate and parse conversation event from RabbitMQ message."""
    return ConversationEvent.model_validate(data)


def validate_research_request(data: dict) -> ResearchRequest:
    """Validate and parse research request."""
    return ResearchRequest.model_validate(data)
