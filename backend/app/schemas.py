"""
Nexus Assistant Unified - Pydantic Schemas
Request/Response models for all API endpoints
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CircuitState(str, Enum):
    """Circuit breaker state"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class ChatMode(str, Enum):
    """Chat processing mode"""
    RAG = "rag"
    RAGNAROK = "ragnarok"
    HYBRID = "hybrid"


class SSEEventType(str, Enum):
    """Server-Sent Event types"""
    STATUS = "status"
    CHUNK = "chunk"
    COMPLETE = "complete"
    ERROR = "error"


# ============================================================================
# Chat Schemas
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request payload"""
    session_id: Optional[str] = Field(default=None, description="Session identifier for conversation continuity")
    message: str = Field(..., min_length=1, max_length=4096, description="User message")
    stream: bool = Field(default=True, description="Enable SSE streaming")
    mode: Optional[ChatMode] = Field(default=None, description="Force specific processing mode")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class Source(BaseModel):
    """Knowledge source reference"""
    title: str = Field(..., description="Source document title")
    chunk_id: str = Field(..., description="Chunk identifier")
    relevance: float = Field(..., ge=0, le=1, description="Relevance score 0-1")
    excerpt: str = Field(..., description="Relevant text excerpt")
    file: Optional[str] = Field(default=None, description="Source filename")


class SSEStatusEvent(BaseModel):
    """SSE status update event"""
    type: Literal["status"] = "status"
    step: str = Field(..., description="Processing step identifier")
    message: str = Field(..., description="Human-readable status message")
    trace_id: str = Field(..., description="Request trace identifier")


class SSEChunkEvent(BaseModel):
    """SSE text chunk event"""
    type: Literal["chunk"] = "chunk"
    content: str = Field(..., description="Text chunk content")


class SSECompleteEvent(BaseModel):
    """SSE completion event"""
    type: Literal["complete"] = "complete"
    trace_id: str = Field(..., description="Request trace identifier")
    confidence: float = Field(..., ge=0, le=1, description="Response confidence score")
    sources: List[Source] = Field(default_factory=list, description="Knowledge sources used")
    mode: ChatMode = Field(..., description="Processing mode used")
    tokens_used: Optional[int] = Field(default=None, description="Tokens consumed")
    latency_ms: Optional[int] = Field(default=None, description="Total latency in milliseconds")


class SSEErrorEvent(BaseModel):
    """SSE error event"""
    type: Literal["error"] = "error"
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    recoverable: bool = Field(default=True, description="Whether error is recoverable")
    trace_id: Optional[str] = Field(default=None, description="Request trace identifier")


# ============================================================================
# Ragnarok Schemas
# ============================================================================

class RagnarokGenerateRequest(BaseModel):
    """Ragnarok commercial generation request"""
    brief: str = Field(..., min_length=10, max_length=2000, description="Creative brief")
    industry: str = Field(default="technology", description="Target industry")
    duration_seconds: int = Field(default=30, ge=15, le=120, description="Video duration")
    platform: str = Field(default="youtube_1080p", description="Target platform/format")
    style: Optional[str] = Field(default=None, description="Visual style preference")
    voice_style: Optional[str] = Field(default=None, description="Voiceover style")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class RagnarokJob(BaseModel):
    """Ragnarok job status"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    progress: float = Field(default=0, ge=0, le=100, description="Progress percentage")
    brief: str = Field(..., description="Original brief")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Job result when completed")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class RagnarokJobList(BaseModel):
    """List of Ragnarok jobs"""
    jobs: List[RagnarokJob] = Field(default_factory=list)
    total: int = Field(default=0)


# ============================================================================
# Health Schemas
# ============================================================================

class ComponentHealth(BaseModel):
    """Individual component health status"""
    status: str = Field(..., description="Component status")
    latency_ms: Optional[int] = Field(default=None, description="Component latency")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class QdrantRAGHealth(BaseModel):
    """Qdrant RAG client health"""
    enabled: bool = Field(default=False, description="Whether Qdrant RAG is enabled")
    connected: bool = Field(default=False, description="Whether connected to Qdrant")
    collection: Optional[str] = Field(default=None, description="Qdrant collection name")
    embedder_loaded: bool = Field(default=False, description="Whether embedding model loaded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class RAGHealth(BaseModel):
    """RAG service health"""
    loaded: bool = Field(..., description="Whether RAG is loaded")
    chunks: int = Field(default=0, description="Number of loaded chunks")
    knowledge_files: List[str] = Field(default_factory=list, description="Loaded knowledge files")
    load_time_ms: Optional[int] = Field(default=None, description="Load time in milliseconds")
    qdrant: Optional[QdrantRAGHealth] = Field(default=None, description="Qdrant RAG status")


class CircuitBreakerHealth(BaseModel):
    """Circuit breaker status"""
    name: str = Field(..., description="Circuit breaker name")
    state: CircuitState = Field(..., description="Current state")
    failure_count: int = Field(default=0, description="Current failure count")
    last_failure: Optional[datetime] = Field(default=None, description="Last failure timestamp")


class HealthResponse(BaseModel):
    """Complete health check response"""
    status: str = Field(..., description="Overall system status: ONLINE, DEGRADED, OFFLINE")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    rag: RAGHealth = Field(..., description="RAG service health")
    job_queue: ComponentHealth = Field(..., description="Job queue health")
    ragnarok: ComponentHealth = Field(..., description="Ragnarok bridge health")
    circuit_breakers: List[CircuitBreakerHealth] = Field(default_factory=list, description="Circuit breaker states")


# ============================================================================
# Internal Schemas
# ============================================================================

class RAGResult(BaseModel):
    """Internal RAG retrieval result"""
    query: str
    chunks: List[Dict[str, Any]]
    scores: List[float]
    search_time_ms: int


class GeneratedResponse(BaseModel):
    """Internal generated response"""
    content: str
    sources: List[Source]
    confidence: float
    mode: ChatMode
    trace_id: str
    tokens_used: Optional[int] = None
