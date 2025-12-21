"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      NEXUS RESEARCH ORACLE v1.0                              ║
║              "The Self-Improving Intelligence Engine"                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Autonomous RAG Agent | Neural RAG Brain Architecture | Zero Human Input     ║
║  Barrios A2I Cognitive Systems Division | December 2025                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Core Architecture:
- LangGraph State Machine for pipeline orchestration
- Perplexity AI for real-time web research
- Claude Haiku 4.5 for cost-efficient processing
- Qdrant vector storage with 4-tier memory hierarchy
- RabbitMQ event-driven triggers
- Full circuit breaker + observability stack
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    TypedDict,
    Union,
)

import httpx
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field, validator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# =============================================================================
# LOGGING & TRACING SETUP
# =============================================================================

logger = logging.getLogger("nexus_oracle")
tracer = trace.get_tracer("nexus_research_oracle", "1.0.0")


class StructuredLogger:
    """Cyberpunk-styled structured JSON logging."""

    @staticmethod
    def log(
        level: str,
        event: str,
        component: str = "research_oracle",
        **kwargs,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "trace_id": kwargs.pop("trace_id", str(uuid.uuid4())),
            "component": component,
            "event": event,
            **kwargs,
        }
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(f"⚡ [{event}] " + json.dumps(entry))


slog = StructuredLogger()


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================


class ResearchTrigger(str, Enum):
    """Types of research triggers."""

    SCHEDULED = "scheduled"
    GAP_DETECTED = "gap_detected"
    NEWS_UPDATE = "news_update"
    MANUAL = "manual"
    LOW_CONFIDENCE = "low_confidence"
    REPEATED_QUESTION = "repeated_question"


class ResearchPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"  # Immediate execution
    HIGH = "high"  # Within 1 hour
    MEDIUM = "medium"  # Within 4 hours
    LOW = "low"  # Next scheduled batch


class PipelineAction(str, Enum):
    """CRAG-style corrective actions."""

    GENERATE = "generate"  # High confidence - proceed
    DECOMPOSE = "decompose"  # Medium - break down query
    WEB_SEARCH = "web_search"  # Low - fresh research
    RETRY = "retry"  # Quality failed - retry
    DEAD_LETTER = "dead_letter"  # Max retries - DLQ


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Industry tiers for scheduled research
INDUSTRY_SCHEDULE = {
    "tier_1_weekly": {
        "industries": ["law_firms", "dental_practices", "marketing_agencies"],
        "cron": "0 3 * * MON",
        "priority": ResearchPriority.HIGH,
    },
    "tier_2_biweekly": {
        "industries": ["real_estate", "ecommerce", "saas", "accounting"],
        "cron": "0 3 1,15 * *",
        "priority": ResearchPriority.MEDIUM,
    },
    "tier_3_monthly": {
        "industries": [
            "construction",
            "restaurants",
            "insurance",
            "manufacturing",
            "healthcare",
        ],
        "cron": "0 3 1 * *",
        "priority": ResearchPriority.LOW,
    },
}

# Research query templates
INDUSTRY_QUERIES = {
    "comprehensive": [
        "{industry} business top pain points time wasters 2025",
        "{industry} automation opportunities ROI case studies",
        "{industry} decision makers software buying budget process",
        "{industry} common objections technology adoption concerns how to overcome",
        "{industry} competitors automation software pricing comparison gaps",
        "{industry} terminology jargon glossary professionals use daily",
    ],
    "gap_fill": [
        "{topic} in {industry} detailed explanation best practices",
        "{topic} {industry} statistics data 2025",
    ],
    "news_update": [
        "{industry} news regulations technology changes {current_month} 2025",
    ],
}

# Quality gates
QUALITY_GATES = {
    "completeness": {
        "min_pain_points": 5,
        "min_automation_opportunities": 4,
        "min_objections": 3,
        "min_roi_data_points": 3,
        "min_conversation_starters": 3,
    },
    "specificity": {
        "required_number_density": 0.3,
        "required_dollar_amounts": 2,
        "required_percentages": 3,
    },
    "actionability": {
        "objections_must_have_counters": True,
        "pain_points_must_have_costs": True,
        "opportunities_must_have_roi": True,
    },
}


# =============================================================================
# DATA MODELS (Pydantic Schemas)
# =============================================================================


class PainPoint(BaseModel):
    """Industry pain point with cost impact."""

    issue: str = Field(..., description="The pain point description")
    cost_impact: str = Field(..., description="Financial impact (e.g., '$200/hour lost')")
    quote: Optional[str] = Field(None, description="Supporting quote from research")
    source: Optional[str] = Field(None, description="Source URL")


class AutomationOpportunity(BaseModel):
    """Automation opportunity with ROI."""

    opportunity: str = Field(..., description="What can be automated")
    roi: str = Field(..., description="Expected ROI or time savings")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Implementation difficulty")
    implementation_time: Optional[str] = Field(None, description="Estimated implementation time")


class DecisionMaker(BaseModel):
    """Target decision maker profile."""

    title: str = Field(..., description="Job title (e.g., 'Practice Manager')")
    budget_authority: str = Field(..., description="Budget range they control")
    triggers: List[str] = Field(default_factory=list, description="What triggers them to buy")
    pain_points: List[str] = Field(default_factory=list, description="Their specific pain points")


class ObjectionHandler(BaseModel):
    """Sales objection with counter script."""

    objection: str = Field(..., description="The objection text")
    counter_script: str = Field(..., description="How to respond")
    proof_points: List[str] = Field(default_factory=list, description="Evidence to support counter")


class Competitor(BaseModel):
    """Competitor analysis."""

    name: str = Field(..., description="Competitor name")
    pricing: Optional[str] = Field(None, description="Pricing info if known")
    gaps: List[str] = Field(default_factory=list, description="Gaps we can exploit")
    strengths: List[str] = Field(default_factory=list, description="Their strengths")


class ROIDataPoint(BaseModel):
    """ROI statistic with source."""

    metric: str = Field(..., description="What was measured")
    before: str = Field(..., description="Before automation")
    after: str = Field(..., description="After automation")
    source: Optional[str] = Field(None, description="Source URL or citation")


class IndustryKnowledge(BaseModel):
    """Complete structured knowledge for an industry."""

    industry: str = Field(..., description="Industry identifier")
    pain_points: List[PainPoint] = Field(default_factory=list)
    automation_opportunities: List[AutomationOpportunity] = Field(default_factory=list)
    decision_makers: List[DecisionMaker] = Field(default_factory=list)
    objections: List[ObjectionHandler] = Field(default_factory=list)
    competitors: List[Competitor] = Field(default_factory=list)
    roi_data: List[ROIDataPoint] = Field(default_factory=list)
    terminology: Dict[str, str] = Field(default_factory=dict)
    conversation_starters: List[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    quality_score: float = Field(0.0, ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)


class KnowledgeChunk(BaseModel):
    """A semantic chunk for vector storage."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="Chunk text content")
    chunk_type: Literal[
        "pain_point", "automation", "objection", "script", "terminology", "roi"
    ]
    industry: str
    source: str = "perplexity_research"
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    quality_score: float = 0.0
    citations: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None


class ResearchTask(BaseModel):
    """A research task in the queue."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    industry: str
    trigger: ResearchTrigger
    priority: ResearchPriority = ResearchPriority.MEDIUM
    topic: Optional[str] = None  # For gap-fill research
    queries: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    status: Literal["pending", "running", "completed", "failed", "dlq"] = "pending"
    error_message: Optional[str] = None


# =============================================================================
# LANGGRAPH STATE DEFINITION
# =============================================================================


class ResearchState(TypedDict):
    """LangGraph state for research pipeline."""

    # Task info
    task_id: str
    industry: str
    trigger: str
    priority: str

    # Research data flow
    queries: List[str]
    raw_research: List[Dict[str, Any]]  # Perplexity responses
    structured_data: Optional[Dict[str, Any]]  # Processed by Haiku
    chunks: List[Dict[str, Any]]  # Ready for embedding
    embeddings: List[List[float]]

    # Quality & Control
    quality_score: float
    validation_errors: List[str]
    action: str  # Current corrective action
    retry_count: int

    # Metadata
    tokens_used: Dict[str, int]  # perplexity, haiku
    duration_ms: int
    trace_id: str
    messages: Annotated[List[str], lambda x, y: x + y]  # Append-only log


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for external services.

    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject requests
    - HALF_OPEN: Testing recovery
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_requests: int = 1

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[float] = field(default=None)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    slog.log("INFO", "circuit_half_open", circuit=self.name)
                else:
                    raise CircuitBreakerOpenError(f"Circuit {self.name} is OPEN")

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_requests:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    slog.log("INFO", "circuit_closed", circuit=self.name)
            self.failure_count = 0

    async def _record_failure(self) -> None:
        """Record failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                slog.log("WARN", "circuit_reopened", circuit=self.name)
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                slog.log("WARN", "circuit_opened", circuit=self.name, failures=self.failure_count)

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


# =============================================================================
# EXTERNAL SERVICE CLIENTS
# =============================================================================


class PerplexityClient:
    """
    Perplexity AI API client for real-time web research.

    Model: sonar (latest Perplexity model)
    Features: Citations, recency filtering
    """

    BASE_URL = "https://api.perplexity.ai"
    MODEL = "sonar"

    def __init__(
        self,
        api_key: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.api_key = api_key
        self.circuit = circuit_breaker or CircuitBreaker(
            name="perplexity",
            failure_threshold=5,
            recovery_timeout=300,
        )
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    @tracer.start_as_current_span("perplexity_search")
    async def search(
        self,
        query: str,
        recency_filter: Literal["day", "week", "month", "year"] = "month",
    ) -> Dict[str, Any]:
        """
        Execute Perplexity search with citations.

        Returns:
            {
                "content": str,
                "citations": List[str],
                "tokens_used": int
            }
        """
        span = trace.get_current_span()
        span.set_attribute("query", query[:100])

        async def _do_search():
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.MODEL,
                    "messages": [{"role": "user", "content": query}],
                    "return_citations": True,
                    "search_recency_filter": recency_filter,
                },
            )
            response.raise_for_status()
            return response.json()

        try:
            result = await self.circuit.call(_do_search)
            content = result["choices"][0]["message"]["content"]
            citations = result.get("citations", [])
            tokens = result.get("usage", {}).get("total_tokens", 0)

            slog.log(
                "INFO",
                "perplexity_search_complete",
                query=query[:50],
                citations_count=len(citations),
                tokens=tokens,
            )

            return {
                "content": content,
                "citations": citations,
                "tokens_used": tokens,
            }

        except CircuitBreakerOpenError:
            span.set_status(Status(StatusCode.ERROR, "Circuit breaker open"))
            raise
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            slog.log("ERROR", "perplexity_search_failed", error=str(e), query=query[:50])
            raise

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class HaikuProcessor:
    """
    Claude Haiku 3.5 client for cost-efficient processing.

    Uses: Structured extraction, knowledge formatting
    Cost: ~$0.00025 per 1K tokens
    """

    BASE_URL = "https://api.anthropic.com"
    MODEL = "claude-3-5-haiku-20241022"

    def __init__(
        self,
        api_key: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.api_key = api_key
        self.circuit = circuit_breaker or CircuitBreaker(
            name="haiku",
            failure_threshold=3,
            recovery_timeout=60,
        )
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    @tracer.start_as_current_span("haiku_process")
    async def extract_structured(
        self,
        raw_research: str,
        industry: str,
    ) -> IndustryKnowledge:
        """
        Extract structured knowledge from raw research text.

        Uses tool_use for guaranteed JSON schema output.
        """
        span = trace.get_current_span()
        span.set_attribute("industry", industry)

        extraction_prompt = f"""You are a business intelligence analyst extracting structured data from research about the {industry} industry.

Analyze the following research and extract:
1. Pain points (with specific cost impacts - use $ amounts and percentages)
2. Automation opportunities (with ROI estimates)
3. Decision maker profiles (who buys, their budget, what triggers purchases)
4. Common objections and counter scripts
5. Competitor analysis
6. ROI statistics with sources
7. Industry terminology/jargon
8. Conversation starters for sales

Be SPECIFIC - include numbers, percentages, dollar amounts. Generic insights are worthless.

Research text:
{raw_research}"""

        # Tool definition for structured output
        tools = [
            {
                "name": "extract_industry_knowledge",
                "description": "Extract structured industry knowledge from research",
                "input_schema": IndustryKnowledge.model_json_schema(),
            }
        ]

        async def _do_extract():
            response = await self.client.post(
                "/v1/messages",
                json={
                    "model": self.MODEL,
                    "max_tokens": 4096,
                    "tools": tools,
                    "tool_choice": {"type": "tool", "name": "extract_industry_knowledge"},
                    "messages": [{"role": "user", "content": extraction_prompt}],
                },
            )
            response.raise_for_status()
            return response.json()

        try:
            result = await self.circuit.call(_do_extract)

            # Extract tool use response
            tool_use = None
            for block in result.get("content", []):
                if block.get("type") == "tool_use":
                    tool_use = block.get("input", {})
                    break

            if not tool_use:
                raise ValueError("No tool_use response from Haiku")

            # Parse into Pydantic model (remove industry if present to avoid duplicate)
            tool_use.pop("industry", None)
            knowledge = IndustryKnowledge(industry=industry, **tool_use)

            tokens = result.get("usage", {})
            total_tokens = tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0)

            slog.log(
                "INFO",
                "haiku_extraction_complete",
                industry=industry,
                pain_points=len(knowledge.pain_points),
                opportunities=len(knowledge.automation_opportunities),
                tokens=total_tokens,
            )

            return knowledge

        except CircuitBreakerOpenError:
            span.set_status(Status(StatusCode.ERROR, "Circuit breaker open"))
            raise
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            slog.log("ERROR", "haiku_extraction_failed", error=str(e), industry=industry)
            raise

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class EmbeddingService:
    """
    Embedding service using sentence-transformers.

    Model: all-MiniLM-L6-v2 (384 dimensions)
    Cost: $0 (local)
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSION = 384

    def __init__(self):
        self._model = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.MODEL_NAME)
            slog.log("INFO", "embedding_model_loaded", model=self.MODEL_NAME)

    @tracer.start_as_current_span("embed_batch")
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Embed a batch of texts."""
        self._load_model()

        # Run in thread pool to not block async loop
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, batch_size=batch_size).tolist(),
        )

        slog.log("INFO", "embedding_batch_complete", count=len(texts))
        return embeddings

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]


# =============================================================================
# VECTOR STORE (Qdrant)
# =============================================================================


class NexusVectorStore:
    """
    Qdrant vector store for Nexus knowledge.

    Collection: nexus_knowledge
    Dimensions: 384 (all-MiniLM-L6-v2)
    """

    COLLECTION_NAME = "nexus_knowledge"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
    ):
        self.client = AsyncQdrantClient(url=url, api_key=api_key)
        self._initialized = False

    async def initialize(self):
        """Create collection if not exists."""
        if self._initialized:
            return

        collections = await self.client.get_collections()
        exists = any(c.name == self.COLLECTION_NAME for c in collections.collections)

        if not exists:
            await self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EmbeddingService.DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            slog.log("INFO", "qdrant_collection_created", collection=self.COLLECTION_NAME)

        self._initialized = True

    @tracer.start_as_current_span("qdrant_upsert")
    async def upsert_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        """Upsert knowledge chunks to Qdrant."""
        await self.initialize()

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue

            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload={
                        "content": chunk.content,
                        "type": chunk.chunk_type,
                        "industry": chunk.industry,
                        "source": chunk.source,
                        "processed_at": chunk.processed_at.isoformat(),
                        "quality_score": chunk.quality_score,
                        "citations": chunk.citations,
                    },
                )
            )

        if points:
            await self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points,
            )

        slog.log("INFO", "qdrant_upsert_complete", count=len(points))
        return len(points)

    @tracer.start_as_current_span("qdrant_search")
    async def search(
        self,
        query_embedding: List[float],
        industry: Optional[str] = None,
        chunk_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Semantic search with optional filters."""
        await self.initialize()

        filter_conditions = []
        if industry:
            filter_conditions.append(
                FieldCondition(key="industry", match=MatchValue(value=industry))
            )
        if chunk_type:
            filter_conditions.append(
                FieldCondition(key="type", match=MatchValue(value=chunk_type))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        results = await self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit,
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                **r.payload,
            }
            for r in results
        ]

    async def get_freshness(self, industry: str) -> Optional[datetime]:
        """Get last update time for an industry."""
        await self.initialize()

        results = await self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="industry", match=MatchValue(value=industry))]
            ),
            limit=1,
            order_by="processed_at",
        )

        if results[0]:
            return datetime.fromisoformat(results[0][0].payload["processed_at"])
        return None


# =============================================================================
# QUALITY VALIDATOR
# =============================================================================


class QualityValidator:
    """
    Validates research quality against gates.

    Returns score 0-1 and list of issues.
    """

    @staticmethod
    def validate(knowledge: IndustryKnowledge) -> tuple[float, List[str]]:
        """
        Validate industry knowledge against quality gates.

        Returns:
            (score: float, issues: List[str])
        """
        issues = []
        scores = []

        # Completeness checks
        gates = QUALITY_GATES["completeness"]

        if len(knowledge.pain_points) < gates["min_pain_points"]:
            issues.append(
                f"Insufficient pain points: {len(knowledge.pain_points)}/{gates['min_pain_points']}"
            )
            scores.append(len(knowledge.pain_points) / gates["min_pain_points"])
        else:
            scores.append(1.0)

        if len(knowledge.automation_opportunities) < gates["min_automation_opportunities"]:
            issues.append(
                f"Insufficient automation opportunities: {len(knowledge.automation_opportunities)}/{gates['min_automation_opportunities']}"
            )
            scores.append(
                len(knowledge.automation_opportunities) / gates["min_automation_opportunities"]
            )
        else:
            scores.append(1.0)

        if len(knowledge.objections) < gates["min_objections"]:
            issues.append(
                f"Insufficient objections: {len(knowledge.objections)}/{gates['min_objections']}"
            )
            scores.append(len(knowledge.objections) / gates["min_objections"])
        else:
            scores.append(1.0)

        if len(knowledge.roi_data) < gates["min_roi_data_points"]:
            issues.append(
                f"Insufficient ROI data: {len(knowledge.roi_data)}/{gates['min_roi_data_points']}"
            )
            scores.append(len(knowledge.roi_data) / gates["min_roi_data_points"])
        else:
            scores.append(1.0)

        if len(knowledge.conversation_starters) < gates["min_conversation_starters"]:
            issues.append(
                f"Insufficient conversation starters: {len(knowledge.conversation_starters)}/{gates['min_conversation_starters']}"
            )
            scores.append(
                len(knowledge.conversation_starters) / gates["min_conversation_starters"]
            )
        else:
            scores.append(1.0)

        # Specificity checks (check for numbers/percentages)
        all_text = " ".join(
            [
                p.cost_impact
                for p in knowledge.pain_points
                if p.cost_impact
            ]
            + [
                o.roi
                for o in knowledge.automation_opportunities
                if o.roi
            ]
        )

        dollar_count = len(re.findall(r"\$[\d,]+", all_text))
        percent_count = len(re.findall(r"\d+%", all_text))

        spec_gates = QUALITY_GATES["specificity"]

        if dollar_count < spec_gates["required_dollar_amounts"]:
            issues.append(f"Need more $ figures: {dollar_count}/{spec_gates['required_dollar_amounts']}")
            scores.append(dollar_count / spec_gates["required_dollar_amounts"])
        else:
            scores.append(1.0)

        if percent_count < spec_gates["required_percentages"]:
            issues.append(f"Need more percentages: {percent_count}/{spec_gates['required_percentages']}")
            scores.append(percent_count / spec_gates["required_percentages"])
        else:
            scores.append(1.0)

        # Actionability checks
        action_gates = QUALITY_GATES["actionability"]

        if action_gates["objections_must_have_counters"]:
            without_counters = sum(
                1 for o in knowledge.objections if not o.counter_script
            )
            if without_counters > 0:
                issues.append(f"Objections without counters: {without_counters}")
                scores.append(
                    (len(knowledge.objections) - without_counters) / max(len(knowledge.objections), 1)
                )
            else:
                scores.append(1.0)

        # Calculate final score
        final_score = sum(scores) / len(scores) if scores else 0.0

        return round(final_score, 3), issues


# =============================================================================
# CHUNKER
# =============================================================================


class KnowledgeChunker:
    """
    Converts structured knowledge into semantic chunks for embedding.

    Chunk size: 150-300 words for optimal retrieval.
    """

    @staticmethod
    def chunk(knowledge: IndustryKnowledge) -> List[KnowledgeChunk]:
        """Convert IndustryKnowledge to KnowledgeChunks."""
        chunks = []
        industry = knowledge.industry

        # Pain point chunks
        for i, pp in enumerate(knowledge.pain_points):
            content = f"Pain Point for {industry}: {pp.issue}"
            if pp.cost_impact:
                content += f"\nCost Impact: {pp.cost_impact}"
            if pp.quote:
                content += f"\nInsight: {pp.quote}"

            chunks.append(
                KnowledgeChunk(
                    content=content,
                    chunk_type="pain_point",
                    industry=industry,
                    quality_score=knowledge.quality_score,
                    citations=[pp.source] if pp.source else [],
                )
            )

        # Automation opportunity chunks
        for opp in knowledge.automation_opportunities:
            content = f"Automation Opportunity for {industry}: {opp.opportunity}"
            content += f"\nExpected ROI: {opp.roi}"
            content += f"\nDifficulty: {opp.difficulty}"
            if opp.implementation_time:
                content += f"\nImplementation Time: {opp.implementation_time}"

            chunks.append(
                KnowledgeChunk(
                    content=content,
                    chunk_type="automation",
                    industry=industry,
                    quality_score=knowledge.quality_score,
                )
            )

        # Objection handler chunks
        for obj in knowledge.objections:
            content = f"Common Objection in {industry}: \"{obj.objection}\""
            content += f"\n\nRecommended Response: {obj.counter_script}"
            if obj.proof_points:
                content += "\n\nProof Points:"
                for proof in obj.proof_points:
                    content += f"\n- {proof}"

            chunks.append(
                KnowledgeChunk(
                    content=content,
                    chunk_type="objection",
                    industry=industry,
                    quality_score=knowledge.quality_score,
                )
            )

        # Conversation starter chunks
        if knowledge.conversation_starters:
            content = f"Conversation Starters for {industry}:\n"
            for starter in knowledge.conversation_starters:
                content += f"\n• {starter}"

            chunks.append(
                KnowledgeChunk(
                    content=content,
                    chunk_type="script",
                    industry=industry,
                    quality_score=knowledge.quality_score,
                )
            )

        # ROI data chunk
        if knowledge.roi_data:
            content = f"ROI Statistics for {industry} automation:\n"
            for roi in knowledge.roi_data:
                content += f"\n• {roi.metric}: {roi.before} → {roi.after}"
                if roi.source:
                    content += f" (Source: {roi.source})"

            chunks.append(
                KnowledgeChunk(
                    content=content,
                    chunk_type="roi",
                    industry=industry,
                    quality_score=knowledge.quality_score,
                )
            )

        # Terminology chunk
        if knowledge.terminology:
            content = f"Industry Terminology for {industry}:\n"
            for term, definition in knowledge.terminology.items():
                content += f"\n• {term}: {definition}"

            chunks.append(
                KnowledgeChunk(
                    content=content,
                    chunk_type="terminology",
                    industry=industry,
                    quality_score=knowledge.quality_score,
                )
            )

        slog.log(
            "INFO",
            "chunking_complete",
            industry=industry,
            chunk_count=len(chunks),
        )

        return chunks


# =============================================================================
# LANGGRAPH PIPELINE NODES
# =============================================================================


class ResearchPipeline:
    """
    LangGraph-style pipeline for research orchestration.

    Flow: PLAN → RESEARCH → PROCESS → VALIDATE → CHUNK → EMBED → INGEST → COMPLETE
          ↑                                │
          └───────── RETRY ←───────────────┘ (if quality < 0.8)
                       │
                       └──────→ DEAD_LETTER (after 3 retries)
    """

    def __init__(
        self,
        perplexity_client: PerplexityClient,
        haiku_processor: HaikuProcessor,
        embedding_service: EmbeddingService,
        vector_store: NexusVectorStore,
    ):
        self.perplexity = perplexity_client
        self.haiku = haiku_processor
        self.embedder = embedding_service
        self.vector_store = vector_store

    # -------------------------------------------------------------------------
    # Node: PLAN
    # -------------------------------------------------------------------------
    @tracer.start_as_current_span("node_plan")
    async def plan_node(self, state: ResearchState) -> ResearchState:
        """Generate research queries for the industry."""
        industry = state["industry"]
        trigger = ResearchTrigger(state["trigger"])

        queries = []

        if trigger == ResearchTrigger.GAP_DETECTED and state.get("topic"):
            # Gap-fill queries
            topic = state.get("topic", "")
            for template in INDUSTRY_QUERIES["gap_fill"]:
                queries.append(
                    template.format(industry=industry, topic=topic)
                )
        elif trigger == ResearchTrigger.NEWS_UPDATE:
            # News queries
            current_month = datetime.utcnow().strftime("%B")
            for template in INDUSTRY_QUERIES["news_update"]:
                queries.append(
                    template.format(industry=industry, current_month=current_month)
                )
        else:
            # Full comprehensive research
            for template in INDUSTRY_QUERIES["comprehensive"]:
                queries.append(template.format(industry=industry))

        state["queries"] = queries
        state["messages"].append(f"📋 PLAN: Generated {len(queries)} queries for {industry}")

        slog.log(
            "INFO",
            "plan_complete",
            industry=industry,
            query_count=len(queries),
            trace_id=state["trace_id"],
        )

        return state

    # -------------------------------------------------------------------------
    # Node: RESEARCH
    # -------------------------------------------------------------------------
    @tracer.start_as_current_span("node_research")
    async def research_node(self, state: ResearchState) -> ResearchState:
        """Execute Perplexity searches for all queries."""
        queries = state["queries"]
        industry = state["industry"]

        raw_results = []
        total_tokens = 0

        # Execute queries in parallel (max 3 concurrent)
        semaphore = asyncio.Semaphore(3)

        async def search_with_limit(query: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.perplexity.search(query)

        tasks = [search_with_limit(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                state["messages"].append(f"⚠️ RESEARCH: Query failed: {query[:50]}... - {result}")
                continue

            raw_results.append(
                {
                    "query": query,
                    "content": result["content"],
                    "citations": result["citations"],
                }
            )
            total_tokens += result["tokens_used"]

        state["raw_research"] = raw_results
        state["tokens_used"]["perplexity"] = total_tokens
        state["messages"].append(
            f"🔬 RESEARCH: Completed {len(raw_results)}/{len(queries)} queries, {total_tokens} tokens"
        )

        slog.log(
            "INFO",
            "research_complete",
            industry=industry,
            successful_queries=len(raw_results),
            total_queries=len(queries),
            tokens=total_tokens,
            trace_id=state["trace_id"],
        )

        return state

    # -------------------------------------------------------------------------
    # Node: PROCESS (Haiku extraction)
    # -------------------------------------------------------------------------
    @tracer.start_as_current_span("node_process")
    async def process_node(self, state: ResearchState) -> ResearchState:
        """Process raw research through Haiku for structured extraction."""
        raw_research = state["raw_research"]
        industry = state["industry"]

        if not raw_research:
            state["action"] = PipelineAction.RETRY.value
            state["messages"].append("❌ PROCESS: No raw research to process")
            return state

        # Combine all research text
        combined_text = ""
        all_citations = []

        for result in raw_research:
            combined_text += f"\n\n### Research on: {result['query']}\n{result['content']}"
            all_citations.extend(result.get("citations", []))

        # Extract structured knowledge
        knowledge = await self.haiku.extract_structured(combined_text, industry)
        knowledge.sources = list(set(all_citations))

        state["structured_data"] = knowledge.model_dump()
        state["messages"].append(
            f"⚙️ PROCESS: Extracted {len(knowledge.pain_points)} pain points, "
            f"{len(knowledge.automation_opportunities)} opportunities, "
            f"{len(knowledge.objections)} objections"
        )

        slog.log(
            "INFO",
            "process_complete",
            industry=industry,
            pain_points=len(knowledge.pain_points),
            opportunities=len(knowledge.automation_opportunities),
            trace_id=state["trace_id"],
        )

        return state

    # -------------------------------------------------------------------------
    # Node: VALIDATE
    # -------------------------------------------------------------------------
    @tracer.start_as_current_span("node_validate")
    async def validate_node(self, state: ResearchState) -> ResearchState:
        """Validate processed data against quality gates."""
        structured_data = state.get("structured_data")

        if not structured_data:
            state["quality_score"] = 0.0
            state["validation_errors"] = ["No structured data to validate"]
            state["action"] = PipelineAction.RETRY.value
            return state

        knowledge = IndustryKnowledge(**structured_data)
        score, issues = QualityValidator.validate(knowledge)

        state["quality_score"] = score
        state["validation_errors"] = issues

        # Determine action based on score
        if score >= 0.8:
            state["action"] = PipelineAction.GENERATE.value
            state["messages"].append(f"✅ VALIDATE: Score {score:.2f} - PASS")
        elif state["retry_count"] >= 3:
            state["action"] = PipelineAction.DEAD_LETTER.value
            state["messages"].append(f"💀 VALIDATE: Score {score:.2f} - MAX RETRIES - DLQ")
        else:
            state["action"] = PipelineAction.RETRY.value
            state["messages"].append(f"🔄 VALIDATE: Score {score:.2f} - RETRY needed")

        slog.log(
            "INFO",
            "validate_complete",
            industry=state["industry"],
            score=score,
            issues=issues,
            action=state["action"],
            trace_id=state["trace_id"],
        )

        return state

    # -------------------------------------------------------------------------
    # Node: CHUNK
    # -------------------------------------------------------------------------
    @tracer.start_as_current_span("node_chunk")
    async def chunk_node(self, state: ResearchState) -> ResearchState:
        """Convert structured data to semantic chunks."""
        structured_data = state.get("structured_data")

        if not structured_data:
            state["chunks"] = []
            return state

        knowledge = IndustryKnowledge(**structured_data)
        knowledge.quality_score = state["quality_score"]

        chunks = KnowledgeChunker.chunk(knowledge)
        state["chunks"] = [c.model_dump() for c in chunks]
        state["messages"].append(f"📦 CHUNK: Created {len(chunks)} semantic chunks")

        return state

    # -------------------------------------------------------------------------
    # Node: EMBED
    # -------------------------------------------------------------------------
    @tracer.start_as_current_span("node_embed")
    async def embed_node(self, state: ResearchState) -> ResearchState:
        """Generate embeddings for all chunks."""
        chunks_data = state.get("chunks", [])

        if not chunks_data:
            state["embeddings"] = []
            return state

        texts = [c["content"] for c in chunks_data]
        embeddings = await self.embedder.embed_batch(texts)

        # Attach embeddings to chunks
        for chunk_dict, embedding in zip(chunks_data, embeddings):
            chunk_dict["embedding"] = embedding

        state["chunks"] = chunks_data
        state["embeddings"] = embeddings
        state["messages"].append(f"🧮 EMBED: Generated {len(embeddings)} embeddings")

        return state

    # -------------------------------------------------------------------------
    # Node: INGEST
    # -------------------------------------------------------------------------
    @tracer.start_as_current_span("node_ingest")
    async def ingest_node(self, state: ResearchState) -> ResearchState:
        """Ingest chunks into Qdrant."""
        chunks_data = state.get("chunks", [])

        if not chunks_data:
            state["messages"].append("⚠️ INGEST: No chunks to ingest")
            return state

        chunks = [KnowledgeChunk(**c) for c in chunks_data]
        count = await self.vector_store.upsert_chunks(chunks)

        state["messages"].append(f"💾 INGEST: Stored {count} chunks in Qdrant")

        slog.log(
            "INFO",
            "ingest_complete",
            industry=state["industry"],
            chunk_count=count,
            trace_id=state["trace_id"],
        )

        return state

    # -------------------------------------------------------------------------
    # Router: Determine next node
    # -------------------------------------------------------------------------
    def route_after_validate(self, state: ResearchState) -> str:
        """Route based on validation result."""
        action = state.get("action", PipelineAction.GENERATE.value)

        if action == PipelineAction.DEAD_LETTER.value:
            return "dead_letter"
        elif action == PipelineAction.RETRY.value:
            return "retry"
        else:
            return "chunk"

    # -------------------------------------------------------------------------
    # Node: RETRY
    # -------------------------------------------------------------------------
    async def retry_node(self, state: ResearchState) -> ResearchState:
        """Handle retry logic with refined queries."""
        state["retry_count"] += 1
        issues = state.get("validation_errors", [])

        # Refine queries based on issues
        refined_queries = []
        industry = state["industry"]

        for issue in issues[:3]:  # Address top 3 issues
            if "pain points" in issue.lower():
                refined_queries.append(
                    f"{industry} specific pain points with dollar cost impact 2025"
                )
            if "automation" in issue.lower():
                refined_queries.append(
                    f"{industry} automation case studies ROI statistics real examples"
                )
            if "objections" in issue.lower():
                refined_queries.append(
                    f"{industry} sales objections technology resistance how to overcome scripts"
                )

        if refined_queries:
            state["queries"] = refined_queries

        state["messages"].append(
            f"🔄 RETRY: Attempt {state['retry_count']}/3 with {len(refined_queries)} refined queries"
        )

        slog.log(
            "WARN",
            "retry_triggered",
            industry=industry,
            retry_count=state["retry_count"],
            issues=issues,
            trace_id=state["trace_id"],
        )

        return state

    # -------------------------------------------------------------------------
    # Node: DEAD_LETTER
    # -------------------------------------------------------------------------
    async def dead_letter_node(self, state: ResearchState) -> ResearchState:
        """Handle failed tasks by moving to DLQ."""
        state["messages"].append("💀 DEAD_LETTER: Task moved to dead letter queue")

        slog.log(
            "ERROR",
            "task_dlq",
            industry=state["industry"],
            retry_count=state["retry_count"],
            quality_score=state["quality_score"],
            errors=state.get("validation_errors", []),
            trace_id=state["trace_id"],
        )

        # In production, would publish to RabbitMQ DLQ here
        return state

    # -------------------------------------------------------------------------
    # Execute Pipeline
    # -------------------------------------------------------------------------
    @tracer.start_as_current_span("execute_pipeline")
    async def execute(self, task: ResearchTask) -> ResearchState:
        """Execute the full research pipeline."""
        span = trace.get_current_span()
        trace_id = str(uuid.uuid4())
        span.set_attribute("trace_id", trace_id)
        span.set_attribute("industry", task.industry)

        start_time = time.time()

        # Initialize state
        state: ResearchState = {
            "task_id": task.id,
            "industry": task.industry,
            "trigger": task.trigger.value,
            "priority": task.priority.value,
            "queries": [],
            "raw_research": [],
            "structured_data": None,
            "chunks": [],
            "embeddings": [],
            "quality_score": 0.0,
            "validation_errors": [],
            "action": PipelineAction.GENERATE.value,
            "retry_count": 0,
            "tokens_used": {"perplexity": 0, "haiku": 0},
            "duration_ms": 0,
            "trace_id": trace_id,
            "messages": [f"🚀 Starting research for {task.industry}"],
        }

        try:
            # Execute pipeline stages
            max_iterations = 5  # Prevent infinite loops

            for iteration in range(max_iterations):
                # PLAN
                state = await self.plan_node(state)

                # RESEARCH
                state = await self.research_node(state)

                # PROCESS
                state = await self.process_node(state)

                # VALIDATE
                state = await self.validate_node(state)

                # Route based on validation
                next_node = self.route_after_validate(state)

                if next_node == "dead_letter":
                    state = await self.dead_letter_node(state)
                    break
                elif next_node == "retry":
                    state = await self.retry_node(state)
                    continue  # Retry loop
                else:
                    # Continue to CHUNK → EMBED → INGEST
                    state = await self.chunk_node(state)
                    state = await self.embed_node(state)
                    state = await self.ingest_node(state)
                    state["messages"].append("✅ COMPLETE: Pipeline finished successfully")
                    break

            state["duration_ms"] = int((time.time() - start_time) * 1000)

            slog.log(
                "INFO",
                "pipeline_complete",
                industry=task.industry,
                quality_score=state["quality_score"],
                duration_ms=state["duration_ms"],
                tokens_perplexity=state["tokens_used"]["perplexity"],
                trace_id=trace_id,
            )

            return state

        except Exception as e:
            state["messages"].append(f"❌ ERROR: {str(e)}")
            state["duration_ms"] = int((time.time() - start_time) * 1000)

            slog.log(
                "ERROR",
                "pipeline_failed",
                industry=task.industry,
                error=str(e),
                trace_id=trace_id,
            )

            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


# =============================================================================
# MAIN ORACLE CLASS
# =============================================================================


class NexusResearchOracle:
    """
    The Self-Improving Intelligence Engine.

    Orchestrates autonomous research for Nexus Brain.
    """

    def __init__(
        self,
        perplexity_api_key: str,
        anthropic_api_key: str,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
    ):
        # Initialize clients
        self.perplexity = PerplexityClient(perplexity_api_key)
        self.haiku = HaikuProcessor(anthropic_api_key)
        self.embedder = EmbeddingService()
        self.vector_store = NexusVectorStore(qdrant_url, qdrant_api_key)

        # Initialize pipeline
        self.pipeline = ResearchPipeline(
            perplexity_client=self.perplexity,
            haiku_processor=self.haiku,
            embedding_service=self.embedder,
            vector_store=self.vector_store,
        )

        # Task queue (in production, would be RabbitMQ)
        self.task_queue: deque[ResearchTask] = deque()
        self.dead_letter_queue: deque[ResearchTask] = deque()

        slog.log("INFO", "oracle_initialized")

    async def queue_research(
        self,
        industry: str,
        trigger: ResearchTrigger = ResearchTrigger.MANUAL,
        priority: ResearchPriority = ResearchPriority.MEDIUM,
        topic: Optional[str] = None,
    ) -> str:
        """Queue a research task."""
        task = ResearchTask(
            industry=industry,
            trigger=trigger,
            priority=priority,
            topic=topic,
        )

        # Insert based on priority
        if priority == ResearchPriority.CRITICAL:
            self.task_queue.appendleft(task)
        else:
            self.task_queue.append(task)

        slog.log(
            "INFO",
            "task_queued",
            task_id=task.id,
            industry=industry,
            trigger=trigger.value,
            priority=priority.value,
        )

        return task.id

    async def process_next_task(self) -> Optional[ResearchState]:
        """Process the next task in queue."""
        if not self.task_queue:
            return None

        task = self.task_queue.popleft()
        task.status = "running"
        task.started_at = datetime.utcnow()

        try:
            result = await self.pipeline.execute(task)

            if result.get("action") == PipelineAction.DEAD_LETTER.value:
                task.status = "dlq"
                self.dead_letter_queue.append(task)
            else:
                task.status = "completed"
                task.completed_at = datetime.utcnow()

            return result

        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            self.dead_letter_queue.append(task)
            raise

    async def run_scheduled_batch(
        self,
        tier: str = "tier_1_weekly",
    ) -> List[ResearchState]:
        """Run scheduled research for a tier."""
        schedule = INDUSTRY_SCHEDULE.get(tier)
        if not schedule:
            raise ValueError(f"Unknown tier: {tier}")

        results = []

        for industry in schedule["industries"]:
            await self.queue_research(
                industry=industry,
                trigger=ResearchTrigger.SCHEDULED,
                priority=schedule["priority"],
            )

        # Process all queued tasks
        while self.task_queue:
            result = await self.process_next_task()
            if result:
                results.append(result)

        return results

    async def detect_knowledge_gap(
        self,
        message: str,
        conversation_history: List[Dict],
    ) -> Optional[str]:
        """
        Detect knowledge gaps from conversation.

        Returns industry to research if gap detected.
        """
        # Simple pattern matching for unknown industries
        industry_patterns = {
            "plumbing": ["plumber", "plumbing", "pipe", "drain"],
            "hvac": ["hvac", "heating", "cooling", "air conditioning"],
            "landscaping": ["landscaping", "lawn", "gardening", "yard"],
            "photography": ["photographer", "photography", "photos", "wedding photos"],
            "fitness": ["gym", "fitness", "personal trainer", "workout"],
            "veterinary": ["vet", "veterinary", "animal", "pet"],
            "pharmacy": ["pharmacy", "pharmacist", "prescription"],
            "automotive": ["auto shop", "mechanic", "car repair", "dealership"],
        }

        message_lower = message.lower()

        for industry, patterns in industry_patterns.items():
            if any(p in message_lower for p in patterns):
                # Check if we have fresh knowledge
                freshness = await self.vector_store.get_freshness(industry)

                if freshness is None or (datetime.utcnow() - freshness).days > 30:
                    slog.log(
                        "INFO",
                        "gap_detected",
                        industry=industry,
                        freshness=str(freshness),
                    )
                    return industry

        return None

    async def search_knowledge(
        self,
        query: str,
        industry: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base."""
        query_embedding = await self.embedder.embed_single(query)

        results = await self.vector_store.search(
            query_embedding=query_embedding,
            industry=industry,
            limit=limit,
        )

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get oracle status."""
        return {
            "queued_tasks": len(self.task_queue),
            "dlq_depth": len(self.dead_letter_queue),
            "circuits": {
                "perplexity": self.perplexity.circuit.get_status(),
                "haiku": self.haiku.circuit.get_status(),
            },
        }

    async def close(self):
        """Cleanup resources."""
        await self.perplexity.close()
        await self.haiku.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import os

    async def main():
        # Initialize oracle
        oracle = NexusResearchOracle(
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )

        # Queue a test task
        task_id = await oracle.queue_research(
            industry="dental_practices",
            trigger=ResearchTrigger.MANUAL,
            priority=ResearchPriority.HIGH,
        )

        print(f"Queued task: {task_id}")

        # Process it
        result = await oracle.process_next_task()

        if result:
            print("\n" + "=" * 60)
            print("PIPELINE MESSAGES:")
            print("=" * 60)
            for msg in result["messages"]:
                print(msg)

            print(f"\nQuality Score: {result['quality_score']}")
            print(f"Duration: {result['duration_ms']}ms")
            print(f"Chunks Created: {len(result['chunks'])}")

        await oracle.close()

    asyncio.run(main())
