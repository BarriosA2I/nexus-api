"""
Nexus Assistant Unified - Nexus Router
Chat SSE streaming and health endpoints
"""
import json
import time
import uuid
import asyncio
import logging
import re
from typing import AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from ..config import settings
from ..schemas import (
    ChatRequest,
    ChatMode,
    HealthResponse,
    RAGHealth,
    ComponentHealth,
    CircuitBreakerHealth,
    Source,
)
from ..services.rag_local import get_rag_service
from ..services.job_store import get_job_store
from ..services.ragnarok_bridge import get_ragnarok_bridge
from ..services.circuit_breaker import circuit_registry, CircuitBreakerError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/nexus", tags=["nexus"])

# Server start time for uptime calculation
_start_time = time.time()


def generate_trace_id() -> str:
    """Generate unique trace ID"""
    return f"nxs_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def sse_event(data: dict) -> str:
    """Format SSE event"""
    return f"data: {json.dumps(data)}\n\n"


def detect_mode(message: str) -> ChatMode:
    """
    Detect chat mode from message content.

    Returns RAGNAROK for video/commercial requests, RAG otherwise.
    """
    ragnarok_triggers = [
        r"\bcreate\b.*\b(commercial|video|ad)\b",
        r"\bgenerate\b.*\b(commercial|video|ad)\b",
        r"\bmake\b.*\b(commercial|video|ad)\b",
        r"\bproduce\b.*\b(commercial|video|ad)\b",
        r"\b(commercial|video|ad)\b.*\bfor\b",
        r"\bragnarok\b",
    ]

    message_lower = message.lower()
    for pattern in ragnarok_triggers:
        if re.search(pattern, message_lower):
            return ChatMode.RAGNAROK

    return ChatMode.RAG


async def generate_rag_response(
    message: str,
    trace_id: str,
) -> AsyncGenerator[str, None]:
    """
    Generate RAG-based response with SSE streaming.

    Steps:
    1. Circuit check
    2. Semantic search
    3. Context assembly
    4. Response generation (simulated LLM)
    5. Completion with sources
    """
    rag_service = get_rag_service()
    sources = []
    confidence = 0.0

    try:
        # Step 1: Circuit check
        yield sse_event({
            "type": "status",
            "step": "circuit_check",
            "message": "Checking neural pathways...",
            "trace_id": trace_id,
        })
        await asyncio.sleep(0.1)

        # Check circuit breaker
        rag_circuit = circuit_registry.get_or_create("rag_service")
        if rag_circuit.is_open:
            raise CircuitBreakerError("rag_service", rag_circuit.state)

        # Step 2: Semantic search
        yield sse_event({
            "type": "status",
            "step": "semantic_search",
            "message": "Searching knowledge base...",
            "trace_id": trace_id,
        })

        start_search = time.time()
        context, source_list = rag_service.get_context(message, max_tokens=2000)
        search_time = int((time.time() - start_search) * 1000)

        # Convert sources to Source objects
        sources = [
            Source(
                title=s["title"],
                chunk_id=s["chunk_id"],
                relevance=s["relevance"],
                excerpt=s["excerpt"],
                file=s.get("file"),
            )
            for s in source_list
        ]

        # Calculate confidence based on source relevance
        if sources:
            confidence = min(0.95, sum(s.relevance for s in sources) / len(sources) + 0.2)
        else:
            confidence = 0.3

        logger.info(f"[{trace_id}] RAG search: {len(sources)} sources in {search_time}ms")

        # Step 3: Generate response
        yield sse_event({
            "type": "status",
            "step": "generation",
            "message": "Synthesizing response...",
            "trace_id": trace_id,
        })

        # Generate response based on context
        response = await generate_contextual_response(message, context, sources)

        # Step 4: Stream response chunks
        words = response.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield sse_event({
                "type": "chunk",
                "content": chunk,
            })
            await asyncio.sleep(0.03 + (0.02 * (len(word) / 10)))  # Variable delay

        # Step 5: Completion
        yield sse_event({
            "type": "complete",
            "trace_id": trace_id,
            "confidence": round(confidence, 2),
            "sources": [s.model_dump() for s in sources],
            "mode": "rag",
            "tokens_used": len(response.split()) * 2,
            "latency_ms": int((time.time() - start_search) * 1000),
        })

    except CircuitBreakerError as e:
        yield sse_event({
            "type": "error",
            "code": "CIRCUIT_OPEN",
            "message": f"Service temporarily unavailable: {e.circuit_name}",
            "recoverable": True,
            "trace_id": trace_id,
        })

    except Exception as e:
        logger.error(f"[{trace_id}] RAG error: {e}")
        yield sse_event({
            "type": "error",
            "code": "RAG_ERROR",
            "message": str(e),
            "recoverable": True,
            "trace_id": trace_id,
        })


async def generate_contextual_response(
    message: str,
    context: str,
    sources: list,
) -> str:
    """
    Generate contextual response based on RAG context.

    This is a template-based response generator. In production,
    replace with actual LLM call (OpenAI, Anthropic, etc.)
    """
    message_lower = message.lower()

    # Check for specific topics
    if "cost" in message_lower and "reduction" in message_lower:
        return (
            "The **Nexus v2.1 architecture** achieves significant cost reduction through several mechanisms:\n\n"
            "1. **Intelligent Model Routing**: Automatically routes queries to the most cost-effective model "
            "(Haiku for simple queries, Sonnet for complex ones, Opus only when necessary)\n"
            "2. **Semantic Caching**: Identifies semantically similar queries to avoid redundant API calls, "
            "reducing costs by up to 70%\n"
            "3. **Circuit Breakers**: Prevent cascade failures that would otherwise waste API credits\n"
            "4. **Local TF-IDF RAG**: Zero-cost retrieval for knowledge base queries\n\n"
            "Combined, these optimizations can reduce operational costs by **60-70%** compared to naive implementations."
        )

    if "circuit breaker" in message_lower:
        return (
            "The **Circuit Breaker** pattern in Nexus follows the Netflix Hystrix model:\n\n"
            "**States:**\n"
            "- **CLOSED**: Normal operation, requests pass through\n"
            "- **OPEN**: After 5 consecutive failures, rejects all requests for 30 seconds\n"
            "- **HALF_OPEN**: After recovery timeout, allows limited requests to test recovery\n\n"
            "**Benefits:**\n"
            "- Prevents cascade failures across services\n"
            "- Provides graceful degradation\n"
            "- Allows automatic recovery\n"
            "- Protects downstream services from overload"
        )

    if "ragnarok" in message_lower or "video" in message_lower:
        return (
            "**RAGNAROK v7.0 APEX** is our 9-agent video generation system:\n\n"
            "**Capabilities:**\n"
            "- Generate commercial videos in ~243 seconds\n"
            "- Cost per commercial: ~$2.60\n"
            "- 97.5% success rate\n"
            "- Supports multiple platforms (YouTube, TikTok, LinkedIn)\n\n"
            "**To generate a commercial**, use a command like:\n"
            "*\"Create a 30-second commercial about AI automation for the technology industry\"*"
        )

    if "architecture" in message_lower or "system" in message_lower:
        return (
            "The **Nexus v2.1 Architecture** consists of several integrated components:\n\n"
            "**Core Services:**\n"
            "- **Local RAG**: TF-IDF based retrieval (zero API cost)\n"
            "- **Circuit Breakers**: Netflix Hystrix pattern for resilience\n"
            "- **Job Store**: Async job queue for long-running tasks\n"
            "- **RAGNAROK Bridge**: Video generation pipeline integration\n\n"
            "**API Layer:**\n"
            "- FastAPI with SSE streaming\n"
            "- RESTful endpoints for health, chat, and job management\n"
            "- CORS support for frontend integration"
        )

    # Default response based on context availability
    if context and sources:
        return (
            f"Based on the knowledge base, I found {len(sources)} relevant sources for your query.\n\n"
            f"The most relevant information indicates that {context[:500]}...\n\n"
            "Would you like me to elaborate on any specific aspect?"
        )

    return (
        "I'm **Nexus**, the intelligence interface for Barrios A2I. "
        "I can help you with:\n\n"
        "- **Architecture questions**: How the v2.1 system works\n"
        "- **Cost optimization**: How we achieve 70% cost reduction\n"
        "- **RAGNAROK**: Commercial video generation\n"
        "- **Deployment**: Service protocols and pricing\n\n"
        "What would you like to know more about?"
    )


async def generate_ragnarok_response(
    message: str,
    trace_id: str,
) -> AsyncGenerator[str, None]:
    """
    Handle RAGNAROK mode - queue video generation job.
    """
    job_store = get_job_store()

    try:
        yield sse_event({
            "type": "status",
            "step": "ragnarok_init",
            "message": "Initializing RAGNAROK pipeline...",
            "trace_id": trace_id,
        })
        await asyncio.sleep(0.2)

        # Extract brief from message
        brief = message

        # Submit job
        job = await job_store.submit(
            job_type="ragnarok_generate",
            payload={
                "brief": brief,
                "industry": "technology",
                "duration_seconds": 30,
                "platform": "youtube_1080p",
            },
            metadata={"trace_id": trace_id},
        )

        yield sse_event({
            "type": "status",
            "step": "job_queued",
            "message": f"Job queued: {job.id}",
            "trace_id": trace_id,
        })

        # Stream response
        response = (
            f"I've queued your commercial generation request.\n\n"
            f"**Job ID**: `{job.id}`\n"
            f"**Status**: {job.status.value}\n\n"
            f"You can check the status at:\n"
            f"`GET /api/nexus/ragnarok/jobs/{job.id}`\n\n"
            f"The RAGNAROK pipeline will process your request:\n"
            f"- Brief: *{brief[:100]}{'...' if len(brief) > 100 else ''}*\n"
            f"- Duration: 30 seconds\n"
            f"- Platform: YouTube 1080p\n"
            f"- Estimated time: ~4 minutes"
        )

        words = response.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield sse_event({
                "type": "chunk",
                "content": chunk,
            })
            await asyncio.sleep(0.02)

        yield sse_event({
            "type": "complete",
            "trace_id": trace_id,
            "confidence": 1.0,
            "sources": [],
            "mode": "ragnarok",
            "job_id": job.id,
        })

    except Exception as e:
        logger.error(f"[{trace_id}] RAGNAROK error: {e}")
        yield sse_event({
            "type": "error",
            "code": "RAGNAROK_ERROR",
            "message": str(e),
            "recoverable": True,
            "trace_id": trace_id,
        })


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with SSE streaming.

    Automatically detects mode (RAG or RAGNAROK) based on message content.
    """
    trace_id = generate_trace_id()
    logger.info(f"[{trace_id}] Chat request: {request.message[:50]}...")

    # Detect mode
    mode = request.mode or detect_mode(request.message)
    logger.info(f"[{trace_id}] Mode: {mode.value}")

    # Select generator
    if mode == ChatMode.RAGNAROK:
        generator = generate_ragnarok_response(request.message, trace_id)
    else:
        generator = generate_rag_response(request.message, trace_id)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Trace-ID": trace_id,
        },
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns comprehensive system health including:
    - Overall status (ONLINE, DEGRADED, OFFLINE)
    - Component health (RAG, job store, Ragnarok)
    - Circuit breaker states
    """
    rag_service = get_rag_service()
    job_store = get_job_store()
    ragnarok = get_ragnarok_bridge()

    # Determine overall status
    rag_healthy = rag_service.is_loaded
    job_healthy = job_store.is_running
    ragnarok_healthy = ragnarok.is_initialized

    if rag_healthy and job_healthy:
        status = "ONLINE"
    elif rag_healthy or job_healthy:
        status = "DEGRADED"
    else:
        status = "OFFLINE"

    # Get circuit breaker states
    circuit_states = []
    for name, cb in circuit_registry.all().items():
        circuit_states.append(CircuitBreakerHealth(
            name=name,
            state=cb.state.value,
            failure_count=cb.stats.consecutive_failures,
            last_failure=datetime.fromtimestamp(cb.stats.last_failure_time)
                if cb.stats.last_failure_time else None,
        ))

    return HealthResponse(
        status=status,
        uptime_seconds=round(time.time() - _start_time, 2),
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
        rag=RAGHealth(
            loaded=rag_service.is_loaded,
            chunks=rag_service.chunk_count,
            knowledge_files=rag_service.loaded_files,
            load_time_ms=rag_service.load_time_ms,
        ),
        job_queue=ComponentHealth(
            status="running" if job_store.is_running else "stopped",
            details=job_store.health_info(),
        ),
        ragnarok=ComponentHealth(
            status="available" if ragnarok.is_initialized else "unavailable",
            details=ragnarok.health_info(),
        ),
        circuit_breakers=circuit_states,
    )
