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
    Generate sales-focused conversational response.

    Personality: Friendly business consultant
    Goal: Understand their needs, guide toward strategy call
    Rules: No technical jargon, focus on business outcomes
    """
    message_lower = message.lower()

    # GREETINGS - Ask about their business first
    if "hello" in message_lower or "hey" in message_lower or message_lower.strip() in ["hi", "yo", "sup", "hola"]:
        return (
            "Hey! Welcome to Barrios A2I. I'm here to help you figure out if we're the right fit. "
            "What kind of work is eating up most of your team's time right now?"
        )

    # SERVICES - Focus on outcomes, not tech
    if "service" in message_lower or "offer" in message_lower or "what do you do" in message_lower or "what can you" in message_lower:
        return (
            "We build automation systems that handle the stuff your team shouldn't be doing manually - "
            "market research, content creation, lead qualification, that kind of thing.\n\n"
            "Most clients come to us when they're growing fast but can't hire fast enough. "
            "What's going on in your business that made you curious?"
        )

    # HOW IT WORKS - Deflect gracefully, protect IP
    if "how does" in message_lower or "how do you" in message_lower or "how it work" in message_lower or "explain" in message_lower:
        return (
            "Honestly, the *how* is less important than the *what* - we've spent years figuring out "
            "the technical stuff so you don't have to.\n\n"
            "What matters is: does it solve your problem and is it worth the investment? "
            "Tell me more about what you're trying to accomplish."
        )

    # TECHNICAL TERMS - Deflect any AI/tech jargon questions
    tech_terms = ["rag", "llm", "gpt", "neural", "vector", "embedding", "model", "algorithm",
                  "architecture", "api", "database", "machine learning", "artificial intelligence",
                  "circuit breaker", "cache", "pipeline"]
    if any(term in message_lower for term in tech_terms):
        return (
            "I could bore you with the technical details, but here's what actually matters: "
            "our systems work, they're reliable, and they get results.\n\n"
            "We've built these tools for clients across dozens of industries. "
            "What industry are you in? I can share some relevant examples."
        )

    # PRICING - Guide toward conversation
    if "pricing" in message_lower or "cost" in message_lower or "price" in message_lower or "how much" in message_lower or "expensive" in message_lower:
        return (
            "Depends on what we're building. Smaller projects start around $50K, "
            "enterprise solutions can go up to $300K. We also do equity partnerships for the right fit.\n\n"
            "But honestly, pricing only matters if we can actually solve your problem. "
            "What's the challenge you're facing?"
        )

    # VIDEO/COMMERCIAL - Focus on results
    if "video" in message_lower or "commercial" in message_lower or "ad" in message_lower:
        return (
            "We have a video production system that can turn around professional commercials in minutes, "
            "not weeks. Great for companies that need a lot of content fast.\n\n"
            "Are you looking to scale up your video marketing? What platforms are you targeting?"
        )

    # MARKETING - Results focused
    if "marketing" in message_lower or "content" in message_lower or "social" in message_lower:
        return (
            "We build systems that run your marketing while you sleep - content, social posts, "
            "ad campaigns, all coordinated and optimized automatically.\n\n"
            "Most clients see their team's time freed up by 60-70%. "
            "What's your current marketing setup like?"
        )

    # RESEARCH/INTELLIGENCE - Business outcomes
    if "research" in message_lower or "intelligence" in message_lower or "competitor" in message_lower or "market" in message_lower:
        return (
            "Think of it as having a team of analysts working 24/7 - finding opportunities, "
            "tracking competitors, surfacing insights your team would miss.\n\n"
            "What kind of research is your team doing manually right now?"
        )

    # WEBSITE - Focus on what it does
    if "website" in message_lower or "site" in message_lower:
        return (
            "Not just pretty pages - we build sites that actually have conversations with visitors, "
            "qualify leads, and book meetings for you. Like what we're doing right now.\n\n"
            "What's your current website doing for lead generation?"
        )

    # HELP/CAPABILITIES - Guide the conversation
    if "help" in message_lower or "can you" in message_lower:
        return (
            "I'm here to figure out if Barrios A2I can help your business. "
            "We specialize in automating the repetitive work that's slowing your team down.\n\n"
            "What brought you here today? I'd love to understand what you're dealing with."
        )

    # BOOK/CALL/MEETING - Facilitate
    if "call" in message_lower or "meeting" in message_lower or "book" in message_lower or "schedule" in message_lower or "talk" in message_lower:
        return (
            "Love to chat more! The best next step is a quick strategy call where we can "
            "dig into your specific situation.\n\n"
            "Head to **barriosa2i.com** and book a time that works for you. "
            "Or tell me more about your project and I can point you in the right direction."
        )

    # WHO/ABOUT - Company story
    if "who" in message_lower or "about" in message_lower or "company" in message_lower or "barrios" in message_lower:
        return (
            "Barrios A2I is a premium automation agency - we build custom systems that "
            "handle the work your team shouldn't be doing manually.\n\n"
            "Our clients are usually growing companies that need to scale without "
            "hiring a ton of people. Does that sound like your situation?"
        )

    # DEFAULT - Curious and helpful
    if context and sources:
        return (
            "Interesting question! I'd love to give you a proper answer - "
            "can you tell me a bit more about what you're trying to accomplish? "
            "That'll help me point you in the right direction."
        )

    return (
        "Hey, I'm here to help you figure out if Barrios A2I is the right fit for your business. "
        "We build automation systems that save companies serious time and money.\n\n"
        "What's the biggest bottleneck in your business right now?"
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
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Trace-ID": trace_id,
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
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
