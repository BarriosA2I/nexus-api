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
from typing import AsyncGenerator, Dict, List
from datetime import datetime
from collections import defaultdict

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from ..config import settings
from ..schemas import (
    QdrantRAGHealth,
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
from ..services.nexus_brain import generate_nexus_response, get_nexus_brain
from ..services.nexus_rag import get_rag_client
from ..services.event_publisher import get_event_publisher
from ..utils.sse_helpers import sse_thinking, sse_delta, sse_final, sse_error

logger = logging.getLogger(__name__)


# =============================================================================
# INDUSTRY DETECTION (for event publishing)
# =============================================================================

def detect_industry_from_message(message: str, history: List[Dict] = None) -> str:
    """
    Detect industry from user message and conversation history.
    Returns industry key or None if not detected.
    """
    # Combine message with recent history for context
    text_to_analyze = message.lower()
    if history:
        for msg in history[-5:]:
            text_to_analyze += " " + msg.get("content", "").lower()

    industry_keywords = {
        "dental_practices": ["dental", "dentist", "orthodontist", "teeth", "hygienist"],
        "law_firms": ["law firm", "attorney", "lawyer", "legal", "paralegal"],
        "real_estate": ["realtor", "real estate", "broker", "property", "escrow"],
        "ecommerce": ["ecommerce", "shopify", "online store", "amazon", "woocommerce"],
        "healthcare": ["doctor", "clinic", "medical", "healthcare", "physician"],
        "marketing_agencies": ["marketing agency", "ad agency", "digital agency", "creative agency"],
        "saas": ["saas", "software company", "tech startup", "platform"],
        "accounting": ["accountant", "cpa", "bookkeeper", "tax", "financial advisor"],
        "construction": ["contractor", "construction", "plumber", "electrician", "hvac"],
        "restaurants": ["restaurant", "hotel", "hospitality", "bar", "cafe"],
    }

    for industry, keywords in industry_keywords.items():
        for keyword in keywords:
            if keyword in text_to_analyze:
                return industry

    return None

router = APIRouter(prefix="/api/nexus", tags=["nexus"])

# Server start time for uptime calculation
_start_time = time.time()

# Conversation memory store - keeps history per session
# Format: {session_id: [{"role": "user/assistant", "content": "..."}]}
conversation_store: Dict[str, List[Dict[str, str]]] = defaultdict(list)


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
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    LLM-powered RAG streaming generator with conversation memory.

    Uses Nexus Brain for intelligent, context-aware responses.
    Maintains conversation history per session for context continuity.
    Publishes conversation events for feedback loop.
    """
    start_time = time.time()
    detected_industry = None
    full_response = ""

    try:
        # Opening status
        yield sse_thinking("One sec...")
        await asyncio.sleep(0.1)

        # Get conversation history for this session
        history = conversation_store[session_id]

        # Detect industry from message and history (for event publishing)
        detected_industry = detect_industry_from_message(message, history)

        # Add user message to history
        history.append({"role": "user", "content": message})

        # Log the provider and history size
        brain = get_nexus_brain()
        logger.info(
            f"[{trace_id}] Using Nexus Brain ({brain.get_provider()}) | "
            f"History: {len(history)} | Industry: {detected_industry or 'unknown'}"
        )

        # Generate response using LLM brain with conversation history
        async for chunk in generate_nexus_response(message, conversation_history=history[:-1]):
            full_response += chunk
            yield sse_delta(chunk)
            await asyncio.sleep(0.02)

        # Save assistant response to history
        history.append({"role": "assistant", "content": full_response})

        # Keep history manageable (last 20 messages = 10 exchanges)
        if len(history) > 20:
            conversation_store[session_id] = history[-20:]

        # Determine next action based on message content
        message_lower = message.lower()
        next_action = "question"
        if any(term in message_lower for term in ["commercial", "video", "ad", "intake"]):
            next_action = "intake"
        elif any(term in message_lower for term in ["book", "call", "meeting", "schedule"]):
            next_action = "booking"

        # Final event
        yield sse_final(trace_id, next_action)

        # Publish conversation event for feedback loop (non-blocking)
        try:
            response_latency_ms = (time.time() - start_time) * 1000
            publisher = await get_event_publisher()
            await publisher.publish_conversation_event(
                conversation_id=session_id,
                user_message=message,
                nexus_response=full_response,
                detected_industry=detected_industry,
                confidence_score=1.0,  # TODO: Get from brain when RAG enabled
                response_latency_ms=response_latency_ms,
            )
        except Exception as pub_error:
            logger.debug(f"[{trace_id}] Event publish skipped: {pub_error}")

    except Exception as e:
        logger.error(f"[{trace_id}] Chat error: {e}")
        yield sse_error("Something went wrong. Let me try again.")
        yield sse_final(trace_id, "question")


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

    # =========================================================================
    # INDUSTRY-SPECIFIC RESPONSES - Check these FIRST before generic handlers
    # =========================================================================
    industry_examples = {
        # Healthcare
        "dental": (
            "For dental practices, we typically automate:\n\n"
            "• **Patient reminders** - Appointment confirmations, recall campaigns for cleanings\n"
            "• **Insurance verification** - Pull eligibility before patients arrive\n"
            "• **Review generation** - Automatically ask happy patients for Google reviews\n"
            "• **New patient intake** - Digital forms that sync to your practice management system\n\n"
            "What's eating up the most time for your front desk right now?"
        ),
        "medical": (
            "For medical practices, we typically automate:\n\n"
            "• **Patient intake** - Digital forms, insurance capture, consent signatures\n"
            "• **Appointment management** - Reminders, confirmations, waitlist fills\n"
            "• **Prior authorizations** - Automate the paperwork nightmare\n"
            "• **Follow-up workflows** - Post-visit surveys, care plan reminders\n\n"
            "What's the biggest admin headache in your practice right now?"
        ),
        "clinic": (
            "For clinics, we typically automate:\n\n"
            "• **Patient communication** - Reminders, follow-ups, recall campaigns\n"
            "• **Intake workflows** - Digital forms, insurance verification\n"
            "• **Review requests** - Automatically ask patients for feedback\n"
            "• **Reporting** - Pull data from your EHR without manual exports\n\n"
            "What's taking up too much of your staff's time?"
        ),
        # Real Estate
        "real estate": (
            "For real estate teams, we typically automate:\n\n"
            "• **Lead qualification** - Score and route leads 24/7\n"
            "• **Follow-up sequences** - Never let a lead go cold\n"
            "• **Market reports** - Auto-generate CMAs and neighborhood stats\n"
            "• **Transaction coordination** - Deadline tracking, document collection\n\n"
            "Are you looking to handle more leads or streamline your back office?"
        ),
        "realtor": (
            "For realtors, we typically automate:\n\n"
            "• **Lead follow-up** - Instant response, qualification, scheduling\n"
            "• **Market analysis** - Auto-generate CMAs when listings hit\n"
            "• **Client updates** - Keep buyers and sellers in the loop automatically\n"
            "• **Review requests** - Ask happy clients at closing\n\n"
            "What's your biggest bottleneck - lead gen or transaction management?"
        ),
        # Legal
        "law firm": (
            "For law firms, we typically automate:\n\n"
            "• **Intake qualification** - Screen potential clients 24/7\n"
            "• **Document assembly** - Generate standard docs from templates\n"
            "• **Client communication** - Case status updates, deadline reminders\n"
            "• **Research summaries** - Pull relevant case law faster\n\n"
            "What practice area are you in? That'll help me give better examples."
        ),
        "attorney": (
            "For attorneys, we typically automate:\n\n"
            "• **Client intake** - Qualify leads, collect docs, schedule consults\n"
            "• **Document drafting** - First drafts of standard agreements\n"
            "• **Deadline tracking** - Never miss a filing date\n"
            "• **Billing prep** - Time entry assistance and invoice generation\n\n"
            "Are you solo or part of a larger firm?"
        ),
        "lawyer": (
            "For law practices, we typically automate:\n\n"
            "• **Lead qualification** - Screen inquiries, book consults\n"
            "• **Document automation** - Generate contracts, letters, filings\n"
            "• **Client updates** - Keep clients informed without manual work\n"
            "• **Research assistance** - Summarize case law and precedents\n\n"
            "What type of law do you practice?"
        ),
        # E-commerce
        "ecommerce": (
            "For e-commerce brands, we typically automate:\n\n"
            "• **Customer service** - Handle FAQs, order status, returns\n"
            "• **Product descriptions** - Generate SEO-optimized copy at scale\n"
            "• **Inventory alerts** - Reorder notifications, stockout prevention\n"
            "• **Review management** - Respond to reviews, request new ones\n\n"
            "What platform are you on - Shopify, WooCommerce, something else?"
        ),
        "shopify": (
            "For Shopify stores, we typically automate:\n\n"
            "• **Customer support** - Answer questions, track orders, process returns\n"
            "• **Product content** - Descriptions, meta tags, alt text at scale\n"
            "• **Abandoned cart** - Smart recovery sequences beyond basic emails\n"
            "• **Review collection** - Post-purchase review requests\n\n"
            "What's your biggest challenge - traffic, conversion, or operations?"
        ),
        # Agency
        "agency": (
            "For agencies, we typically automate:\n\n"
            "• **Client reporting** - Pull data, generate insights, send automatically\n"
            "• **Content creation** - First drafts of blog posts, social, ads\n"
            "• **Competitive monitoring** - Track client competitors automatically\n"
            "• **Lead qualification** - Score and route inbound inquiries\n\n"
            "What kind of agency are you - marketing, creative, development?"
        ),
        # SaaS
        "saas": (
            "For SaaS companies, we typically automate:\n\n"
            "• **Lead qualification** - Score trials, identify high-intent users\n"
            "• **Onboarding** - Personalized sequences based on user behavior\n"
            "• **Churn prevention** - Identify at-risk accounts early\n"
            "• **Support triage** - Route tickets, suggest solutions\n\n"
            "What stage are you at - pre-revenue, scaling, or established?"
        ),
        "software": (
            "For software companies, we typically automate:\n\n"
            "• **Demo scheduling** - Qualify and book prospects 24/7\n"
            "• **User onboarding** - Behavior-triggered guidance\n"
            "• **Support** - Answer common questions, escalate complex ones\n"
            "• **Feedback analysis** - Summarize feature requests and bugs\n\n"
            "Are you B2B or B2C?"
        ),
        # Hospitality
        "restaurant": (
            "For restaurants, we typically automate:\n\n"
            "• **Reservations** - Handle bookings, confirmations, waitlist\n"
            "• **Review management** - Respond to reviews, request new ones\n"
            "• **Staff scheduling** - Optimize based on expected traffic\n"
            "• **Inventory/ordering** - Track usage, auto-reorder supplies\n\n"
            "What type of restaurant - casual, fine dining, fast casual?"
        ),
        "hotel": (
            "For hotels, we typically automate:\n\n"
            "• **Guest communication** - Pre-arrival, during stay, post-checkout\n"
            "• **Review management** - Respond and request reviews\n"
            "• **Upsell sequences** - Promote amenities and upgrades\n"
            "• **Competitive monitoring** - Track rates and availability\n\n"
            "What size property are you working with?"
        ),
        # Finance
        "accounting": (
            "For accounting firms, we typically automate:\n\n"
            "• **Document collection** - Chase clients for missing docs\n"
            "• **Data entry** - Extract from receipts, statements, invoices\n"
            "• **Client communication** - Deadline reminders, status updates\n"
            "• **Report generation** - Standard financial reports\n\n"
            "Is this for tax season prep or year-round bookkeeping?"
        ),
        "financial": (
            "For financial services, we typically automate:\n\n"
            "• **Client onboarding** - KYC, document collection, account setup\n"
            "• **Portfolio updates** - Auto-generate client reports\n"
            "• **Compliance monitoring** - Track regulatory requirements\n"
            "• **Lead qualification** - Score and route prospects\n\n"
            "What area - wealth management, lending, insurance?"
        ),
        # Construction/Trades
        "construction": (
            "For construction companies, we typically automate:\n\n"
            "• **Bid management** - Track opportunities, generate proposals\n"
            "• **Project updates** - Keep clients informed automatically\n"
            "• **Subcontractor coordination** - Scheduling, payments, docs\n"
            "• **Safety compliance** - Track certifications and training\n\n"
            "What type of construction - residential, commercial, specialty?"
        ),
        "contractor": (
            "For contractors, we typically automate:\n\n"
            "• **Lead follow-up** - Respond to inquiries, schedule estimates\n"
            "• **Proposal generation** - Create quotes from templates\n"
            "• **Project updates** - Keep homeowners in the loop\n"
            "• **Review requests** - Ask happy customers for referrals\n\n"
            "What trade are you in?"
        ),
        "plumber": (
            "For plumbing businesses, we typically automate:\n\n"
            "• **Dispatch optimization** - Route techs efficiently\n"
            "• **Customer follow-up** - Maintenance reminders, review requests\n"
            "• **Quote generation** - Standard pricing templates\n"
            "• **Lead response** - Answer inquiries 24/7\n\n"
            "Are you looking to grow or just run more efficiently?"
        ),
        "hvac": (
            "For HVAC companies, we typically automate:\n\n"
            "• **Maintenance reminders** - Seasonal tune-up campaigns\n"
            "• **Dispatch routing** - Optimize tech schedules\n"
            "• **Quote follow-up** - Chase pending proposals\n"
            "• **Review collection** - Ask customers after service\n\n"
            "What's your biggest pain point - lead gen or operations?"
        ),
    }

    # Check for industry mentions FIRST
    for industry, response in industry_examples.items():
        if industry in message_lower:
            return response

    # =========================================================================
    # GENERIC HANDLERS (only if no industry detected)
    # =========================================================================

    # SERVICES - Generic fallback
    if "service" in message_lower or "offer" in message_lower or "what do you do" in message_lower or "what can you" in message_lower:
        return (
            "We build automation systems that handle the stuff your team shouldn't be doing manually - "
            "market research, content creation, lead qualification, that kind of thing.\n\n"
            "What industry are you in? I can give you some specific examples of what we do."
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
            "What kind of business are you promoting?"
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
            "What industry are you in? I can give you some specific examples."
        )

    # BOOK/CALL/MEETING - Facilitate
    if "call" in message_lower or "meeting" in message_lower or "book" in message_lower or "schedule" in message_lower or "talk" in message_lower:
        return (
            "Happy to chat! The best way is to book a strategy call - we'll dig into your specific situation "
            "and figure out if there's a fit.\n\n"
            "You can book directly at **barriosa2i.com/book** - takes about 30 minutes."
        )

    # WHO/ABOUT - Company story
    if "who" in message_lower or "about" in message_lower or "company" in message_lower or "barrios" in message_lower:
        return (
            "Barrios A2I is a premium automation agency - we build custom systems that "
            "handle the work your team shouldn't be doing manually.\n\n"
            "Our clients are usually growing companies that need to scale without "
            "hiring a ton of people. What industry are you in?"
        )

    # DEFAULT - Ask for industry
    return (
        "Interesting - tell me more about your business. "
        "What industry are you in and what's eating up too much of your team's time?"
    )


async def generate_ragnarok_response(
    message: str,
    trace_id: str,
) -> AsyncGenerator[str, None]:
    """
    Sales-Safe video generation streaming.
    Uses human-friendly event types: meta, delta, final
    """
    job_store = get_job_store()

    try:
        yield sse_thinking("Setting up your video request...")
        await asyncio.sleep(0.2)

        # Extract brief from message
        brief = message

        # Submit job (internal)
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

        logger.info(f"[{trace_id}] Video job queued: {job.id}")

        # Sales-friendly response (no technical details)
        response = (
            f"Great! I've started your video request.\n\n"
            f"Your video will be ready in about 4-5 minutes. "
            f"We'll create a professional 30-second commercial based on your brief.\n\n"
            f"While you wait, is there anything else you'd like to know about our services?"
        )

        words = response.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield sse_delta(chunk)
            await asyncio.sleep(0.02)

        yield sse_final(trace_id, "intake")

    except Exception as e:
        logger.error(f"[{trace_id}] Video request error: {e}")
        yield sse_error("I had trouble setting up your video request. Let's try again.")
        yield sse_final(trace_id, "question")


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with SSE streaming.

    Automatically detects mode (RAG or RAGNAROK) based on message content.
    Maintains conversation memory per session_id for context continuity.
    """
    trace_id = generate_trace_id()

    # Get or generate session_id for conversation memory
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"

    logger.info(f"[{trace_id}] Chat request: {request.message[:50]}... | Session: {session_id}")

    # Detect mode
    mode = request.mode or detect_mode(request.message)
    logger.info(f"[{trace_id}] Mode: {mode.value}")

    # Select generator
    if mode == ChatMode.RAGNAROK:
        generator = generate_ragnarok_response(request.message, trace_id)
    else:
        generator = generate_rag_response(request.message, trace_id, session_id)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Trace-ID": trace_id,
            "X-Session-ID": session_id,
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

    # Get Qdrant RAG status
    qdrant_health = None
    try:
        rag_client = await get_rag_client()
        qdrant_health = QdrantRAGHealth(
            enabled=rag_client.enabled,
            connected=rag_client.enabled and rag_client._client is not None,
            collection=rag_client._cfg.collection if rag_client._cfg else None,
            embedder_loaded=rag_client._embedder is not None,
            error=None
        )
    except Exception as e:
        qdrant_health = QdrantRAGHealth(
            enabled=False,
            connected=False,
            collection=None,
            embedder_loaded=False,
            error=str(e)
        )

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
            qdrant=qdrant_health,
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
