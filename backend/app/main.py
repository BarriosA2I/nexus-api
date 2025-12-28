"""
Nexus Assistant Unified - Main Application
FastAPI application with lifecycle management

Production features:
- OpenTelemetry distributed tracing
- Prometheus metrics endpoint
- PostgreSQL checkpointing via LangGraph
- Circuit breaker protection
"""
import logging
import os
import random
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

# OpenTelemetry instrumentation (optional - graceful degradation)
_otel_enabled = False
if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        _otel_enabled = True
    except ImportError:
        pass
from .routers import nexus_router, ragnarok_router, intake_router, INTAKE_AVAILABLE
from .routers.creative_director import (
    router as creative_director_router,
    initialize_creative_director,
    shutdown_creative_director,
)
from .routers.unified_chat import router as unified_chat_router
from .services.rag_local import init_rag_service
from .services.nexus_brain import get_nexus_brain, initialize_nexus_brain_rag, get_brain_status
from .services.nexus_rag import get_rag_client
from .services.semantic_router import get_semantic_router
from .services.job_store import get_job_store
from .services.ragnarok_bridge import get_ragnarok_bridge, ragnarok_job_handler
from .services.circuit_breaker import circuit_registry
from .services.event_publisher import get_event_publisher, close_event_publisher

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# =============================================================================
# TRACE ID MIDDLEWARE
# =============================================================================

class TraceIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that propagates trace IDs through the request lifecycle.

    - Extracts X-Trace-Id from request headers
    - Generates one if missing: nxs_{timestamp}_{rand}
    - Stores in request.state.trace_id
    - Returns in response headers
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract or generate trace ID
        trace_id = request.headers.get("X-Trace-Id")
        if not trace_id:
            timestamp = int(time.time())
            rand_hex = format(random.randint(0, 0xFFFFFFFF), "08x")
            trace_id = f"nxs_{timestamp}_{rand_hex}"

        # Store in request state
        request.state.trace_id = trace_id

        # Process request
        response = await call_next(request)

        # Add trace ID to response headers
        response.headers["X-Trace-Id"] = trace_id

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown tasks:
    - Initialize RAG service
    - Start job store workers
    - Initialize Ragnarok bridge
    - Register job handlers
    """
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # Initialize RAG service
    logger.info("Initializing RAG service...")
    rag_service = init_rag_service()
    logger.info(f"RAG loaded: {rag_service.chunk_count} chunks from {len(rag_service.loaded_files)} files")

    # Initialize circuit breakers
    logger.info("Initializing circuit breakers...")
    circuit_registry.get_or_create("rag_service")
    circuit_registry.get_or_create("ragnarok_bridge")
    circuit_registry.get_or_create("llm_gateway")

    # Start job store
    logger.info("Starting job store...")
    job_store = get_job_store()
    job_store.register_handler("ragnarok_generate", ragnarok_job_handler)
    await job_store.start()

    # Initialize Ragnarok bridge
    logger.info("Initializing Ragnarok bridge...")
    ragnarok = get_ragnarok_bridge()
    await ragnarok.initialize()
    logger.info(f"Ragnarok mode: {ragnarok.mode.value}")

    # Initialize Nexus Brain (LLM)
    logger.info("Initializing Nexus Brain v4.0...")
    brain = get_nexus_brain()
    logger.info(f"Nexus Brain provider: {brain.get_provider()}")
    logger.info(f"Nexus Brain available: {brain.is_available()}")
    if not brain.is_available():
        logger.warning("Nexus Brain using FALLBACK mode - set ANTHROPIC_API_KEY for LLM responses")

    # Initialize Nexus RAG client (new standalone async client - preferred)
    import os
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        logger.info("Initializing Nexus RAG client...")
        try:
            rag_client = await get_rag_client()
            if rag_client.enabled:
                logger.info("✅ Nexus RAG client operational (Qdrant connected)")
            else:
                logger.info("⚠️ Nexus RAG client disabled (check QDRANT_URL)")
        except Exception as e:
            logger.warning(f"⚠️ Nexus RAG client unavailable: {e}")
    else:
        logger.info("Nexus RAG not configured (set QDRANT_URL to enable)")

    # Warm up Semantic Router (pre-compute industry vectors on startup)
    if qdrant_url:
        logger.info("Warming up Semantic Router...")
        try:
            semantic_router = await get_semantic_router()
            if semantic_router.is_ready:
                logger.info("✅ Semantic Router warmed up (16 industries embedded)")
            else:
                logger.info("⚠️ Semantic Router not ready (falling back to regex)")
        except Exception as e:
            logger.warning(f"⚠️ Semantic Router warmup failed: {e}")

    # Initialize legacy Research Oracle RAG (fallback - optional)
    redis_url = os.getenv("REDIS_URL")
    if qdrant_url or redis_url:
        logger.info("Initializing legacy Research Oracle RAG...")
        try:
            rag_enabled = await initialize_nexus_brain_rag(
                qdrant_url=qdrant_url,
                redis_url=redis_url,
                rabbitmq_url=os.getenv("RABBITMQ_URL")
            )
            if rag_enabled:
                logger.info("✅ Legacy Research Oracle RAG available")
            else:
                logger.info("⚠️ Legacy Research Oracle RAG partially available")
        except Exception as e:
            logger.warning(f"⚠️ Legacy Research Oracle RAG unavailable: {e}")

    # Initialize RabbitMQ event publisher (optional - graceful degradation)
    rabbitmq_url = os.getenv("RABBITMQ_URL")
    if rabbitmq_url:
        logger.info("Initializing RabbitMQ event publisher...")
        publisher = await get_event_publisher()
        if publisher.is_enabled:
            logger.info("✅ RabbitMQ event publishing enabled (feedback loop active)")
        else:
            logger.info("⚠️ RabbitMQ event publishing failed to connect")
    else:
        logger.info("Event publishing disabled (set RABBITMQ_URL to enable feedback loop)")

    # Initialize Creative Director (6-agent video pipeline)
    logger.info("Initializing Creative Director...")
    try:
        await initialize_creative_director(app)
        logger.info("Creative Director initialized (6-agent video pipeline ready)")
    except Exception as e:
        logger.warning(f"Creative Director unavailable: {e}")

    # Initialize SuperGraph (LangGraph orchestration with PostgreSQL checkpointing)
    logger.info("Initializing NEXUS SuperGraph...")
    try:
        from .orchestrator.supergraph import get_supergraph
        supergraph = await get_supergraph()
        if supergraph:
            logger.info("NEXUS SuperGraph initialized (LangGraph enabled)")
            if hasattr(supergraph, 'checkpointer') and supergraph.checkpointer:
                logger.info("PostgreSQL checkpointing enabled")
        else:
            logger.info("NEXUS SuperGraph using fallback orchestrator")
    except Exception as e:
        logger.warning(f"SuperGraph initialization failed (using fallback): {e}")

    logger.info("=" * 60)
    logger.info(f"Server ready at http://{settings.HOST}:{settings.PORT}")
    logger.info(f"API docs at http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down...")

    # Shutdown SuperGraph (closes PostgreSQL checkpointer)
    try:
        from .orchestrator.supergraph import shutdown_supergraph
        await shutdown_supergraph()
        logger.info("SuperGraph shutdown complete")
    except Exception as e:
        logger.warning(f"SuperGraph shutdown error: {e}")

    # Shutdown Creative Director
    await shutdown_creative_director()

    # Close event publisher
    await close_event_publisher()

    # Stop job store
    await job_store.stop()

    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Unified backend for Nexus Assistant with RAG and RAGNAROK integration",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Apply OpenTelemetry instrumentation (if configured)
if _otel_enabled:
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry FastAPI instrumentation enabled")
    except Exception as e:
        logger.warning(f"OpenTelemetry instrumentation failed: {e}")

# Add CORS middleware - LOCKED DOWN for production security
# P0-A: Use explicit allowed origins instead of wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,  # From CORS_ORIGINS env var
    allow_credentials=True,  # Enable for session cookies
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],  # Restrict to needed methods
    allow_headers=["Content-Type", "Authorization", "X-Trace-Id", "X-Session-ID"],
    expose_headers=["X-Trace-Id", "X-Session-ID"],
)

# Add trace ID middleware
app.add_middleware(TraceIdMiddleware)

# Include routers
app.include_router(nexus_router)
app.include_router(ragnarok_router)
# Mount Creative Director router at both paths for compatibility
# Primary: /api/creative-director (new standard)
# Legacy: /api/legendary (frontend compatibility)
app.include_router(creative_director_router, prefix="/api/creative-director", tags=["Creative Director"])
app.include_router(creative_director_router, prefix="/api/legendary", tags=["Creative Director Legacy"])

# Unified Chat API v2 - SuperGraph orchestration
app.include_router(unified_chat_router, prefix="/api/v2", tags=["Unified Chat"])

if INTAKE_AVAILABLE and intake_router:
    app.include_router(intake_router)
    logger.info("Intake router enabled")
else:
    logger.warning("Intake router disabled (missing dependencies)")


@app.get("/")
async def root():
    """Root endpoint - basic info"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


# Direct execution support
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
    )
