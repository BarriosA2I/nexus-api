"""
Nexus Assistant Unified - Main Application
FastAPI application with lifecycle management
"""
import logging
import random
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings
from .routers import nexus_router, ragnarok_router
from .services.rag_local import init_rag_service
from .services.job_store import get_job_store
from .services.ragnarok_bridge import get_ragnarok_bridge, ragnarok_job_handler
from .services.circuit_breaker import circuit_registry

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

    logger.info("=" * 60)
    logger.info(f"Server ready at http://{settings.HOST}:{settings.PORT}")
    logger.info(f"API docs at http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down...")

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trace ID middleware
app.add_middleware(TraceIdMiddleware)

# Include routers
app.include_router(nexus_router)
app.include_router(ragnarok_router)


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
