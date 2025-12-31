# NEXUS SuperGraph - Production Dockerfile
# Multi-stage build for minimal image size

# ============================================================================
# STAGE 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install production extras
RUN pip install --no-cache-dir \
    langgraph \
    langgraph-checkpoint-postgres \
    "psycopg[binary,pool]" \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-instrumentation-fastapi \
    prometheus_client \
    gunicorn \
    uvloop \
    httptools

# ============================================================================
# STAGE 2: Production
# ============================================================================
FROM python:3.11-slim as production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY backend/app ./app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash nexus && \
    chown -R nexus:nexus /app
USER nexus

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Expose port (Render sets PORT dynamically, default 10000)
EXPOSE 10000

# Health check - use localhost:$PORT but Docker build needs static value
# Render's own health check will use /api/v2/health on the assigned port
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/api/v2/health || exit 1

# Start with Gunicorn + Uvicorn workers
# CRITICAL: Use $PORT from Render (defaults to 10000 if not set)
CMD gunicorn app.main:app \
    --bind 0.0.0.0:${PORT:-10000} \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance
